"""Mantiene en marcha los bots que la ventana dejó operando: los relanza si se caen y avisa por Telegram.

Uso: python vigilante.py   (o iniciar_vigilante.bat; Ctrl+C lo detiene, y los bots siguen)

La ventana (panel_control.py) apunta en `vigilante.json` cada bot que inicia, con su línea de órdenes, y lo borra
al detenerlo. El vigilante mira el latido que cada bot publica para el panel de MT5:

* Sin latido y sin proceso: lo relanza con los mismos ajustes (también tras reiniciar el equipo). Si cae
  MAX_REINTENTOS veces seguidas, avisa y deja de intentarlo.
* Sin latido pero con el proceso vivo: avisa de que no responde. Si sigue así MATAR_COLGADO_S, termina el proceso
  (solo si su línea de órdenes es la del bot de ese activo) y lo relanza. Sus posiciones conservan el stop y el objetivo.
* Al cierre de Nueva York envía por Telegram el resumen de las últimas 24 horas, leído de los operaciones.csv.
* Atiende las consultas que le llegan por Telegram desde el chat configurado: /estado, /resumen y /detener.
* Una sesión con duración se relanza solo por el tiempo que le quedaba, y al terminar deja de vigilarse.

Solo lanza y termina procesos de demo_exness_bot.py: nunca envía órdenes ni toca posiciones.
"""
import json
import logging
import math
import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import alertas
import config
import medicion
import riesgo_cuenta
from activos import ACTIVOS, Activo, hora_modelo
from cuenta import CUENTA, LOGS

RUTA = LOGS / "vigilante.json"
CREATE_NO_WINDOW = 0x08000000
INTERVALO_S = 10
MAX_LATIDO_S = 60     # el bot publica al menos cada 5 s; más margen que la ventana para no relanzar por una pausa
ARRANQUE_S = 90       # tiempo que se le da a un bot recién lanzado antes de juzgarlo; se duplica en cada reintento
ESTABLE_S = 300       # en marcha este tiempo tras un relanzamiento, los reintentos vuelven a cero
MAX_REINTENTOS = 5
EN_MARCHA = "en marcha"
SONDEO_TELEGRAM_S = 5  # cada cuánto vuelve Telegram con las manos vacías si nadie escribe
AYUDA = ("Consultas:\n/estado: cómo está cada bot y la cuenta\n/resumen: operaciones de las últimas 24 horas\n"
         "/detener SIMBOLO: detiene ese bot (o /detener todos); sus posiciones conservan el stop y el objetivo")
CUENTA_RECIENTE_S = 3600  # el resumen no da un balance ni una equity leídos hace más que esto
MATAR_COLGADO_S = 300  # un bot vivo pero sin latido durante este tiempo se termina y se relanza

log = logging.getLogger("vigilante")


# ---------------- lo que comparte con la ventana ----------------
def lanzar(activo: Activo, orden: list[str]) -> subprocess.Popen:
    """Arranca el bot sin consola. Lo que escriba al salir con error queda en `activo.salida`."""
    activo.carpeta.mkdir(parents=True, exist_ok=True)
    with open(activo.salida, "w", encoding="utf-8") as salida:
        return subprocess.Popen(orden, cwd=config.BASE_DIR, stdin=subprocess.DEVNULL, stdout=salida, stderr=subprocess.STDOUT,
                                creationflags=CREATE_NO_WINDOW, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def leer_panel(ruta: Path) -> dict:
    """Lo último que publicó el bot de ese activo; vacío si todavía no hay nada."""
    try:
        texto = ruta.read_text(encoding="utf-8")
    except (OSError, ValueError):  # el archivo se reemplaza justo en este momento
        return {}
    return dict(linea.split("=", 1) for linea in texto.splitlines() if "=" in linea)


def ultima_linea(ruta: Path) -> str:
    try:
        lineas = [l.strip() for l in ruta.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    except OSError:
        return ""
    return lineas[-1] if lineas else ""


def leer_deseados(ruta: Path) -> dict:
    """{símbolo: {"orden": [...], "fin": hora de fin o None}} de los bots que deben estar en marcha."""
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(datos, dict):
        return {}
    return {s: e for s, e in datos.items() if isinstance(e, dict) and isinstance(e.get("orden"), list)}


def _modificar(ruta: Path, cambio) -> None:
    """Lee, cambia y guarda de forma atómica, con turno: la ventana y el vigilante escriben el mismo archivo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with riesgo_cuenta.turno(ruta.with_name(ruta.name + ".lock"), espera_s=5):
        datos = leer_deseados(ruta)
        cambio(datos)
        temporal = ruta.with_name(f"{ruta.name}.{os.getpid()}.tmp")
        temporal.write_text(json.dumps(datos, indent=1), encoding="utf-8")
        os.replace(temporal, ruta)


def desear(ruta: Path, simbolo: str, orden: list[str], fin: float | None) -> None:
    """Apunta que ese bot debe estar en marcha, hasta `fin` (None = hasta que se detenga)."""
    _modificar(ruta, lambda datos: datos.update({simbolo: {"orden": orden, "fin": fin}}))


def dejar_de_desear(ruta: Path, simbolo: str) -> None:
    if simbolo in leer_deseados(ruta):
        _modificar(ruta, lambda datos: datos.pop(simbolo, None))


def horas_de(orden: list[str]) -> float | None:
    """La duración que lleva la línea de órdenes, o None si es una sesión sin límite."""
    return float(orden[orden.index("--horas") + 1]) if "--horas" in orden else None


def orden_para_relanzar(entrada: dict, ahora: float) -> list[str]:
    """La misma línea de órdenes; si la sesión tenía duración, solo con el tiempo que le queda."""
    orden = list(entrada["orden"])
    if entrada.get("fin") is not None and "--horas" in orden:
        orden[orden.index("--horas") + 1] = f"{max(0.01, (entrada['fin'] - ahora) / 3600):.4f}"
    return orden


def tiene_el_bloqueo(ruta: Path) -> bool:
    """True si un proceso tiene tomado ese bloqueo de instancia única, es decir, si el bot sigue vivo."""
    try:
        with riesgo_cuenta.turno(ruta, espera_s=0):
            return False
    except riesgo_cuenta.CuentaOcupada:
        return True


def linea_de_ordenes(pid: int) -> str:
    """La línea de órdenes de ese proceso; vacía si no existe o no se pudo leer."""
    consulta = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", consulta], capture_output=True, text=True, timeout=30,
                              stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def terminar(activo: Activo, pid: int, linea_de_ordenes=linea_de_ordenes, matar=os.kill) -> bool:
    """Termina el proceso solo si es el bot de ese activo: el número de un proceso que ya murió puede haber pasado a otro."""
    linea = linea_de_ordenes(pid)
    if "demo_exness_bot.py" not in linea or activo.simbolo not in linea.split():
        return False
    try:
        matar(pid, signal.SIGTERM)  # en Windows termina el proceso sin más; su bloqueo queda libre
    except OSError:
        return False
    return True


# ---------------- el resumen diario ----------------
def dia_de_los_modelos(ahora: float) -> int:
    """Cambia a las 00:00 de la hora de los modelos, es decir, al cierre de Nueva York."""
    return int(hora_modelo([int(ahora)])[0]) // 86400


def resumen_pendiente(ruta_dia: Path, ahora: float) -> bool:
    """True una vez por día de los modelos, al empezar el siguiente. El primer arranque solo apunta el día."""
    hoy = dia_de_los_modelos(ahora)
    try:
        ultimo = int(ruta_dia.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ultimo = None
    if ultimo == hoy:
        return False
    ruta_dia.write_text(str(hoy), encoding="utf-8")
    return ultimo is not None


def _cuantas(n: int, singular: str) -> str:
    return f"{n} {singular}" if n == 1 else f"{n} {singular}s"


def linea_de_la_cuenta(paneles: list, ahora: float) -> str | None:
    """Balance y equity según el bot que los leyó más recientemente; None si ninguno los publicó en la última hora."""
    def leida(panel: dict) -> float:
        try:
            return float(panel.get("cuenta_leida") or 0)
        except ValueError:
            return 0.0

    panel = max(paneles, key=leida, default={})
    try:
        balance, equity = float(panel["balance"]), float(panel["equity"])
    except (KeyError, ValueError):
        return None
    if ahora - leida(panel) > CUENTA_RECIENTE_S:
        return None
    return f"Balance {balance:.2f} · equity {equity:.2f} (flotante {equity - balance:+.2f}) {panel.get('moneda', '')}".strip()


def estado_del_bot(panel: dict, deseado: bool, ahora: float) -> str:
    try:
        responde = panel.get("estado") == "en_marcha" and ahora - float(panel.get("actualizado") or 0) <= MAX_LATIDO_S
    except ValueError:
        responde = False
    return EN_MARCHA if responde else "sin responder" if deseado else "detenido"


def texto_resumen(activos: dict, deseados: dict, ahora: float) -> str | None:
    """Estado de cada bot y sus operaciones de las últimas 24 horas; None si no hay bots apuntados ni operaciones."""
    desde = datetime.fromtimestamp(ahora - 86400).isoformat(timespec="seconds")
    lineas, total, paneles = [], 0.0, []
    for simbolo, activo in activos.items():
        m = medicion.medir(medicion.leer_filas(activo.carpeta / "operaciones.csv"), desde=desde)
        if simbolo not in deseados and not (m.entradas or m.cerradas):
            continue
        panel = leer_panel(activo.panel)
        paneles.append(panel)
        estado = estado_del_bot(panel, simbolo in deseados, ahora)
        if m.entradas or m.cerradas:
            detalle = f"{_cuantas(len(m.entradas), 'entrada')}, {_cuantas(len(m.cerradas), 'cierre')}, {m.resultado:+.2f}"
        else:
            detalle = "sin operaciones"
        lineas.append(f"{activo.nombre}: {estado} · {detalle}")
        total += m.resultado
    if not lineas:
        return None
    cuenta = linea_de_la_cuenta(paneles, ahora)
    return "\n".join(["📊 Resumen de las últimas 24 horas", *lineas, f"Resultado neto: {total:+.2f}", *([cuenta] if cuenta else [])])


# ---------------- consultas por Telegram ----------------
def texto_estado(activos: dict, deseados: dict, ahora: float) -> str:
    """Cómo está cada bot apuntado o en marcha, con lo último que dijo, y la cuenta."""
    lineas, paneles = [], []
    for simbolo, activo in activos.items():
        panel = leer_panel(activo.panel)
        estado = estado_del_bot(panel, simbolo in deseados, ahora)
        if estado == "detenido":
            continue
        paneles.append(panel)
        nota = f" · {panel.get('nota_hora', '')} {panel.get('nota', '')}".rstrip() if estado == EN_MARCHA and panel.get("nota") else ""
        lineas.append(f"{activo.nombre} ({simbolo}): {estado}{nota}")
    cuenta = linea_de_la_cuenta(paneles, ahora)
    return "\n".join([*(lineas or ["Ningún bot en marcha"]), *([cuenta] if cuenta else [])])


def detener_bots(activos: dict, ruta: Path, cual: str, ahora: float) -> str:
    """Lo mismo que el botón Detener de la ventana: deja de vigilarlo y le pide al bot que termine."""
    deseados = leer_deseados(ruta)
    en_marcha = [s for s, a in activos.items() if estado_del_bot(leer_panel(a.panel), s in deseados, ahora) != "detenido"]
    pedido = cual.strip().lower()
    elegidos = en_marcha if pedido == "todos" else [s for s in activos if pedido in (s.lower(), s.lower().removesuffix("m"))]
    if not pedido or (pedido != "todos" and not elegidos):
        return f"Indica cuál: /detener todos o /detener seguido de {', '.join(activos)}"
    elegidos = [s for s in elegidos if s in en_marcha]
    if not elegidos:
        return "Ningún bot en marcha" if pedido == "todos" else f"{cual.strip()} no está en marcha"
    for simbolo in elegidos:
        dejar_de_desear(ruta, simbolo)  # antes: que el vigilante no lo relance
        activos[simbolo].carpeta.mkdir(parents=True, exist_ok=True)
        activos[simbolo].detener.touch()
    return f"Orden de detener enviada a {', '.join(elegidos)}. Sus posiciones abiertas conservan el stop y el objetivo"


def atender(texto: str, activos: dict, ruta: Path, ahora: float) -> str:
    """La respuesta a un mensaje del chat."""
    orden, _, resto = texto.strip().partition(" ")
    orden = orden.lower().partition("@")[0]  # en un grupo, Telegram añade @nombre_del_bot
    if orden == "/estado":
        return texto_estado(activos, leer_deseados(ruta), ahora)
    if orden == "/resumen":
        return texto_resumen(activos, leer_deseados(ruta), ahora) or "Sin bots apuntados ni operaciones en las últimas 24 horas"
    if orden == "/detener":
        return detener_bots(activos, ruta, resto, ahora)
    return AYUDA


def mensajes_nuevos(novedades: list, chat: str) -> list:
    """Los textos que llegaron desde el chat configurado; lo de cualquier otro chat se ignora."""
    mensajes = (n.get("message") or {} for n in novedades)
    return [m["text"] for m in mensajes if str((m.get("chat") or {}).get("id")) == str(chat) and isinstance(m.get("text"), str)]


def escuchar_telegram(token: str, chat: str, atender, llamar=alertas._llamar, esperar=time.sleep, seguir=lambda: True) -> None:
    """Bucle de consultas. Lo que se escribió antes de arrancar no se atiende: un /detener viejo no debe parar nada."""
    siguiente = None
    while seguir():
        try:
            if siguiente is None:
                ultimas = llamar(token, "getUpdates", {"offset": -1}).get("result", [])
                siguiente = ultimas[-1]["update_id"] + 1 if ultimas else 0
                continue
            novedades = llamar(token, "getUpdates", {"offset": siguiente, "timeout": SONDEO_TELEGRAM_S}).get("result", [])
            if novedades:
                siguiente = novedades[-1]["update_id"] + 1
            for texto in mensajes_nuevos(novedades, chat):
                try:
                    respuesta = atender(texto)
                except Exception as e:
                    log.exception("Error al atender una consulta de Telegram")
                    respuesta = f"No se pudo atender la consulta ({type(e).__name__})"
                llamar(token, "sendMessage", {"chat_id": chat, "text": respuesta[:alertas.MAX_CARACTERES]})
        except Exception as e:
            log.debug("Consultas de Telegram: %s", e if isinstance(e, alertas.ErrorTelegram) else type(e).__name__)
            esperar(30)


# ---------------- el vigilante ----------------
class Vigilante:
    def __init__(self, activos: dict = ACTIVOS, ruta: Path = RUTA, lanzar=lanzar, avisar=log.warning, reloj=time.time,
                 vivo=tiene_el_bloqueo, terminar=terminar):
        self.activos, self.ruta, self.lanzar, self.avisar, self.reloj, self.vivo = activos, ruta, lanzar, avisar, reloj, vivo
        self.terminar = terminar
        self.intentos = {}      # símbolo -> (relanzamientos seguidos, hora del último)
        self.colgados = {}      # símbolo -> desde cuándo se le ve vivo pero sin latido (ya se avisó)
        self.sin_terminar = set()  # colgados que no se pudieron terminar: no se insiste

    def revisar(self) -> None:
        ahora = self.reloj()
        for simbolo, entrada in leer_deseados(self.ruta).items():
            activo = self.activos.get(simbolo)
            if activo is None:
                continue
            if entrada.get("fin") is not None and ahora >= entrada["fin"]:
                dejar_de_desear(self.ruta, simbolo)  # su sesión terminó: ya no se vigila
                self.intentos.pop(simbolo, None)
                continue
            self._revisar_bot(activo, entrada, ahora)

    def _revisar_bot(self, activo: Activo, entrada: dict, ahora: float) -> None:
        simbolo = activo.simbolo
        panel = leer_panel(activo.panel)
        try:
            latido = float(panel.get("actualizado") or 0)
        except ValueError:
            latido = 0.0
        intentos, lanzado = self.intentos.get(simbolo, (0, -math.inf))
        if panel.get("estado") == "en_marcha" and ahora - latido <= MAX_LATIDO_S:
            self.sin_terminar.discard(simbolo)
            if self.colgados.pop(simbolo, None) is not None:
                self.avisar(f"✅ {simbolo}: vuelve a responder")
            if intentos and ahora - lanzado >= ESTABLE_S:
                del self.intentos[simbolo]
            return
        if intentos and ahora - lanzado < ARRANQUE_S * 2 ** (intentos - 1):
            return  # recién relanzado: todavía puede estar arrancando
        if self.vivo(activo.bloqueo):
            if simbolo not in self.colgados:
                self.colgados[simbolo] = ahora
                self.avisar(f"⚠️ {simbolo}: el proceso sigue vivo pero no da señales desde hace {ahora - latido:.0f} s")
            elif ahora - self.colgados[simbolo] >= MATAR_COLGADO_S and simbolo not in self.sin_terminar:
                self._terminar_colgado(activo, panel, ahora)
            return
        self.colgados.pop(simbolo, None)
        self.sin_terminar.discard(simbolo)
        motivo = ultima_linea(activo.salida)
        if intentos >= MAX_REINTENTOS:
            dejar_de_desear(self.ruta, simbolo)
            del self.intentos[simbolo]
            self.avisar(f"🛑 {simbolo}: cayó {intentos} veces seguidas y no se relanza más. Último mensaje: {motivo or 'ninguno'}")
            return
        self.lanzar(activo, orden_para_relanzar(entrada, ahora))
        self.intentos[simbolo] = (intentos + 1, ahora)
        self.avisar(f"🔄 {simbolo}: no estaba en marcha y se relanzó (intento {intentos + 1} de {MAX_REINTENTOS})"
                    + (f". Último mensaje: {motivo}" if motivo else ""))


    def _terminar_colgado(self, activo: Activo, panel: dict, ahora: float) -> None:
        simbolo = activo.simbolo
        try:
            pid = int(panel.get("pid") or 0)
        except ValueError:
            pid = 0
        if pid and self.terminar(activo, pid):
            self.colgados[simbolo] = ahora  # si no muere, se vuelve a intentar pasado el mismo plazo
            self.avisar(f"🔪 {simbolo}: seguía sin responder y se terminó su proceso; se relanza a continuación")
        else:
            self.sin_terminar.add(simbolo)
            self.avisar(f"🛑 {simbolo}: sigue sin responder y no se pudo terminar su proceso; hay que cerrarlo a mano")


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "vigilante.log", encoding="utf-8")])
    avisos = alertas.crear(f"vigilante · {CUENTA.tipo} {CUENTA.login}")

    def avisar(texto: str) -> None:
        log.warning(texto)
        if avisos is not None:
            avisos.avisar(texto)

    token, chat = alertas.credenciales()
    if token is not None:
        threading.Thread(target=escuchar_telegram, daemon=True, name="consultas-telegram",
                         args=(token, chat, lambda texto: atender(texto, ACTIVOS, RUTA, time.time()))).start()
    latido = RUTA.with_name("vigilante.txt")  # la ventana muestra si el vigilante está en marcha
    try:
        with riesgo_cuenta.turno(RUTA.with_name("vigilante.lock"), espera_s=0):
            vigilante = Vigilante(avisar=avisar)
            log.info("Vigilante en marcha: %s", ", ".join(leer_deseados(RUTA)) or "ningún bot apuntado todavía")
            while True:
                try:
                    vigilante.revisar()
                    if resumen_pendiente(RUTA.with_name("resumen_dia.txt"), time.time()):
                        resumen = texto_resumen(ACTIVOS, leer_deseados(RUTA), time.time())
                        if resumen:
                            avisar(resumen)
                    latido.write_text(str(int(time.time())), encoding="utf-8")
                except Exception:
                    log.exception("Error al revisar los bots; se reintenta")
                time.sleep(INTERVALO_S)
    except riesgo_cuenta.CuentaOcupada:
        raise SystemExit("Ya hay un vigilante en marcha") from None
    except KeyboardInterrupt:
        log.info("Vigilante detenido; los bots siguen en marcha")
    finally:
        if avisos is not None:
            avisos.cerrar()


if __name__ == "__main__":
    main()
