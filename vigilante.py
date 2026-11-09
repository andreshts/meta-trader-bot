"""Mantiene en marcha los bots que la ventana dejó operando: los relanza si se caen y avisa por Telegram.

Uso: python vigilante.py   (o iniciar_vigilante.bat; Ctrl+C lo detiene, y los bots siguen)

La ventana (panel_control.py) apunta en `vigilante.json` cada bot que inicia, con su línea de órdenes, y lo borra
al detenerlo. El vigilante mira el latido que cada bot publica para el panel de MT5:

* Sin latido y sin proceso: lo relanza con los mismos ajustes (también tras reiniciar el equipo). Si cae
  MAX_REINTENTOS veces seguidas, avisa y deja de intentarlo.
* Sin latido pero con el proceso vivo: avisa de que no responde. No lo mata.
* Una sesión con duración se relanza solo por el tiempo que le quedaba, y al terminar deja de vigilarse.

Solo lanza procesos de demo_exness_bot.py: nunca envía órdenes ni toca posiciones.
"""
import json
import logging
import math
import os
import subprocess
import time
from pathlib import Path

import alertas
import config
import riesgo_cuenta
from activos import ACTIVOS, Activo
from cuenta import CUENTA, LOGS

RUTA = LOGS / "vigilante.json"
CREATE_NO_WINDOW = 0x08000000
INTERVALO_S = 10
MAX_LATIDO_S = 60     # el bot publica al menos cada 5 s; más margen que la ventana para no relanzar por una pausa
ARRANQUE_S = 90       # tiempo que se le da a un bot recién lanzado antes de juzgarlo; se duplica en cada reintento
ESTABLE_S = 300       # en marcha este tiempo tras un relanzamiento, los reintentos vuelven a cero
MAX_REINTENTOS = 5

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


# ---------------- el vigilante ----------------
class Vigilante:
    def __init__(self, activos: dict = ACTIVOS, ruta: Path = RUTA, lanzar=lanzar, avisar=log.warning, reloj=time.time,
                 vivo=tiene_el_bloqueo):
        self.activos, self.ruta, self.lanzar, self.avisar, self.reloj, self.vivo = activos, ruta, lanzar, avisar, reloj, vivo
        self.intentos = {}      # símbolo -> (relanzamientos seguidos, hora del último)
        self.colgados = set()   # ya se avisó de que no responden

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
            if simbolo in self.colgados:
                self.colgados.discard(simbolo)
                self.avisar(f"✅ {simbolo}: vuelve a responder")
            if intentos and ahora - lanzado >= ESTABLE_S:
                del self.intentos[simbolo]
            return
        if intentos and ahora - lanzado < ARRANQUE_S * 2 ** (intentos - 1):
            return  # recién relanzado: todavía puede estar arrancando
        if self.vivo(activo.bloqueo):
            if simbolo not in self.colgados:
                self.colgados.add(simbolo)
                self.avisar(f"⚠️ {simbolo}: el proceso sigue vivo pero no da señales desde hace {ahora - latido:.0f} s")
            return
        self.colgados.discard(simbolo)
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


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "vigilante.log", encoding="utf-8")])
    avisos = alertas.crear(f"vigilante · {CUENTA.tipo} {CUENTA.login}")

    def avisar(texto: str) -> None:
        log.warning(texto)
        if avisos is not None:
            avisos.avisar(texto)

    latido = RUTA.with_name("vigilante.txt")  # la ventana muestra si el vigilante está en marcha
    try:
        with riesgo_cuenta.turno(RUTA.with_name("vigilante.lock"), espera_s=0):
            vigilante = Vigilante(avisar=avisar)
            log.info("Vigilante en marcha: %s", ", ".join(leer_deseados(RUTA)) or "ningún bot apuntado todavía")
            while True:
                try:
                    vigilante.revisar()
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
