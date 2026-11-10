"""El vigilante: relanza lo que se cae, no toca lo que funciona ni lo que se detuvo a propósito, y sabe rendirse."""
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

import riesgo_cuenta
import vigilante as vg
from activos import Activo

AHORA = 1_800_000_000.0
ORDEN = ["python.exe", "demo_exness_bot.py", "--simbolo", "XAUUSDm", "--volumen", "0.01"]


def crear_activo(tmp_path, simbolo="XAUUSDm") -> Activo:
    return Activo(simbolo, simbolo, 202602, 0.01, 750, simbolo, tmp_path / f"{simbolo}.pkl", tmp_path / simbolo,
                  tmp_path / f"{simbolo}.lock", tmp_path / f"panel_{simbolo}.txt")


def publicar(activo, momento, estado="en_marcha"):
    activo.panel.write_text(f"estado={estado}\nactualizado={int(momento)}\n", encoding="utf-8")


class Escena:
    def __init__(self, tmp_path):
        self.ahora, self.ruta, self.vivos = AHORA, tmp_path / "vigilante.json", set()
        self.oro = crear_activo(tmp_path)
        self.lanzar, self.avisos, self.terminar = Mock(), [], Mock(return_value=True)
        self.vigilante = vg.Vigilante({self.oro.simbolo: self.oro}, self.ruta, self.lanzar, self.avisos.append,
                                      lambda: self.ahora, lambda bloqueo: bloqueo in self.vivos, self.terminar)

    def pasan(self, segundos):
        self.ahora += segundos
        self.vigilante.revisar()


@pytest.fixture
def escena(tmp_path):
    return Escena(tmp_path)


# ---------------- la lista de bots que deben estar en marcha ----------------
def test_apuntar_y_borrar(tmp_path):
    ruta = tmp_path / "v.json"
    assert vg.leer_deseados(ruta) == {}
    vg.desear(ruta, "XAUUSDm", ORDEN, None)
    vg.desear(ruta, "DE30m", ORDEN, AHORA)
    vg.dejar_de_desear(ruta, "XAUUSDm")
    vg.dejar_de_desear(ruta, "no_estaba")
    assert vg.leer_deseados(ruta) == {"DE30m": {"orden": ORDEN, "fin": AHORA}}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("contenido", ["", "{", "[]", '{"XAUUSDm": 5}', '{"XAUUSDm": {"orden": "x"}}'])
def test_una_lista_ilegible_se_trata_como_vacia(tmp_path, contenido):
    ruta = tmp_path / "v.json"
    ruta.write_text(contenido)
    assert vg.leer_deseados(ruta) == {}


def test_una_sesion_con_duracion_se_relanza_con_lo_que_le_queda():
    entrada = {"orden": [*ORDEN, "--horas", "5"], "fin": AHORA + 2 * 3600}
    assert vg.orden_para_relanzar(entrada, AHORA)[-2:] == ["--horas", "2.0000"]
    assert entrada["orden"][-1] == "5"  # no modifica lo apuntado
    assert vg.orden_para_relanzar({"orden": ORDEN, "fin": None}, AHORA) == ORDEN
    assert vg.horas_de(ORDEN) is None and vg.horas_de([*ORDEN, "--horas", "5"]) == 5.0


def test_el_bloqueo_tomado_delata_un_proceso_vivo(tmp_path):
    bloqueo = tmp_path / "bot.lock"
    assert not vg.tiene_el_bloqueo(bloqueo)
    with riesgo_cuenta.turno(bloqueo, espera_s=0):
        assert vg.tiene_el_bloqueo(bloqueo)
    assert not vg.tiene_el_bloqueo(bloqueo)


# ---------------- revisar ----------------
def test_un_bot_con_latido_no_se_toca(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    publicar(escena.oro, AHORA - 5)
    escena.pasan(0)
    escena.lanzar.assert_not_called()
    assert escena.avisos == []


def test_un_bot_caido_se_relanza_con_sus_ajustes_y_se_avisa(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    publicar(escena.oro, AHORA - 300)
    escena.oro.carpeta.mkdir()
    escena.oro.salida.write_text("Traceback...\nMemoryError: sin memoria\n", encoding="utf-8")
    escena.pasan(0)
    escena.lanzar.assert_called_once_with(escena.oro, ORDEN)
    assert "se relanzó (intento 1 de 5)" in escena.avisos[0] and "MemoryError: sin memoria" in escena.avisos[0]


def test_tras_reiniciar_el_equipo_se_lanza_lo_apuntado_aunque_dijera_detenido(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    publicar(escena.oro, AHORA - 5, estado="detenido")  # el bot salió por su cuenta con un error
    escena.pasan(0)
    escena.lanzar.assert_called_once()


def test_lo_que_no_esta_apuntado_no_se_lanza(escena):
    publicar(escena.oro, AHORA - 300)  # se detuvo desde la ventana: ya no está en la lista
    escena.pasan(0)
    escena.lanzar.assert_not_called()


def test_recien_relanzado_se_le_da_tiempo_a_arrancar(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    escena.pasan(0)
    escena.pasan(vg.ARRANQUE_S - 1)  # todavía sin latido, pero puede estar cargando el modelo
    assert escena.lanzar.call_count == 1
    escena.pasan(2)
    assert escena.lanzar.call_count == 2


def test_un_proceso_vivo_sin_latido_se_avisa_una_vez_y_no_se_relanza(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    publicar(escena.oro, AHORA - 300)
    escena.vivos.add(escena.oro.bloqueo)
    escena.pasan(0)
    escena.pasan(10)
    escena.lanzar.assert_not_called()
    assert len(escena.avisos) == 1 and "sigue vivo pero no da señales desde hace 300 s" in escena.avisos[0]
    publicar(escena.oro, escena.ahora + 10)
    escena.pasan(10)
    assert "vuelve a responder" in escena.avisos[1]


def test_si_cae_una_y_otra_vez_se_rinde_avisa_y_lo_borra_de_la_lista(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    for _ in range(vg.MAX_REINTENTOS + 1):
        escena.pasan(vg.ARRANQUE_S * 2 ** vg.MAX_REINTENTOS)
    assert escena.lanzar.call_count == vg.MAX_REINTENTOS
    assert "no se relanza más" in escena.avisos[-1] and vg.leer_deseados(escena.ruta) == {}
    escena.pasan(10_000)
    assert escena.lanzar.call_count == vg.MAX_REINTENTOS


def test_un_relanzamiento_que_aguanta_pone_los_reintentos_a_cero(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    escena.pasan(0)                       # cae y se relanza: intento 1
    publicar(escena.oro, escena.ahora + vg.ESTABLE_S)
    escena.pasan(vg.ESTABLE_S)            # sigue en marcha: se olvida el intento
    escena.pasan(vg.MAX_LATIDO_S + 1)     # vuelve a caer mucho después
    assert "intento 1 de" in escena.avisos[-1] and escena.lanzar.call_count == 2


def test_una_sesion_terminada_deja_de_vigilarse_sin_relanzar(escena):
    vg.desear(escena.ruta, "XAUUSDm", [*ORDEN, "--horas", "5"], AHORA + 100)
    publicar(escena.oro, AHORA)
    escena.pasan(101)
    escena.lanzar.assert_not_called()
    assert vg.leer_deseados(escena.ruta) == {} and escena.avisos == []


def test_una_sesion_a_medias_se_relanza_solo_por_lo_que_queda(escena):
    vg.desear(escena.ruta, "XAUUSDm", [*ORDEN, "--horas", "5"], AHORA + 3600)
    escena.pasan(0)
    assert escena.lanzar.call_args.args[1][-2:] == ["--horas", "1.0000"]


def test_un_activo_desconocido_en_la_lista_se_ignora(escena):
    vg.desear(escena.ruta, "BTCUSDm", ORDEN, None)
    escena.pasan(0)
    escena.lanzar.assert_not_called()


def test_solo_puede_haber_un_vigilante(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, "RUTA", tmp_path / "vigilante.json")
    monkeypatch.setattr(vg, "LOGS", tmp_path)
    monkeypatch.setattr(vg.logging, "basicConfig", lambda **_k: None)
    monkeypatch.setattr(vg.logging, "FileHandler", Mock())
    with riesgo_cuenta.turno(tmp_path / "vigilante.lock", espera_s=0):
        with pytest.raises(SystemExit, match="Ya hay un vigilante"):
            vg.main()


def test_lanzar_arranca_el_proceso_y_guarda_su_salida(tmp_path):
    activo = crear_activo(tmp_path)
    proceso = vg.lanzar(activo, [sys.executable, "-c", "print('hola desde el bot')"])
    assert isinstance(proceso, subprocess.Popen) and proceso.wait(timeout=30) == 0
    assert vg.ultima_linea(activo.salida) == "hola desde el bot"
    assert json.loads(json.dumps(vg.leer_panel(activo.panel))) == {}


def colgar(escena, pid="4321"):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    escena.oro.panel.write_text(f"estado=en_marcha\nactualizado={int(AHORA - 300)}\npid={pid}\n", encoding="utf-8")
    escena.vivos.add(escena.oro.bloqueo)
    escena.pasan(0)


def test_un_bot_colgado_se_termina_pasado_el_plazo_y_luego_se_relanza(escena):
    colgar(escena)
    escena.pasan(vg.MATAR_COLGADO_S - 10)
    escena.terminar.assert_not_called()
    escena.pasan(10)
    escena.terminar.assert_called_once_with(escena.oro, 4321)
    assert "se terminó su proceso" in escena.avisos[-1]
    escena.pasan(10)
    assert escena.terminar.call_count == 1  # no se repite en cada revisión
    escena.vivos.clear()  # el proceso murió y soltó su bloqueo
    escena.pasan(10)
    escena.lanzar.assert_called_once()
    assert "se relanzó" in escena.avisos[-1]


@pytest.mark.parametrize("pid,se_puede", [("4321", False), ("", True), ("abc", True)])
def test_si_no_se_puede_terminar_se_avisa_una_vez_y_no_se_insiste(escena, pid, se_puede):
    escena.terminar.return_value = se_puede
    colgar(escena, pid)
    escena.pasan(vg.MATAR_COLGADO_S)
    escena.pasan(vg.MATAR_COLGADO_S)
    assert escena.terminar.call_count == (1 if pid == "4321" else 0)
    assert [a for a in escena.avisos if "cerrarlo a mano" in a] == [escena.avisos[-1]]
    escena.lanzar.assert_not_called()


def test_solo_se_termina_el_proceso_que_es_el_bot_de_ese_activo(tmp_path):
    oro, matar = crear_activo(tmp_path), Mock()
    del_oro = r"C:\venv\python.exe C:\bot\demo_exness_bot.py --simbolo XAUUSDm --volumen 0.01"
    for linea in ("", "notepad.exe", del_oro.replace("XAUUSDm", "US500m"), r"python.exe otro.py --simbolo XAUUSDm"):
        assert not vg.terminar(oro, 4321, lambda pid: linea, matar)
    matar.assert_not_called()
    assert vg.terminar(oro, 4321, lambda pid: del_oro, matar)
    matar.assert_called_once()
    matar.side_effect = OSError("acceso denegado")
    assert not vg.terminar(oro, 4321, lambda pid: del_oro, matar)


def test_la_linea_de_ordenes_de_este_mismo_proceso_se_puede_leer():
    import os
    assert "pytest" in vg.linea_de_ordenes(os.getpid())
    assert vg.linea_de_ordenes(0x7FFFFFF0) == ""  # no existe


def test_el_resumen_toca_una_vez_al_cambiar_el_dia_de_los_modelos(tmp_path):
    ruta, cierre = tmp_path / "resumen_dia.txt", 1794261600  # 2026-11-09 22:00 UTC = 17:00 en Nueva York
    assert vg.dia_de_los_modelos(cierre) == vg.dia_de_los_modelos(cierre - 1) + 1
    assert not vg.resumen_pendiente(ruta, cierre - 3600)  # primer arranque: solo apunta el día
    assert not vg.resumen_pendiente(ruta, cierre - 60)
    assert vg.resumen_pendiente(ruta, cierre + 5)
    assert not vg.resumen_pendiente(ruta, cierre + 15)


def test_el_resumen_cuenta_las_operaciones_de_las_ultimas_24_horas(tmp_path):
    import csv
    from datetime import datetime

    import trading_bot

    oro, dax, plata = crear_activo(tmp_path), crear_activo(tmp_path, "DE30m"), crear_activo(tmp_path, "XAGUSDm")
    activos = {a.simbolo: a for a in (oro, dax, plata)}
    assert vg.texto_resumen(activos, {}, AHORA) is None

    def fecha(hace_s):
        return datetime.fromtimestamp(AHORA - hace_s).isoformat(timespec="seconds")

    filas = [{"fecha": fecha(90000), "evento": "cierre", "ticket": "1", "beneficio": "100"},  # de ayer: no cuenta
             {"fecha": fecha(7200), "evento": "Apertura LONG", "ticket": "2", "tipo": "BUY", "slippage_puntos": "1", "retcode": "10009"},
             {"fecha": fecha(3600), "evento": "cierre", "ticket": "2", "beneficio": "-5.02", "swap": "0"}]
    oro.carpeta.mkdir()
    with open(oro.carpeta / "operaciones.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=trading_bot.CAMPOS_OPERACIONES)
        escritor.writeheader()
        escritor.writerows(filas)
    publicar(dax, AHORA - 5)
    lineas = vg.texto_resumen(activos, {"DE30m": {}, "XAGUSDm": {}}, AHORA).splitlines()
    assert lineas[1:] == ["XAUUSDm: detenido · 1 entrada, 1 cierre, -5.02", "DE30m: en marcha · sin operaciones",
                          "XAGUSDm: sin responder · sin operaciones", "Resultado neto: -5.02"]


def test_el_resumen_lleva_el_balance_y_la_equity_mas_recientes(tmp_path):
    oro, dax = crear_activo(tmp_path), crear_activo(tmp_path, "DE30m")
    activos, deseados = {a.simbolo: a for a in (oro, dax)}, {"XAUUSDm": {}, "DE30m": {}}

    def publicar_cuenta(activo, hace_s, balance, equity):
        activo.panel.write_text(f"estado=en_marcha\nactualizado={int(AHORA)}\nbalance={balance}\nequity={equity}\nmoneda=USD\n"
                                f"cuenta_leida={int(AHORA - hace_s)}\n", encoding="utf-8")

    publicar(oro, AHORA)
    publicar(dax, AHORA)
    assert vg.texto_resumen(activos, deseados, AHORA).splitlines()[-1] == "Resultado neto: +0.00"  # nadie publicó la cuenta
    publicar_cuenta(oro, 120, "10000.00", "10010.00")
    publicar_cuenta(dax, 5, "9994.98", "9990.00")
    assert vg.texto_resumen(activos, deseados, AHORA).splitlines()[-1] == "Balance 9994.98 · equity 9990.00 (flotante -4.98) USD"
    publicar_cuenta(oro, vg.CUENTA_RECIENTE_S + 200, "1", "1")
    publicar_cuenta(dax, vg.CUENTA_RECIENTE_S + 100, "2", "2")
    assert "Balance" not in vg.texto_resumen(activos, deseados, AHORA)  # demasiado antiguos para darlos por buenos


# ---------------- consultas por Telegram ----------------
def test_estado_dice_como_esta_cada_bot_y_lo_ultimo_que_dijo(tmp_path):
    oro, dax, plata = crear_activo(tmp_path), crear_activo(tmp_path, "DE30m"), crear_activo(tmp_path, "XAGUSDm")
    activos, ruta = {a.simbolo: a for a in (oro, dax, plata)}, tmp_path / "vigilante.json"
    assert vg.atender("/estado", activos, ruta, AHORA) == "Ningún bot en marcha"
    oro.panel.write_text(f"estado=en_marcha\nactualizado={int(AHORA)}\nnota=Sin señal\nnota_hora=10:15:00\nbalance=100\n"
                         f"equity=101\nmoneda=USD\ncuenta_leida={int(AHORA)}\n", encoding="utf-8")
    vg.desear(ruta, "DE30m", ORDEN, None)
    assert vg.atender(" /ESTADO@mi_bot ", activos, ruta, AHORA).splitlines() == [
        "XAUUSDm (XAUUSDm): en marcha · 10:15:00 Sin señal", "DE30m (DE30m): sin responder",
        "Balance 100.00 · equity 101.00 (flotante +1.00) USD"]


def test_resumen_y_ayuda(tmp_path):
    oro = crear_activo(tmp_path)
    activos, ruta = {oro.simbolo: oro}, tmp_path / "vigilante.json"
    assert "Sin bots apuntados" in vg.atender("/resumen", activos, ruta, AHORA)
    vg.desear(ruta, "XAUUSDm", ORDEN, None)
    assert "Resumen de las últimas 24 horas" in vg.atender("/resumen", activos, ruta, AHORA)
    for texto in ("hola", "/start", "/otra cosa", ""):
        assert vg.atender(texto, activos, ruta, AHORA) == vg.AYUDA


def test_detener_hace_lo_mismo_que_el_boton_de_la_ventana(tmp_path):
    oro, dax = crear_activo(tmp_path), crear_activo(tmp_path, "DE30m")
    activos, ruta = {a.simbolo: a for a in (oro, dax)}, tmp_path / "vigilante.json"
    for simbolo, activo in activos.items():
        vg.desear(ruta, simbolo, ORDEN, None)
        publicar(activo, AHORA)
    for sin_efecto, respuesta in (("/detener", "Indica cuál"), ("/detener EURUSDm", "Indica cuál")):
        assert respuesta in vg.atender(sin_efecto, activos, ruta, AHORA)
    assert not oro.detener.exists() and set(vg.leer_deseados(ruta)) == {"XAUUSDm", "DE30m"}

    assert "enviada a XAUUSDm" in vg.atender("/detener xauusd", activos, ruta, AHORA)
    assert oro.detener.exists() and not dax.detener.exists() and set(vg.leer_deseados(ruta)) == {"DE30m"}
    publicar(oro, AHORA, "detenido")
    oro.detener.unlink()
    assert "no está en marcha" in vg.atender("/detener XAUUSDm", activos, ruta, AHORA) and not oro.detener.exists()

    assert "enviada a DE30m" in vg.atender("/detener todos", activos, ruta, AHORA)
    assert dax.detener.exists() and vg.leer_deseados(ruta) == {}
    publicar(dax, AHORA, "detenido")
    assert vg.atender("/detener todos", activos, ruta, AHORA) == "Ningún bot en marcha"


def test_solo_se_atiende_el_chat_configurado():
    novedades = [{"update_id": 1, "message": {"chat": {"id": 42}, "text": "/estado"}},
                 {"update_id": 2, "message": {"chat": {"id": 666}, "text": "/detener todos"}},
                 {"update_id": 3, "message": {"chat": {"id": 42}, "photo": []}}, {"update_id": 4, "edited_message": {}}]
    assert vg.mensajes_nuevos(novedades, "42") == ["/estado"]


def test_la_escucha_ignora_lo_anterior_al_arranque_y_sobrevive_a_los_fallos():
    import alertas

    viejo = {"update_id": 7, "message": {"chat": {"id": 42}, "text": "/detener todos"}}
    nuevo = {"update_id": 8, "message": {"chat": {"id": 42}, "text": "/estado"}}
    roto = {"update_id": 9, "message": {"chat": {"id": 42}, "text": "/rompe"}}
    respuestas = [{"result": [viejo]}, alertas.ErrorTelegram("sin red"), {"result": [nuevo]}, {"ok": True},
                  {"result": [roto]}, {"ok": True}, {"result": []}]
    llamadas, atendidos, esperas = [], [], []

    def llamar(token, metodo, parametros):
        llamadas.append((metodo, parametros))
        respuesta = respuestas.pop(0)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    def atender(texto):
        atendidos.append(texto)
        if texto == "/rompe":
            raise RuntimeError("fallo")
        return "todo bien"

    vg.escuchar_telegram("token", "42", atender, llamar, esperas.append, seguir=lambda: bool(respuestas))
    assert atendidos == ["/estado", "/rompe"] and esperas == [30]
    assert llamadas[0] == ("getUpdates", {"offset": -1})
    assert [p["offset"] for m, p in llamadas[1:] if m == "getUpdates"] == [8, 8, 9, 10]
    enviados = [p["text"] for m, p in llamadas if m == "sendMessage"]
    assert enviados == ["todo bien", "No se pudo atender la consulta (RuntimeError)"]


def test_el_informe_semanal_toca_al_cierre_del_viernes():
    viernes_cierre = 1794607200  # 2026-11-13 22:00 UTC = 17:00 en Nueva York
    assert not vg.es_cierre_semanal(viernes_cierre - 60) and vg.es_cierre_semanal(viernes_cierre + 60)
    assert not vg.es_cierre_semanal(viernes_cierre + 86400 + 60)  # el sábado a esa hora ya es domingo para los modelos
    assert not vg.es_cierre_semanal(1794261600 + 60)  # un lunes


def test_el_informe_se_lanza_en_su_propio_proceso(monkeypatch, tmp_path):
    lanzado = Mock()
    monkeypatch.setattr(vg, "LOGS", tmp_path)
    monkeypatch.setattr(vg.subprocess, "Popen", lanzado)
    vg.lanzar_informe()
    orden = lanzado.call_args.args[0]
    assert orden[1].endswith("informe.py") and orden[2:] == ["--dias", "7", "--telegram"]


def test_informe_por_telegram_lanza_el_proceso_del_informe(tmp_path):
    lanzar = Mock()
    respuesta = vg.atender("/informe", {}, tmp_path / "vigilante.json", AHORA, lanzar)
    lanzar.assert_called_once_with()
    assert "Preparando el informe" in respuesta and "/informe" in vg.AYUDA
