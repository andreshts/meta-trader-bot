"""Avisos por Telegram: qué se avisa, qué se agrupa, y que un fallo de envío nunca llega al bot."""
import io
import logging
import sys
import urllib.error
from unittest.mock import Mock

import pytest

import alertas
import config
import demo_exness_bot as lanzador


class Reloj:
    def __init__(self):
        self.ahora = 0.0

    def __call__(self):
        return self.ahora


@pytest.fixture
def avisos():
    """(logger con el manejador, lista de mensajes enviados, reloj)."""
    enviados, reloj = [], Reloj()
    manejador = alertas.AlertasTelegram("XAUUSDm · demo 1", enviados.append, reloj)
    log = logging.Logger("prueba", logging.INFO)
    log.addHandler(manejador)

    def recibidos():
        manejador.cerrar(espera_s=5)
        return enviados
    return log, recibidos, reloj


def test_los_fragmentos_existen_en_los_mensajes_del_bot():
    """Si alguien cambia un mensaje del bot, este test avisa de que hay que actualizar alertas.py."""
    fuente = "".join((config.BASE_DIR / f).read_text(encoding="utf-8") for f in ("trading_bot.py", "demo_exness_bot.py"))
    for fragmento in alertas.SIEMPRE + alertas.BLOQUEOS + alertas.IGNORADOS:
        assert fragmento in fuente, fragmento


def test_cada_ejecucion_y_cada_cierre_se_avisan_con_su_etiqueta(avisos):
    log, recibidos, _ = avisos
    log.info("%s ejecutada: orden=%s volumen=%s precio=%s", "Apertura LONG", 1, 0.01, 4100.5)
    log.info("%s ejecutada: orden=%s volumen=%s precio=%s", "Apertura LONG", 2, 0.01, 4101.5)  # no se agrupan
    log.info("Deal de %s, posición %s: precio=%s beneficio=%s swap=%s comisión=%s tasas=%s (%s)", "cierre", 1, 4090.0, -10.5, 0, 0, 0, "sl")
    assert recibidos() == ["[XAUUSDm · demo 1] ℹ️ Apertura LONG ejecutada: orden=1 volumen=0.01 precio=4100.5",
                           "[XAUUSDm · demo 1] ℹ️ Apertura LONG ejecutada: orden=2 volumen=0.01 precio=4101.5",
                           "[XAUUSDm · demo 1] ℹ️ Deal de cierre, posición 1: precio=4090.0 beneficio=-10.5 swap=0 comisión=0 tasas=0 (sl)"]


def test_lo_rutinario_no_se_avisa(avisos):
    log, recibidos, _ = avisos
    log.info("P_long=%.3f P_short=%.3f", 0.2, 0.2)
    log.info("Vela nueva detectada %ss después de su apertura; se espera a la siguiente", 80)
    log.warning("Spread de %s puntos > %s; no se opera", 900, 750)
    log.warning("EXIGIR_VENTAJA_SIGNIFICATIVA = False: el modelo no ha demostrado ventaja")
    assert recibidos() == []


def test_un_error_repetido_se_avisa_una_vez_y_luego_con_su_cuenta(avisos):
    log, recibidos, reloj = avisos
    for _ in range(50):  # el ciclo reintenta cada 2 s
        try:
            raise RuntimeError("positions_get falló: (-10004, 'No IPC connection')")
        except RuntimeError:
            log.exception("Error en el ciclo; se reintenta en la siguiente vuelta")
    reloj.ahora = alertas.PAUSA_AVISOS_S + 1
    log.error("Error en el ciclo; se reintenta en la siguiente vuelta", exc_info=RuntimeError("positions_get falló: (-10004, 'No IPC connection')"))
    primero, segundo = recibidos()
    assert primero.endswith("❌ Error en el ciclo; se reintenta en la siguiente vuelta (RuntimeError: positions_get falló: (-10004, 'No IPC connection'))")
    assert "repetido 49 veces" in segundo


def test_dos_errores_distintos_no_se_tapan_entre_si(avisos):
    log, recibidos, _ = avisos
    log.error("%s fallida: retcode=%s comment=%s", "Apertura LONG", 10019, "No money")
    log.error("%s fallida: retcode=%s comment=%s", "Apertura LONG", 10018, "Market closed")
    log.error("%s fallida: retcode=%s comment=%s", "Apertura LONG", 10018, "Market closed")
    assert len(recibidos()) == 2


def test_un_limite_alcanzado_se_avisa_y_se_recuerda_cada_varias_horas(avisos):
    log, recibidos, reloj = avisos
    for equity in (8990.0, 8985.5, 8970.25):  # cada vela, con la equity moviéndose
        log.info("Sin nuevas entradas: %s", f"drawdown de 10.1% desde el máximo de equity 10000.00; equity {equity}")
    log.info("Sin nuevas entradas: %s", "pausa manual (existe logs/PAUSA)")  # otro motivo: aviso aparte
    reloj.ahora = alertas.PAUSA_BLOQUEOS_S + 1
    log.info("Sin nuevas entradas: %s", "drawdown de 10.4% desde el máximo de equity 10000.00; equity 8960.0")
    mensajes = recibidos()
    assert len(mensajes) == 3 and "drawdown" in mensajes[0] and "pausa manual" in mensajes[1]
    assert "repetido 2 veces" in mensajes[2]


def test_un_fallo_al_enviar_no_llega_al_bot_ni_detiene_los_siguientes(caplog):
    enviados = []

    def enviar(texto):
        if not enviados:
            enviados.append("fallo")
            raise alertas.ErrorTelegram("Telegram respondió 401: Unauthorized")
        enviados.append(texto)
    manejador = alertas.AlertasTelegram("x", enviar)
    log = logging.Logger("prueba", logging.INFO)
    log.addHandler(manejador)
    with caplog.at_level(logging.WARNING, logger="alertas"):
        log.info("Bot iniciado: %s", "XAUUSDm")
        log.info("Bot detenido por el usuario")
        manejador.cerrar(espera_s=5)
    assert enviados == ["fallo", "[x] ℹ️ Bot detenido por el usuario"]
    assert "401" in caplog.text


def test_el_envio_no_espera_a_la_red():
    manejador = alertas.AlertasTelegram("x", Mock(side_effect=lambda _t: __import__("time").sleep(0.5)))
    log = logging.Logger("prueba", logging.INFO)
    log.addHandler(manejador)
    antes = __import__("time").monotonic()
    for n in range(20):
        log.info("Bot iniciado: %s", n)
    assert __import__("time").monotonic() - antes < 0.2


def test_sin_configurar_no_hay_avisos():
    assert alertas.credenciales({}) == (None, None)
    assert alertas.credenciales({"TELEGRAM_TOKEN": "t"}) == (None, None)
    assert alertas.credenciales({"TELEGRAM_TOKEN": " t ", "TELEGRAM_CHAT_ID": "5"}) == ("t", "5")
    assert alertas.crear("x") is None  # conftest deja los tests sin Telegram


def test_el_token_nunca_aparece_en_un_error(monkeypatch):
    monkeypatch.setattr(alertas.time, "sleep", lambda _s: None)

    def falla(url, **_k):
        raise urllib.error.URLError(f"no se pudo abrir {url}")
    monkeypatch.setattr(alertas.urllib.request, "urlopen", falla)
    with pytest.raises(alertas.ErrorTelegram) as error:
        alertas.enviar_telegram("hola", "TOKEN-SECRETO", "5")
    assert "TOKEN-SECRETO" not in str(error.value) and error.value.__cause__ is None and error.value.__suppress_context__


def test_un_token_o_chat_incorrecto_no_se_reintenta(monkeypatch):
    llamadas = []

    def rechaza(url, **_k):
        llamadas.append(url)
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b'{"description": "Bad Request: chat not found"}'))
    monkeypatch.setattr(alertas.urllib.request, "urlopen", rechaza)
    with pytest.raises(alertas.ErrorTelegram, match="chat not found"):
        alertas.enviar_telegram("hola", "t", "5")
    assert len(llamadas) == 1


def test_el_mensaje_se_envia_al_chat_configurado(monkeypatch):
    visto = {}

    def responde(url, data=None, timeout=None):
        visto.update(url=url, data=data.decode())
        return io.BytesIO(b'{"ok": true}')
    monkeypatch.setattr(alertas.urllib.request, "urlopen", responde)
    alertas.enviar_telegram("hola " + "x" * 5000, "t", "-100")
    assert visto["url"] == "https://api.telegram.org/bott/sendMessage" and "chat_id=-100" in visto["data"]
    assert len(visto["data"]) < 4200


# ---------------- el lanzador ----------------
@pytest.fixture
def lanzar(monkeypatch):
    """main() del lanzador con un manejador de avisos simulado; devuelve (ejecutar, avisos)."""
    tb = lanzador.trading_bot
    for nombre in ("SYMBOL", "MT5_PATH", "EXIGIR_VENTAJA_SIGNIFICATIVA", "PERMITIR_CUENTA_REAL", "MAGIC_NUMBER", "VOLUMEN",
                   "MAX_SPREAD_PUNTOS", "DESVIACION_SLIPPAGE", "MAX_PERDIDA_DIARIA_PCT", "MAX_DRAWDOWN_PCT", "RIESGO_PCT",
                   "MAGICS_CUENTA", "ESTADO_CUENTA_PATH", "BLOQUEO_CUENTA_PATH", "MAX_RIESGO_ABIERTO_PCT", "MAX_POSICIONES_CUENTA"):
        monkeypatch.setattr(config, nombre, getattr(config, nombre))
    for nombre in ("verificar_cuenta", "cargar_modelo", "build_features", "ciclo", "resultado_del_dia", "es_salto_de_fin_de_semana",
                   "al_esperar"):
        monkeypatch.setattr(tb, nombre, getattr(tb, nombre))
    avisos = Mock()
    monkeypatch.setattr(lanzador, "PanelMT5", Mock())
    monkeypatch.setattr(lanzador, "ctypes", Mock())
    monkeypatch.setattr(lanzador.alertas, "crear", Mock(return_value=avisos))
    monkeypatch.setattr(tb, "log", Mock())
    monkeypatch.setattr(sys, "argv", ["demo_exness_bot.py", "--horas", "1", "--simbolo", "XAUUSDm"])

    def ejecutar(main_del_bot):
        monkeypatch.setattr(tb, "main", main_del_bot)
        lanzador.main()
    return ejecutar, avisos


def test_el_lanzador_engancha_los_avisos_con_activo_y_cuenta(lanzar):
    ejecutar, avisos = lanzar
    ejecutar(lambda: None)
    lanzador.alertas.crear.assert_called_once_with(f"XAUUSDm · demo {lanzador.CUENTA.login}")
    lanzador.trading_bot.log.addHandler.assert_any_call(avisos)
    avisos.avisar.assert_not_called()
    avisos.cerrar.assert_called_once()


def test_una_parada_con_motivo_se_avisa(lanzar):
    ejecutar, avisos = lanzar

    def main():
        raise SystemExit("Cuenta inesperada (1, x); el bot se detiene")
    with pytest.raises(SystemExit):
        ejecutar(main)
    avisos.avisar.assert_called_once_with("🛑 El bot se detuvo: Cuenta inesperada (1, x); el bot se detiene")
    avisos.cerrar.assert_called_once()


def test_el_fin_de_sesion_normal_no_duplica_el_aviso(lanzar):
    ejecutar, avisos = lanzar

    def main():
        raise SystemExit(0)
    with pytest.raises(SystemExit):
        ejecutar(main)
    avisos.avisar.assert_not_called()
    avisos.cerrar.assert_called_once()


def test_una_caida_inesperada_se_avisa(lanzar):
    ejecutar, avisos = lanzar

    def main():
        raise MemoryError("sin memoria")
    with pytest.raises(MemoryError):
        ejecutar(main)
    avisos.avisar.assert_called_once_with("🛑 El bot se cayó: MemoryError: sin memoria")
