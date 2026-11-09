"""Límites del lanzador: solo la demo autorizada, duración acotada, hora del modelo y umbral manual."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import config
import demo_exness_bot as lanzador
from activos import ACTIVOS, EURUSD
from features import build_features


@pytest.fixture
def cuenta(monkeypatch):
    datos = SimpleNamespace(login=lanzador.LOGIN, server=lanzador.SERVER, trade_mode=0)
    monkeypatch.setattr(lanzador.mt5, "account_info", lambda: datos)
    return datos


def test_acepta_la_cuenta_autorizada(cuenta):
    assert lanzador.verificar_cuenta() == lanzador.LOGIN


@pytest.mark.parametrize("campo,valor", [("trade_mode", 2), ("trade_mode", 1), ("login", 123), ("server", "Exness-Real")])
def test_cuenta_distinta_detiene_el_bot(cuenta, campo, valor):
    setattr(cuenta, campo, valor)
    with pytest.raises(SystemExit, match="Cuenta inesperada"):
        lanzador.verificar_cuenta()


def test_el_limite_de_tiempo_termina_sin_ejecutar_el_ciclo(monkeypatch):
    ciclo = Mock()
    monkeypatch.setattr(lanzador, "_ciclo", ciclo)
    with pytest.raises(SystemExit):
        lanzador.ciclo_con_limite(0, Mock())({}, {})
    ciclo.assert_not_called()


def test_antes_del_limite_ejecuta_el_ciclo_y_da_el_latido(monkeypatch):
    ciclo, panel = Mock(), Mock()
    monkeypatch.setattr(lanzador, "_ciclo", ciclo)
    lanzador.ciclo_con_limite(float("inf"), panel)({}, {})
    ciclo.assert_called_once()
    panel.latido.assert_called_once()


def test_el_modelo_se_valida_contra_su_simbolo_y_se_restaura(monkeypatch):
    visto = []
    monkeypatch.setattr(config, "SYMBOL", EURUSD.simbolo)
    monkeypatch.setattr(lanzador, "_cargar_modelo", lambda: visto.append(config.SYMBOL) or {})
    lanzador.cargar_modelo()
    assert visto == ["EURUSD"]
    assert config.SYMBOL == "EURUSDm"


def test_la_hora_del_modelo_va_desplazada_y_el_resto_no_cambia():
    n = 60
    velas = pd.DataFrame({"time": [1793994300 + 900 * i for i in range(n)], "open": 1.1, "high": 1.2, "low": 1.0,
                          "close": [1.1 + i * 1e-4 for i in range(n)], "tick_volume": 100, "spread": 8})
    desplazadas = lanzador.features_en_hora_del_modelo(velas, 1e-5)
    esperadas = build_features(velas.assign(time=velas["time"] + 7200), 1e-5)
    pd.testing.assert_series_equal(desplazadas["hora_sin"], esperadas["hora_sin"])
    assert not desplazadas["hora_sin"].equals(build_features(velas, 1e-5)["hora_sin"])
    pd.testing.assert_series_equal(desplazadas["atr"], build_features(velas, 1e-5)["atr"])
    assert velas["time"].iloc[0] == 1793994300  # no modifica las velas originales


@pytest.fixture
def lanzar(monkeypatch):
    """Ejecuta main() sin MT5, sin bucle y sin tocar el panel real; devuelve el modelo que recibiría el bot y el panel."""
    tb = lanzador.trading_bot
    for nombre in ("SYMBOL", "MT5_PATH", "EXIGIR_VENTAJA_SIGNIFICATIVA", "PERMITIR_CUENTA_REAL", "MAGIC_NUMBER", "VOLUMEN",
                   "MAX_SPREAD_PUNTOS", "DESVIACION_SLIPPAGE", "MAX_PERDIDA_DIARIA_PCT", "MAX_DRAWDOWN_PCT"):
        monkeypatch.setattr(config, nombre, getattr(config, nombre))
    for nombre in ("verificar_cuenta", "cargar_modelo", "build_features", "ciclo", "resultado_del_dia",
                   "es_salto_de_fin_de_semana"):
        monkeypatch.setattr(tb, nombre, getattr(tb, nombre))  # main() los sustituye: así se restauran
    panel, modelo = Mock(), {}
    monkeypatch.setattr(lanzador, "PanelMT5", Mock(return_value=panel))
    monkeypatch.setattr(lanzador, "ctypes", Mock())
    monkeypatch.setattr(lanzador, "_cargar_modelo", lambda: {"umbral_long": 0.33, "umbral_short": 0.38})
    monkeypatch.setattr(tb, "log", Mock())
    monkeypatch.setattr(tb, "main", lambda: modelo.update(tb.cargar_modelo()))

    def ejecutar(*argumentos):
        monkeypatch.setattr(sys, "argv", ["demo_exness_bot.py", "--horas", "1", *argumentos])
        lanzador.main()
        return modelo, panel
    return ejecutar


def test_el_umbral_manual_llega_al_bot_y_al_panel(lanzar):
    modelo, panel = lanzar("--umbral", "0.30")
    assert (modelo["umbral_long"], modelo["umbral_short"]) == (0.30, 0.30)
    panel.fijar_umbrales.assert_called_once_with(0.30, 0.30)


def test_sin_umbral_manual_se_usan_los_del_modelo(lanzar):
    modelo, panel = lanzar()
    assert (modelo["umbral_long"], modelo["umbral_short"]) == (0.33, 0.38)
    panel.fijar_umbrales.assert_called_once_with(0.33, 0.38)


def test_un_lado_desactivado_por_el_entrenamiento_sigue_desactivado():
    modelo = {"umbral_long": None, "umbral_short": 0.38}
    lanzador.aplicar_umbral(modelo, 0.30)
    assert modelo == {"umbral_long": None, "umbral_short": 0.30}


@pytest.mark.parametrize("valor", ["0", "1", "-0.3", "30", "nan", "abc"])
def test_un_umbral_no_valido_no_arranca(lanzar, valor):
    with pytest.raises(SystemExit) as salida:
        lanzar("--umbral", valor)
    assert salida.value.code == 2  # lo rechaza argparse


def test_por_defecto_el_eurusd_no_cambia_nada(lanzar):
    lanzar()
    assert (config.SYMBOL, config.MAGIC_NUMBER, config.VOLUMEN, config.MAX_SPREAD_PUNTOS) == ("EURUSDm", 202601, 0.01, 25)
    assert config.DESVIACION_SLIPPAGE == 20
    assert config.ESTADO_PATH == EURUSD.carpeta / "estado_bot.json"
    assert lanzador.trading_bot.es_salto_de_fin_de_semana is lanzador._es_salto_de_fin_de_semana


def test_cada_activo_lleva_su_modelo_lote_spread_y_archivos(lanzar):
    oro = ACTIVOS["XAUUSDm"]
    lanzar("--simbolo", "XAUUSDm", "--volumen", "0.02", "--max-spread", "700")
    assert (config.SYMBOL, config.MODEL_PATH, config.MAGIC_NUMBER) == ("XAUUSDm", oro.modelo, oro.magic)
    assert (config.VOLUMEN, config.MAX_SPREAD_PUNTOS) == (0.02, 700)
    assert config.DESVIACION_SLIPPAGE == 600
    assert (config.LOG_PATH.parent, config.ESTADO_PATH.parent, config.LOCK_PATH) == (oro.carpeta, oro.carpeta, oro.bloqueo)
    assert lanzador.trading_bot.es_salto_de_fin_de_semana is lanzador.salto_admitido


def test_ningun_par_de_activos_comparte_magic_ni_archivos():
    for campo in ("magic", "modelo", "carpeta", "bloqueo", "panel"):
        assert len({getattr(a, campo) for a in ACTIVOS.values()}) == len(ACTIVOS), campo


def test_umbrales_por_lado_y_lado_apagado(lanzar):
    modelo, panel = lanzar("--umbral-long", "0.25", "--umbral-short", "off")
    assert (modelo["umbral_long"], modelo["umbral_short"]) == (0.25, None)
    panel.fijar_umbrales.assert_called_once_with(0.25, None)


def test_un_umbral_por_lado_puede_encender_uno_que_estaba_apagado(lanzar, monkeypatch):
    monkeypatch.setattr(lanzador, "_cargar_modelo", lambda: {"umbral_long": None, "umbral_short": 0.38})
    modelo, _ = lanzar("--umbral-long", "0.30")
    assert (modelo["umbral_long"], modelo["umbral_short"]) == (0.30, 0.38)


def test_los_limites_de_riesgo_llegan_como_fraccion(lanzar):
    lanzar("--perdida-diaria", "3", "--drawdown", "15")
    assert (config.MAX_PERDIDA_DIARIA_PCT, config.MAX_DRAWDOWN_PCT) == (0.03, 0.15)


@pytest.mark.parametrize("argumentos", [("--simbolo", "BTCUSDm"), ("--volumen", "0"), ("--max-spread", "-1"),
                                        ("--perdida-diaria", "101"), ("--umbral-short", "1.5")])
def test_argumentos_no_validos_no_arrancan(lanzar, argumentos):
    with pytest.raises(SystemExit) as salida:
        lanzar(*argumentos)
    assert salida.value.code == 2


def test_la_orden_de_detener_termina_la_sesion_y_se_consume(monkeypatch, tmp_path):
    ciclo, archivo = Mock(), tmp_path / "DETENER"
    monkeypatch.setattr(lanzador, "_ciclo", ciclo)
    archivo.touch()
    with pytest.raises(SystemExit) as salida:
        lanzador.ciclo_con_limite(float("inf"), Mock(), archivo)({}, {})
    assert salida.value.code == 0
    ciclo.assert_not_called()
    assert not archivo.exists()


def test_el_limite_diario_suma_todos_los_activos_y_nada_ajeno(monkeypatch):
    deal = lambda magic, profit, t=1000: SimpleNamespace(magic=magic, profit=profit, swap=0.0, commission=0.0, fee=0.0, time=t)  # noqa: E731
    falso = SimpleNamespace(
        symbol_info_tick=lambda _s: SimpleNamespace(time=1000),
        history_deals_get=lambda *_a: (deal(202601, -2.0), deal(202603, -3.0), deal(999, -50.0), deal(202602, -7.0, t=-90000)),
        positions_get=lambda: (SimpleNamespace(magic=202604, profit=-1.0, swap=-0.5), SimpleNamespace(magic=999, profit=-9.0, swap=0.0)),
        last_error=lambda: (1, "x"))
    monkeypatch.setattr(lanzador, "mt5", falso)
    assert lanzador.resultado_del_dia() == (-5.0, -1.5)
