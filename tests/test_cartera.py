"""Reglas de cartera: lo que mide cartera.py y lo que aplica el bot (correlación, reducción en drawdown, confianza)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import activos
import cartera
import config
import mt5_utils
import train_model as tm
import trading_bot as tb
from fake_mt5 import FakeMT5

H = 3600


def trade(simbolo, entrada, salida, r, lado="long", margen=0.0) -> dict:
    return {"entrada": entrada, "salida": salida, "lado": lado, "r": r, "margen": margen, "simbolo": simbolo, "semilla": 42}


def test_los_grupos_de_correlacion_son_los_indices():
    assert set(activos.grupo_de("US500m")) == {"US500m", "USTECm", "DE30m"} == set(activos.grupo_de("DE30m"))
    assert activos.grupo_de("XAUUSDm") == ("XAUUSDm",) and activos.grupo_de("EURUSDm") == ("EURUSDm",)


def test_el_detalle_reproduce_los_trades_de_la_simulacion_del_entrenamiento():
    rng = np.random.default_rng(1)
    n = 300
    filas = pd.DataFrame({"r_long": rng.normal(0, 1, n), "r_short": rng.normal(0, 1, n),
                          "k_long": np.arange(n) + rng.integers(1, 20, n), "k_short": np.arange(n) + rng.integers(1, 20, n),
                          "to_long": rng.random(n) < 0.3, "to_short": rng.random(n) < 0.3})
    p_long, p_short, tiempos = rng.random(n), rng.random(n), 1_000_000 + 900 * np.arange(n + 30)
    detalle = cartera.simular_con_detalle(filas, p_long, p_short, 0.7, 0.8, tiempos)
    assert np.allclose(detalle["r"], tm.simular(filas, p_long, p_short, 0.7, 0.8)) and len(detalle) > 5
    assert (detalle["margen"] >= 0).all() and (detalle["salida"] > detalle["entrada"]).all()
    primero = detalle.iloc[0]
    i = int(np.flatnonzero((p_long >= 0.7) != (p_short >= 0.8))[0])
    assert primero["entrada"] == tiempos[i + 1] and primero["lado"] == ("long" if p_long[i] >= 0.7 else "short")
    solo_ventas = cartera.simular_con_detalle(filas, p_long, p_short, None, 0.8, tiempos)
    assert set(solo_ventas["lado"]) == {"short"}


def test_la_correlacion_descarta_la_segunda_posicion_del_mismo_lado_en_el_grupo():
    trades = pd.DataFrame([trade("US500m", 0, 10 * H, 1.0), trade("USTECm", 1 * H, 5 * H, -1.0),  # mismo lado y solapada: fuera
                           trade("DE30m", 2 * H, 6 * H, 2.0, lado="short"),                        # lado contrario: entra
                           trade("XAUUSDm", 3 * H, 4 * H, 0.5),                                    # otro grupo: entra
                           trade("USTECm", 10 * H, 12 * H, 3.0)])                                  # el S&P 500 ya cerró: entra
    assert len(cartera.simular_cartera(trades)) == 5
    tomados = cartera.simular_cartera(trades, correlacion=True)
    assert list(tomados["simbolo"]) == ["US500m", "DE30m", "XAUUSDm", "USTECm"] and tomados["r_cartera"].sum() == 6.5


def test_la_reduccion_pesa_la_mitad_mientras_lo_cerrado_este_10r_bajo_su_maximo():
    perdidas = [trade("XAUUSDm", n * H, n * H + 60, -1.0) for n in range(11)]
    despues = [trade("XAUUSDm", 11 * H, 11 * H + 60, 12.0), trade("XAUUSDm", 12 * H, 12 * H + 60, 1.0)]
    tomados = cartera.simular_cartera(pd.DataFrame(perdidas + despues), reducir=True)
    assert list(tomados["peso"]) == [1.0] * 10 + [0.5, 0.5, 1.0]  # el trade 11 ya ve -10 R; tras recuperar 6 R quedan -4,5
    assert tomados["r_cartera"].iloc[-2] == 6.0
    abierta = [trade("EURUSDm", 0, 100 * H, -30.0), trade("XAUUSDm", 1 * H, 2 * H, 1.0)]
    assert list(cartera.simular_cartera(pd.DataFrame(abierta), reducir=True)["peso"]) == [1.0, 1.0]  # lo abierto no cuenta


def test_la_confianza_pesa_mas_los_trades_que_superan_su_umbral_con_holgura():
    trades = pd.DataFrame([trade("XAUUSDm", 0, H, 2.0, margen=0.05), trade("XAUUSDm", 2 * H, 3 * H, -1.0, margen=0.049)])
    tomados = cartera.simular_cartera(trades, confianza=True)
    assert list(tomados["peso"]) == [1.5, 1.0] and list(tomados["r_cartera"]) == [3.0, -1.0]


def test_las_metricas_miden_el_drawdown_por_orden_de_salida():
    tomados = cartera.simular_cartera(pd.DataFrame([trade("EURUSDm", 0, 10 * H, 3.0), trade("XAUUSDm", H, 2 * H, -2.0),
                                                    trade("XAUUSDm", 3 * H, 4 * H, -1.0)]))
    m = cartera.metricas(tomados)
    assert (m["trades"], m["total_R"], m["max_drawdown_R"]) == (3, 0.0, 3.0)  # -2, -3 y luego +3
    assert cartera.metricas(tomados.iloc[0:0])["trades"] == 0
    filas = cartera.comparar(pd.DataFrame([trade("EURUSDm", 0, H, 1.0), trade("XAUUSDm", 0, H, -0.5)]))
    assert [f["regla"] for f in filas] == list(cartera.REGLAS) and "las tres" in cartera.tabla(filas)


# ---------------- el bot ----------------
@pytest.fixture
def mt5(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setattr(tb, "mt5", fake)
    monkeypatch.setattr(mt5_utils, "descargar_velas", fake.descargar_velas)
    return fake


def otra_posicion(fake, simbolo, tipo, magic=202604):
    pos = fake.abrir_posicion(tipo=tipo)
    pos.symbol, pos.magic = simbolo, magic
    return pos


def test_el_bot_no_entra_si_un_activo_de_su_grupo_ya_tiene_ese_lado(mt5, monkeypatch):
    BUY, SELL = tb.mt5.ORDER_TYPE_BUY, tb.mt5.ORDER_TYPE_SELL
    monkeypatch.setattr(config, "SYMBOL", "US500m")
    monkeypatch.setattr(config, "MAGICS_CUENTA", {config.MAGIC_NUMBER, 202604})
    otra_posicion(mt5, "USTECm", tb.mt5.POSITION_TYPE_BUY)
    assert tb.motivo_exposicion(BUY, 1.1, 1.09, 0.01) is None  # sin la regla
    monkeypatch.setattr(config, "SIMBOLOS_CORRELACIONADOS", {"US500m", "USTECm", "DE30m"})
    assert "USTECm ya tiene una compra" in tb.motivo_exposicion(BUY, 1.1, 1.09, 0.01)
    assert tb.motivo_exposicion(SELL, 1.1, 1.11, 0.01) is None  # el lado contrario no es la misma apuesta


def test_la_correlacion_ignora_otros_grupos_las_posiciones_ajenas_y_las_del_propio_activo(mt5, monkeypatch):
    BUY = tb.mt5.ORDER_TYPE_BUY
    monkeypatch.setattr(config, "SYMBOL", "US500m")
    monkeypatch.setattr(config, "MAGICS_CUENTA", {config.MAGIC_NUMBER, 202604})
    monkeypatch.setattr(config, "SIMBOLOS_CORRELACIONADOS", {"US500m", "USTECm", "DE30m"})
    otra_posicion(mt5, "XAUUSDm", tb.mt5.POSITION_TYPE_BUY)
    otra_posicion(mt5, "USTECm", tb.mt5.POSITION_TYPE_BUY, magic=999)  # abierta a mano
    otra_posicion(mt5, "US500m", tb.mt5.POSITION_TYPE_BUY)
    assert tb.motivo_correlacion(BUY) is None


def test_el_factor_de_tamano_combina_confianza_y_drawdown(mt5, monkeypatch):
    estado = {"equity_max": 10_000.0}
    assert tb.factor_de_tamano(estado, 0.2) == 1.0  # sin reglas activas
    monkeypatch.setattr(config, "TAMANO_POR_CONFIANZA", True)
    assert tb.factor_de_tamano(estado, 0.05) == 1.5 and tb.factor_de_tamano(estado, 0.049) == 1.0
    assert tb.factor_de_tamano(estado, None) == 1.0
    monkeypatch.setattr(config, "REDUCIR_EN_DRAWDOWN", True)
    mt5.cuenta.equity = 9_600.0
    assert tb.factor_de_tamano(estado, 0.0) == 1.0  # -4 %
    mt5.cuenta.equity = 9_500.0
    assert tb.factor_de_tamano(estado, 0.0) == 0.5 and tb.factor_de_tamano(estado, 0.1) == 0.75


def test_el_factor_escala_el_lote_sin_pasar_los_topes_ni_perder_la_entrada(mt5, monkeypatch):
    monkeypatch.setattr(config, "VOLUMEN", 0.04)
    assert tb.calcular_volumen(mt5.info, 0.001) == 0.04
    assert tb.calcular_volumen(mt5.info, 0.001, 0.5) == 0.02 and tb.calcular_volumen(mt5.info, 0.001, 1.5) == 0.06
    monkeypatch.setattr(config, "VOLUMEN", 0.01)
    assert tb.calcular_volumen(mt5.info, 0.001, 0.5) == 0.01  # no baja del lote mínimo
    monkeypatch.setattr(config, "VOLUMEN", 1.0)
    assert tb.calcular_volumen(mt5.info, 0.001, 1.5) == config.MAX_LOTES


def test_la_entrada_lleva_el_margen_de_la_senal_hasta_el_lote(mt5, monkeypatch):
    from features import FEATURES
    from test_bot import ModeloFijo, aperturas, nuevo_estado
    monkeypatch.setattr(config, "VOLUMEN", 0.02)
    monkeypatch.setattr(config, "TAMANO_POR_CONFIANZA", True)
    bundle = {"features": FEATURES, "model_long": ModeloFijo(0.9), "model_short": ModeloFijo(0.1), "umbral_long": 0.5, "umbral_short": 0.5}
    tb.ciclo(bundle, nuevo_estado(mt5))
    assert aperturas(mt5)[0]["volume"] == 0.03  # 0.9 supera el umbral 0.5 con holgura: lote y medio


def test_el_lanzador_activa_cada_regla_con_su_opcion(monkeypatch):
    import demo_exness_bot as lanzador
    for nombre in ("SIMBOLOS_CORRELACIONADOS", "REDUCIR_EN_DRAWDOWN", "TAMANO_POR_CONFIANZA", "SYMBOL", "MODEL_PATH", "MAGIC_NUMBER",
                   "VOLUMEN", "RIESGO_PCT", "MAX_SPREAD_PUNTOS", "DESVIACION_SLIPPAGE", "MAX_PERDIDA_DIARIA_PCT", "MAX_DRAWDOWN_PCT",
                   "CERRAR_AL_LIMITE", "GESTION", "MT5_PATH", "EXIGIR_VENTAJA_SIGNIFICATIVA", "PERMITIR_CUENTA_REAL", "MAGICS_CUENTA",
                   "ESTADO_CUENTA_PATH", "BLOQUEO_CUENTA_PATH", "MAX_RIESGO_ABIERTO_PCT", "MAX_POSICIONES_CUENTA"):
        monkeypatch.setattr(config, nombre, getattr(config, nombre))
    monkeypatch.setattr("sys.argv", ["x", "--simbolo", "USTECm", "--limite-correlacion", "--reducir-en-drawdown", "--por-confianza"])
    lanzador.configurar_activo(activos.ACTIVOS["USTECm"], lanzador.leer_argumentos())
    assert config.SIMBOLOS_CORRELACIONADOS == {"US500m", "USTECm", "DE30m"} and config.REDUCIR_EN_DRAWDOWN and config.TAMANO_POR_CONFIANZA
    monkeypatch.setattr("sys.argv", ["x", "--simbolo", "USTECm"])
    lanzador.configurar_activo(activos.ACTIVOS["USTECm"], lanzador.leer_argumentos())
    assert config.SIMBOLOS_CORRELACIONADOS is None and not config.REDUCIR_EN_DRAWDOWN and not config.TAMANO_POR_CONFIANZA
