"""Gestión de la posición abierta (config.GESTION): la misma regla que mide experimentos.py, ejecutada por el bot."""
import numpy as np
import pandas as pd
import pytest

import config
import experimentos as ex
import mt5_utils
import trading_bot as tb
from fake_mt5 import FakeMT5

POINT = 0.00001
BUY, SELL = tb.mt5.POSITION_TYPE_BUY, tb.mt5.POSITION_TYPE_SELL


def cerradas(*maximos_minimos, spread=0) -> pd.DataFrame:
    return pd.DataFrame({"high": [h for h, _ in maximos_minimos], "low": [l for _, l in maximos_minimos], "spread": spread})


# Compra en 1.1000 con stop a 100 puntos (1.0990) y objetivo a 200 (1.1020); venta simétrica.
COMPRA = (True, 1.1000, 1.0990, 1.1020)
VENTA = (False, 1.1000, 1.1010, 1.0980)


@pytest.mark.parametrize("gestion", ["breakeven", "parcial"])
def test_a_mas_1r_el_stop_pasa_a_la_entrada(gestion):
    assert tb.plan_de_gestion(*COMPRA, cerradas((1.1009, 1.0995)), POINT, gestion) == (None, False)
    assert tb.plan_de_gestion(*COMPRA, cerradas((1.1009, 1.0995), (1.1010, 1.1001)), POINT, gestion) == (1.1000, True)
    assert tb.plan_de_gestion(*VENTA, cerradas((1.1005, 1.0990)), POINT, gestion) == (1.1000, True)
    con_spread = cerradas((1.1005, 1.0990), spread=20)  # la venta sale al ask: 1.0990 + 20 puntos no llega a -1 R
    assert tb.plan_de_gestion(*VENTA, con_spread, POINT, gestion) == (None, False)


def test_el_stop_nunca_se_aleja_ni_se_repite():
    ya_en_la_entrada = (True, 1.1000, 1.1000, 1.1020)
    assert tb.plan_de_gestion(*ya_en_la_entrada, cerradas((1.1015, 1.1001)), POINT, "breakeven") == (None, True)
    assert tb.plan_de_gestion(*COMPRA, cerradas((1.0999, 1.0992)), POINT, "trailing") == (None, False)  # quedaría más lejos
    assert tb.plan_de_gestion(*COMPRA, pd.DataFrame({"high": [], "low": [], "spread": []}), POINT, "trailing") == (None, False)


def test_el_trailing_sigue_al_mejor_precio_a_la_distancia_del_stop():
    nuevo, activada = tb.plan_de_gestion(*COMPRA, cerradas((1.1004, 1.0998), (1.1007, 1.1002)), POINT, "trailing")
    assert nuevo == pytest.approx(1.0997) and not activada
    nuevo, _ = tb.plan_de_gestion(*VENTA, cerradas((1.1002, 1.0994), (1.0999, 1.0996)), POINT, "trailing")
    assert nuevo == pytest.approx(1.1004)


@pytest.mark.parametrize("gestion", [ex.BREAKEVEN, ex.TRAILING])
def test_el_bot_y_el_banco_de_experimentos_dejan_el_stop_en_el_mismo_sitio(gestion):
    """Tras cada vela, el stop del bot (plan_de_gestion) coincide con el que usa la simulación para la vela siguiente."""
    rng = np.random.default_rng(5)
    close = 1.1 + np.cumsum(rng.normal(0.00003, 0.0003, 30))
    o = np.r_[1.1, close[:-1]]
    h, l = np.maximum(o, close) + 0.0001, np.minimum(o, close) - 0.0001
    entrada, sl_dist = o[0], 0.0010
    sl_bot = entrada - sl_dist
    for j in range(len(o) - 1):
        if l[j] <= sl_bot:
            break
        vistas = pd.DataFrame({"high": h[:j + 1], "low": l[:j + 1], "spread": 0})
        nuevo, _ = tb.plan_de_gestion(True, entrada, sl_bot, entrada + 2 * sl_dist, vistas, POINT, gestion)
        if nuevo is not None and o[j + 1] > nuevo:
            sl_bot = nuevo
        # La simulación, cortada en j+1 y sin objetivo alcanzable, sale por ese mismo stop si la vela siguiente lo toca.
        resultado, _, por_tiempo, _ = ex._salida_gestionada(o, np.r_[l[:j + 1], -np.inf], np.r_[h[:j + 1], -np.inf],
                                                            0, j + 2, entrada, sl_dist, np.inf, gestion)
        # Tolerancia de un punto: el bot no envía cambios de stop menores de medio punto, que el broker no distingue.
        assert not por_tiempo and resultado == pytest.approx(min(o[j + 1], sl_bot) - entrada, abs=POINT)


# ---------------- el bot ----------------
@pytest.fixture
def mt5(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setattr(tb, "mt5", fake)
    monkeypatch.setattr(mt5_utils, "descargar_velas", fake.descargar_velas)
    return fake


def posicion(fake, monkeypatch, maximo, volumen=0.02):
    """Una compra en 1.1000 (stop 1.0990, objetivo 1.1020) y una vela cerrada desde entonces con ese máximo."""
    pos = fake.abrir_posicion(volumen=volumen)
    pos.price_open, pos.sl, pos.tp, pos.time = 1.1000, 1.0990, 1.1020, fake.bar_time - config.TIMEFRAME_SEGUNDOS + 5
    velas = pd.DataFrame({"time": [fake.bar_time - config.TIMEFRAME_SEGUNDOS], "high": [maximo], "low": [1.0998], "spread": [2]})
    monkeypatch.setattr(mt5_utils, "descargar_velas", lambda desde, cantidad: velas)
    fake.tick.bid, fake.tick.ask = 1.1008, 1.10082
    return pos


def test_sin_gestion_el_bot_no_toca_la_posicion(mt5, monkeypatch):
    pos = posicion(mt5, monkeypatch, maximo=1.1015)
    tb.ajustar_salida(pos, mt5.info, mt5.cuenta.login)
    assert mt5.modificaciones == [] and mt5.enviadas == []


def test_breakeven_mueve_el_stop_a_la_entrada_y_conserva_el_objetivo(mt5, monkeypatch):
    monkeypatch.setattr(config, "GESTION", "breakeven")
    pos = posicion(mt5, monkeypatch, maximo=1.1011)
    tb.ajustar_salida(pos, mt5.info, mt5.cuenta.login)
    assert (pos.sl, pos.tp) == (1.1000, 1.1020) and mt5.enviadas == []
    tb.ajustar_salida(pos, mt5.info, mt5.cuenta.login)
    assert len(mt5.modificaciones) == 1  # ya está en la entrada: no se repite


def test_si_el_precio_ya_paso_el_stop_nuevo_o_el_broker_lo_rechaza_se_conserva_el_anterior(mt5, monkeypatch):
    monkeypatch.setattr(config, "GESTION", "breakeven")
    pos = posicion(mt5, monkeypatch, maximo=1.1011)
    mt5.tick.bid = 1.0999  # el precio volvió por debajo de la entrada
    tb.ajustar_salida(pos, mt5.info, mt5.cuenta.login)
    assert mt5.modificaciones == [] and pos.sl == 1.0990
    mt5.tick.bid, mt5.rechaza_modificar = 1.1008, True
    tb.ajustar_salida(pos, mt5.info, mt5.cuenta.login)
    assert len(mt5.modificaciones) == 1 and pos.sl == 1.0990


def test_parcial_cierra_la_mitad_una_sola_vez_y_protege_el_resto(mt5, monkeypatch):
    monkeypatch.setattr(config, "GESTION", "parcial")
    pos = posicion(mt5, monkeypatch, maximo=1.1011, volumen=0.04)
    tb.ajustar_salida(pos, mt5.info, mt5.cuenta.login)
    (cierre,) = mt5.enviadas
    assert (cierre["volume"], cierre["comment"], cierre["position"]) == (0.02, "rf_mt5 parcial", pos.ticket)
    (resto,) = mt5.posiciones
    assert (resto.volume, resto.sl) == (0.02, 1.1000)
    tb.ajustar_salida(resto, mt5.info, mt5.cuenta.login)
    assert len(mt5.enviadas) == 1  # ya hay una salida de esa posición: no se vuelve a partir


def test_parcial_con_el_lote_minimo_solo_mueve_el_stop(mt5, monkeypatch):
    monkeypatch.setattr(config, "GESTION", "parcial")
    pos = posicion(mt5, monkeypatch, maximo=1.1011, volumen=0.01)
    tb.ajustar_salida(pos, mt5.info, mt5.cuenta.login)
    assert mt5.enviadas == [] and pos.sl == 1.1000


def test_el_lanzador_acepta_la_gestion(monkeypatch):
    import demo_exness_bot as lanzador
    monkeypatch.setattr("sys.argv", ["x", "--gestion", "trailing"])
    assert lanzador.leer_argumentos().gestion == "trailing"
    monkeypatch.setattr("sys.argv", ["x"])
    assert lanzador.leer_argumentos().gestion is None
