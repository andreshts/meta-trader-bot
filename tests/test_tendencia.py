"""Seguimiento de tendencia por reglas (tendencia.py): ejecución, costes, causalidad, evaluación y descarga."""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tendencia as td
import train_model as tm
from conftest import POINT
from features import atr_wilder

DIA = 86_400
LUNES = int(datetime(2026, 10, 26, tzinfo=timezone.utc).timestamp())  # lunes 00:00 del servidor
BASE = 1.1000
REGLAS = {"n_entrada": 10, "n_salida": 5, "inicio": 20}


def velas_d1(cierres, rango=0.0010, spread=0, t0=LUNES):
    """Velas diarias consecutivas: cada una abre en el cierre anterior y sobresale `rango / 2` por cada lado."""
    c = np.asarray(cierres, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"time": t0 + DIA * np.arange(len(c)), "open": o, "close": c,
                         "high": np.maximum(o, c) + rango / 2, "low": np.minimum(o, c) - rango / 2,
                         "tick_volume": 100.0, "spread": spread, "real_volume": 0})


def simular(d, **kw):
    return td.simular_donchian(d, POINT, **{**REGLAS, **kw})


def paseo_aleatorio(n=2000, semilla=5, spread=12):
    rng = np.random.default_rng(semilla)
    return velas_d1(BASE + np.cumsum(rng.normal(0, 0.004, n)), rango=0.004, spread=spread)


# Ruptura alcista en la vela 30 (cierre por encima del máximo de las 10 anteriores), 5 velas arriba y un cierre en la
# 36 por debajo del mínimo de las 5 anteriores (1.1015), sin tocar el stop (1.1020 - 2 x 0.0011 = 1.0998).
RUPTURA_ALCISTA = [BASE] * 30 + [BASE + 0.0020] * 6 + [BASE + 0.0012] * 4
RUPTURA_BAJISTA = [BASE] * 30 + [BASE - 0.0020] * 2 + [BASE - 0.0004] * 6


def riesgo_de(d, vela):
    return 2 * atr_wilder(d, 20).iloc[vela]


# ---------------- ejecución ----------------
def test_entra_en_la_apertura_siguiente_y_sale_por_el_canal():
    d = velas_d1(RUPTURA_ALCISTA)
    trades, _ = simular(d)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert (t["lado"], t["vela_senal"], t["vela_entrada"], t["vela_salida"], t["motivo"]) == ("long", 30, 31, 37, "canal")
    assert np.isclose(riesgo_de(d, 30), 0.0022)
    assert np.isclose(t["r"], (BASE + 0.0012 - (BASE + 0.0020)) / 0.0022)    # sale en la apertura de la 37
    assert t["t_entrada"] == d["time"].iloc[31] and t["t_fin"] == d["time"].iloc[37]


def test_long_entra_al_ask_y_sale_al_bid_con_deslizamiento():
    d = velas_d1(RUPTURA_ALCISTA, spread=10)
    t = simular(d, slippage_puntos=1, comision_puntos=3)[0].iloc[0]
    entrada = BASE + 0.0020 + 10 * POINT + POINT
    salida = BASE + 0.0012 - POINT
    assert np.isclose(t["r"], (salida - entrada - 3 * POINT) / riesgo_de(d, 30))


def test_short_con_stop_disparado_por_el_ask():
    # Short al bid en 1.0980 con stop en 1.1002. La vela 32 sube hasta 1.1001: el bid no llega al stop, pero el ask
    # (+20 puntos) sí. Sin spread la posición sigue abierta al final.
    sin_spread, _ = simular(velas_d1(RUPTURA_BAJISTA))
    assert sin_spread.empty
    d = velas_d1(RUPTURA_BAJISTA, spread=20)
    t = simular(d)[0].iloc[0]
    assert (t["lado"], t["vela_entrada"], t["vela_salida"], t["motivo"]) == ("short", 31, 32, "stop")
    assert np.isclose(t["r"], -1.0)
    assert t["t_fin"] == d["time"].iloc[32] + DIA - 1     # depende de toda la vela del stop


def test_stop_con_hueco_se_ejecuta_en_la_apertura():
    d = velas_d1(RUPTURA_ALCISTA)
    d.loc[32, ["open", "high", "low", "close"]] = [1.0950, 1.0955, 1.0945, 1.0950]
    t = simular(d, slippage_puntos=1)[0].iloc[0]
    assert (t["vela_salida"], t["motivo"]) == (32, "stop")
    entrada = BASE + 0.0020 + POINT
    assert np.isclose(t["r"], (1.0950 - POINT - entrada) / riesgo_de(d, 30))
    assert t["r"] < -3


def test_tras_salir_puede_entrar_al_lado_contrario_con_la_misma_senal():
    # El cierre de la 36 (1.0990) queda por debajo del mínimo de las 5 anteriores (salida del long) y de las 10
    # anteriores (entrada short): en la apertura de la 37 se cierra el long y se abre el short. Con stop de 5 ATR,
    # ninguno toca el stop; la subida de la 42 cierra el short.
    d = velas_d1([BASE] * 30 + [BASE + 0.0020] * 6 + [BASE - 0.0010] * 6 + [BASE + 0.0005] * 3)
    trades, _ = simular(d, stop_atr=5.0)
    assert list(trades["lado"]) == ["long", "short"] and list(trades["motivo"]) == ["canal", "canal"]
    assert trades.iloc[0]["vela_salida"] == trades.iloc[1]["vela_entrada"] == 37
    assert trades.iloc[1]["vela_senal"] == 36 and trades.iloc[1]["vela_salida"] == 43


def test_swap_del_simbolo_y_pesimista():
    d = velas_d1(RUPTURA_ALCISTA)
    base = simular(d)[0].iloc[0]
    noches = tm.noches_de_swap(int(d["time"].iloc[31]), int(d["time"].iloc[37]), 3)
    assert noches >= 3
    riesgo = riesgo_de(d, 30)
    con_swap = simular(d, swap_long_puntos=-7)[0].iloc[0]
    assert np.isclose(con_swap["r"] - base["r"], noches * -7 * POINT / riesgo)
    pesimista = simular(d, swap_long_puntos=+50, swap_anual_en_contra=0.02)[0].iloc[0]
    assert np.isclose(pesimista["r"] - base["r"], -noches * 0.02 * (BASE + 0.0020) / 365 / riesgo)
    # En el short también va en contra.
    d = velas_d1(RUPTURA_BAJISTA[:32] + [BASE - 0.0020] * 6 + [BASE - 0.0010] * 3)
    base, pesimista = simular(d)[0].iloc[0], simular(d, swap_anual_en_contra=0.02)[0].iloc[0]
    assert pesimista["r"] < base["r"]


def test_marcado_a_mercado_suma_el_resultado_de_cada_trade():
    d = paseo_aleatorio()
    trades, mtm = simular(d, slippage_puntos=1, comision_puntos=3, swap_long_puntos=-7, swap_short_puntos=2)
    assert len(trades) > 30
    acumulado = mtm.to_numpy().cumsum()
    siguiente = trades["vela_entrada"].shift(-1, fill_value=len(d)).to_numpy()
    plano = siguiente > trades["vela_salida"].to_numpy()       # sin posición al cierre de la vela de salida
    assert plano.sum() > 10
    np.testing.assert_allclose(acumulado[trades["vela_salida"].to_numpy()[plano]],
                               trades["r"].cumsum().to_numpy()[plano], atol=1e-9)


@pytest.mark.parametrize("corte", [700, 1200, 1500])
def test_sin_mirar_el_futuro(corte):
    d = paseo_aleatorio()
    trades, mtm = simular(d)
    pert = d.copy()
    pert.loc[corte:, ["open", "high", "low", "close"]] *= 1.3
    trades_p, mtm_p = simular(pert)
    pd.testing.assert_frame_equal(trades[trades["vela_salida"] < corte].reset_index(drop=True),
                                  trades_p[trades_p["vela_salida"] < corte].reset_index(drop=True))
    np.testing.assert_allclose(mtm.iloc[:corte].to_numpy(), mtm_p.iloc[:corte].to_numpy())


def test_detecta_una_tendencia_plantada_y_no_el_ruido():
    rng = np.random.default_rng(11)
    deriva = np.tile([0.002] * 150 + [-0.002] * 150, 10)               # tramos de 150 días al alza y a la baja
    d = velas_d1(BASE + np.cumsum(deriva + rng.normal(0, 0.004, len(deriva))), rango=0.004, spread=12)
    plantada = tm.resumen(simular(d, n_entrada=55, n_salida=20, inicio=100)[0]["r"].to_numpy())
    assert plantada["t_stat"] > 3
    ruido = tm.resumen(simular(paseo_aleatorio(3000), n_entrada=55, n_salida=20, inicio=100)[0]["r"].to_numpy())
    assert abs(ruido["t_stat"]) < 2


# ---------------- evaluación ----------------
def test_separar_por_corte_descarta_los_trades_que_lo_cruzan():
    trades = pd.DataFrame({"t_entrada": [10, 20, 40], "t_fin": [19, 41, 50], "r": [1.0, 2.0, 3.0]})
    dev, hold = td.separar_por_corte(trades, 30)
    assert list(dev["r"]) == [1.0] and list(hold["r"]) == [3.0]


def test_cartera_mensual_suma_mercados_y_rellena_meses_vacios():
    t = lambda s: int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())  # noqa: E731
    a = pd.Series([1.0, 0.5, -2.0], index=[t("2019-12-31"), t("2020-01-10"), t("2020-03-02")])
    b = pd.Series([0.25, 1.0], index=[t("2020-01-20"), t("2020-04-01")])
    meses = td.cartera_mensual({"A": a, "B": b}, desde=t("2020-01-05"))
    assert list(meses.index.strftime("%Y-%m")) == ["2020-01", "2020-02", "2020-03", "2020-04"]
    assert list(meses) == [0.75, 0.0, -2.0, 1.0]


def test_resumen_mensual_y_segmentos():
    meses = pd.Series([1.0, -1.0, 2.0, 0.0] * 6, index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    m = td.resumen_mensual(meses)
    assert m["meses"] == 24 and np.isclose(m["media_R"], 0.5)
    assert np.isclose(m["sharpe_anual"], 0.5 / meses.std() * np.sqrt(12))
    assert np.isclose(m["t_mensual"], 0.5 / (meses.std() / np.sqrt(24)))
    s = td.segmentos(meses, 8)
    assert len(s) == 8 and np.isclose(sum(x["total_R"] for x in s), 12.0)


def test_problemas_de_datos():
    limpio = velas_d1([BASE] * 50, t0=td.segundos_de("2010-01-04"))
    assert td.problemas_de_datos(limpio) == []
    assert td.problemas_de_datos(velas_d1([BASE] * 50, t0=td.segundos_de("2012-08-06"))) == ["empieza el 2012-08-06"]
    con_hueco = limpio.copy()
    con_hueco.loc[30:, "time"] += 20 * DIA
    assert td.problemas_de_datos(con_hueco) == ["hueco de 21 días"]
    con_split = velas_d1([BASE] * 25 + [BASE * 8] * 25, t0=td.segundos_de("2010-01-04"))   # contrasplit 1:8 sin ajustar
    assert td.problemas_de_datos(con_split) == ["salto de 700% entre cierres"]


def test_velas_con_spread_anomalo():
    d = velas_d1([93.0] * 10, rango=0.5)
    d.loc[[3, 4], "spread"] = 5875            # 58.75 dólares en un ETF de 93 (IEF, noviembre de 2024)
    d.loc[5, "spread"] = 50                   # 0.50 dólares: amplio, pero posible
    assert td.velas_con_spread_anomalo(d, 0.01) == 2


def test_pesos_por_sector_reparten_el_mismo_riesgo_por_sector():
    pesos = td.pesos_por_sector({"A": "indices", "B": "indices", "C": "indices", "D": "bonos"})
    assert pesos == {"A": 2 / 3, "B": 2 / 3, "C": 2 / 3, "D": 2.0}
    assert np.isclose(np.mean(list(pesos.values())), 1.0)
    assert np.isclose(sum(p for m, p in pesos.items() if m != "D"), pesos["D"])


def test_evaluar_pondera_trades_y_cartera():
    t0 = td.segundos_de("2015-01-05")
    velas = {m: velas_d1(BASE + np.cumsum(np.random.default_rng(s).normal(0, 0.004, 2000)), rango=0.004,
                         spread=12, t0=t0) for m, s in (("A", 1), ("B", 2))}
    costes = {m: dict(tm.COSTES_POR_DEFECTO) for m in velas}
    puntos = {m: POINT for m in velas}
    igual = td.evaluar(velas, costes, puntos, "D1")
    solo_a = td.evaluar(velas, costes, puntos, "D1", pesos={"A": 1.0, "B": 0.0})
    doble = td.evaluar(velas, costes, puntos, "D1", pesos={"A": 2.0, "B": 2.0})
    assert igual["desarrollo"]["trades"]["trades"] > 30
    assert np.isclose(doble["desarrollo"]["mensual"]["total_R"], 2 * igual["desarrollo"]["mensual"]["total_R"])
    assert np.isclose(doble["desarrollo"]["trades"]["esperanza_R"], 2 * igual["desarrollo"]["trades"]["esperanza_R"])
    assert np.isclose(solo_a["desarrollo"]["mensual"]["total_R"],
                      td.evaluar({"A": velas["A"]}, costes, puntos, "D1")["desarrollo"]["mensual"]["total_R"])
    assert doble["por_mercado"] == igual["por_mercado"]        # el detalle por mercado no se pondera


def _ev(t_trades=3.0, t_mensual=3.0, positivos=6, esperanza=0.1, esperanza_hold=0.1, trades=100):
    return {"desarrollo": {"trades": {"trades": trades, "t_stat": t_trades, "esperanza_R": esperanza},
                           "mensual": {"t_mensual": t_mensual},
                           "segmentos": [{"total_R": 1.0 if i < positivos else -1.0} for i in range(8)]},
            "holdout": {"trades": {"esperanza_R": esperanza_hold}}}


@pytest.mark.parametrize("principal, robustez, pasa", [
    (_ev(), [_ev(), _ev()], True),
    (_ev(t_trades=1.9), [_ev()], False),
    (_ev(t_mensual=1.9), [_ev()], False),
    (_ev(positivos=4), [_ev()], False),            # 50% de segmentos positivos < 60%
    (_ev(trades=29), [_ev()], False),
    (_ev(), [_ev(), _ev(esperanza=-0.01)], False),  # una comprobación de robustez pierde
    (_ev(esperanza_hold=-0.01), [_ev()], False),
])
def test_veredicto(principal, robustez, pasa):
    assert td.veredicto(principal, robustez)["pasa"] is pasa


# ---------------- descarga ----------------
class TerminalConCache:
    """Devuelve primero una serie desfasada (3 días por detrás del tick) y después la serie al día."""
    TIMEFRAME_D1 = 16408

    def __init__(self, primera_vela=LUNES):
        self.llamadas, self.primera_vela = 0, primera_vela

    def copy_rates_range(self, _simbolo, _tf, _desde, _hasta):
        self.llamadas += 1
        n = 5 if self.llamadas == 1 else 8
        return velas_d1([BASE] * n, t0=self.primera_vela).to_records(index=False)

    def symbol_info_tick(self, _simbolo):
        return SimpleNamespace(time=self.primera_vela + 7 * DIA + 3600)   # dentro de la octava vela, en formación


def test_descarga_reintenta_si_la_cache_esta_desfasada(monkeypatch):
    terminal = TerminalConCache()
    monkeypatch.setattr(td, "mt5", terminal)
    velas = td.descargar_mercado("EURUSD", "D1", "2026-10-26", espera_s=0)
    assert terminal.llamadas == 2
    assert len(velas) == 7 and velas["time"].iloc[-1] == LUNES + 6 * DIA     # sin la vela en formación


def test_descarga_avisa_si_la_terminal_recorta_el_historico(monkeypatch):
    monkeypatch.setattr(td, "mt5", TerminalConCache(primera_vela=LUNES + 10 * DIA))
    with pytest.raises(RuntimeError, match="recorta"):
        td.descargar_mercado("EURUSD", "D1", "2026-10-26", espera_s=0)
