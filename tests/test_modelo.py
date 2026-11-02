"""Pruebas de features.py y train_model.py con datos sintéticos (sin MT5)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import config
import features
import train_model as tm
from conftest import POINT, velas_aleatorias, velas_con_senal, velas_planas

BASE = 1.1000
MIERCOLES_23H = int(datetime(2026, 10, 28, 23, tzinfo=timezone.utc).timestamp())


# ---------------- features ----------------
def test_rsi_coincide_con_la_formula_clasica():
    df = velas_aleatorias(600)
    f = features.build_features(df, POINT)
    d = df["close"].diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    p = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    assert np.allclose(f["rsi"].iloc[1:], ((100 - 100 / (1 + g / p)) / 100).iloc[1:])
    assert f["rsi"].dropna().between(0, 1).all()


def test_sin_nan_tras_el_calentamiento():
    f = features.build_features(velas_aleatorias(600), POINT)
    assert f[features.FEATURES].iloc[config.WARMUP:].notna().all().all()


def test_features_causales():
    df = velas_aleatorias(600)
    df2 = df.copy()
    df2.loc[400:, ["open", "high", "low", "close"]] *= 1.05
    f, f2 = features.build_features(df, POINT), features.build_features(df2, POINT)
    assert np.allclose(f[features.FEATURES].iloc[100:400], f2[features.FEATURES].iloc[100:400])


def test_200_velas_en_vivo_igual_que_el_historico():
    df = velas_aleatorias(600)
    vivo = features.build_features(df.iloc[-200:].reset_index(drop=True), POINT)[features.FEATURES].iloc[-1]
    historico = features.build_features(df, POINT)[features.FEATURES].iloc[-1]
    assert (vivo - historico).abs().max() < 1e-6


def test_features_de_tiempo():
    # 1970-01-05 00:00 (lunes), 2026-10-29 15:00 (jueves), 2026-11-01 23:00 (domingo)
    df = pd.DataFrame({"time": [4 * 86400, 1793286000, 1793574000], "open": 1.1, "high": 1.1, "low": 1.1,
                       "close": 1.1, "tick_volume": 100, "spread": 0})
    f = features.build_features(df, POINT)
    assert list(f["dia_semana"]) == [0, 3, 6]
    assert np.allclose(f["hora_sin"], np.sin(2 * np.pi * np.array([0, 15, 23]) / 24))
    assert np.allclose(f["hora_cos"], np.cos(2 * np.pi * np.array([0, 15, 23]) / 24))


def test_adx_wilder_acotado_direccional_y_causal():
    df = velas_aleatorias(600)
    adx, plus_di, minus_di = features.adx_wilder(df)
    f = pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})
    cols_adx = ["adx", "plus_di", "minus_di"]
    assert f[cols_adx].notna().all().all()
    for c in cols_adx:
        assert f[c].between(0.0, 1.0).all()
    assert (f["plus_di"] + f["minus_di"] <= 1.0 + 1e-12).all()

    # Causalidad estricta: perturbar desde 400 no altera 0..399
    df2 = df.copy()
    df2.loc[400:, ["open", "high", "low", "close"]] *= 1.05
    f2 = pd.DataFrame(dict(zip(cols_adx, features.adx_wilder(df2))))
    assert np.allclose(f[cols_adx].iloc[:400], f2[cols_adx].iloc[:400])

    # Tendencia alcista monotónica -> +DI > -DI y ADX alto; bajista -> -DI > +DI
    n = 200
    subida = pd.DataFrame({
        "time": 1_700_000_000 + 900 * np.arange(n),
        "open": 1.10 + 0.0005 * np.arange(n),
        "close": 1.10 + 0.0005 * (np.arange(n) + 1),
        "high": 1.10 + 0.0005 * (np.arange(n) + 1) + 0.0001,
        "low": 1.10 + 0.0005 * np.arange(n) - 0.0001,
        "tick_volume": 300.0,
        "spread": 5,
    })
    bajada = subida.copy()
    bajada["open"] = 1.20 - 0.0005 * np.arange(n)
    bajada["close"] = 1.20 - 0.0005 * (np.arange(n) + 1)
    bajada["high"] = bajada["open"] + 0.0001
    bajada["low"] = bajada["close"] - 0.0001
    adx_s, pdi_s, mdi_s = features.adx_wilder(subida)
    adx_b, pdi_b, mdi_b = features.adx_wilder(bajada)
    assert pdi_s.iloc[-1] > mdi_s.iloc[-1] and adx_s.iloc[-1] > 0.7
    assert mdi_b.iloc[-1] > pdi_b.iloc[-1] and adx_b.iloc[-1] > 0.7


def test_mtf_cerrado_sin_fuga_en_vela_en_formacion():
    """Verifica que para la vela M15 `t`, H1 y H4 solo usan bloques superiores ya cerrados al cierre de `t`.
    En particular, a las :00, :15 y :30 la vela H1 en curso NO se usa; solo al cierre de :45 (que es :00 de la
    hora siguiente) queda disponible esa vela H1."""
    # Alineamos t0 a un múltiplo exacto de 4 horas (14400 s)
    t0 = (1_700_000_000 // 14400) * 14400
    df = velas_aleatorias(600)
    df["time"] = t0 + 900 * np.arange(len(df))

    def calcular_mtf(d):
        rsi_h1, dist_h1 = features.mtf_cerrado(d, features.SEGUNDOS_H1)
        rsi_h4, dist_h4 = features.mtf_cerrado(d, features.SEGUNDOS_H4)
        return pd.DataFrame({
            "rsi_h1": rsi_h1, "dist_sma20_h1": dist_h1,
            "rsi_h4": rsi_h4, "dist_sma20_h4": dist_h4,
        })

    f = calcular_mtf(df)
    cols_mtf = ["rsi_h1", "dist_sma20_h1", "rsi_h4", "dist_sma20_h4"]

    # En las primeras 3 velas (00:00, 00:15, 00:30) aún no ha cerrado ningún bloque H1; en la 4ª (00:45 -> cierra 01:00) sí
    assert f["rsi_h1"].iloc[:3].isna().all()
    # Dentro de una hora (k, k+1, k+2 = :00, :15, :30), el valor H1 debe ser constante e igual al de k-1 (:45 anterior)
    for k in range(120, 500, 4):
        assert np.isclose(f["rsi_h1"].iloc[k - 1], f["rsi_h1"].iloc[k])
        assert np.isclose(f["rsi_h1"].iloc[k], f["rsi_h1"].iloc[k + 1])
        assert np.isclose(f["rsi_h1"].iloc[k], f["rsi_h1"].iloc[k + 2])
    # Y tras el calentamiento de H4 (20 bloques H4 = 320 velas M15), no hay NaN
    assert f[cols_mtf].iloc[320:].notna().all().all()

    # Prueba anti-fuga en mitad de una vela H1 y H4 en formación (corte en :15, índice 401):
    # Perturbar desde 402 en adelante (que pertenecen a la MISMA hora H1 y al MISMO bloque H4 que 400 y 401)
    # NO puede cambiar las features MTF en 0..401.
    for corte in (400, 401, 402, 403):
        df_pert = df.copy()
        df_pert.loc[corte + 1:, ["open", "high", "low", "close"]] *= 1.25
        f_pert = calcular_mtf(df_pert)
        pd.testing.assert_frame_equal(f[cols_mtf].iloc[:corte + 1], f_pert[cols_mtf].iloc[:corte + 1])


# ---------------- etiquetado ----------------
def barras(filas, atr=0.0010, spread=0, t0=MIERCOLES_23H - 900):
    d = pd.DataFrame(filas, columns=["open", "high", "low", "close"])
    d["spread"], d["atr"], d["time"] = spread, atr, t0 + 900 * np.arange(len(d))
    return d


def etiquetar(d, **kw):
    return tm.etiquetar_triple_barrera(d, POINT, 1.5, 3.0, 12, **kw).iloc[0]


PLANO = [[BASE] * 4]


def test_long_toca_tp_y_short_toca_sl():
    e = etiquetar(barras(PLANO + [[BASE, BASE + 0.0010, BASE - 0.0005, BASE + 0.0008],
                                  [BASE + 0.0008, BASE + 0.0035, BASE + 0.0007, BASE + 0.0030]] + PLANO * 12))
    assert e["y_long"] == 1 and np.isclose(e["r_long"], 2.0) and e["k_long"] == 2 and not e["to_long"]
    assert e["y_short"] == 0 and np.isclose(e["r_short"], -1.0) and e["k_short"] == 2


def test_vela_ambigua_cuenta_como_sl():
    e = etiquetar(barras(PLANO + [[BASE, BASE + 0.0040, BASE - 0.0020, BASE]] + PLANO * 12))
    assert np.isclose(e["r_long"], -1.0) and np.isclose(e["r_short"], -1.0)


def test_time_stop_cierra_en_la_apertura_siguiente():
    # Cierre de la vela 12 en +0.0006, pero la vela 13 abre en +0.0009: el bot cierra a ese precio.
    d = barras(PLANO + [[BASE, BASE + 0.0006, BASE, BASE + 0.0006]] + [[BASE + 0.0006] * 4] * 11
               + [[BASE + 0.0009] * 4] * 2)
    e = etiquetar(d)
    assert e["to_long"] and e["k_long"] == 12 and e["y_long"] == 0
    assert np.isclose(e["r_long"], 0.0009 / 0.0015)
    assert e["fin_etiqueta"] == 13 and e["t_fin_etiqueta"] == d["time"].iloc[13]   # depende de la vela t+N+1


def test_fin_de_etiqueta_con_sl_o_tp():
    e = etiquetar(barras(PLANO + [[BASE, BASE + 0.0010, BASE - 0.0005, BASE + 0.0008],
                                  [BASE + 0.0008, BASE + 0.0035, BASE + 0.0007, BASE + 0.0030]] + PLANO * 12))
    assert e["fin_etiqueta"] == 2      # los dos lados terminan en la vela 2 (TP del long, SL del short)


def test_sin_horizonte_completo_es_nan():
    e = tm.etiquetar_triple_barrera(barras(PLANO * 14), POINT, 1.5, 3.0, 12)
    assert e["y_long"].notna().sum() == 1 and np.isnan(e.at[1, "y_long"])


def test_gap_a_traves_del_sl():
    e = etiquetar(barras(PLANO * 2 + [[BASE - 0.0030] * 4] * 13))
    assert np.isclose(e["r_long"], -2.0)
    assert e["y_short"] == 1 and np.isclose(e["r_short"], 2.0)


def test_spread_en_long_y_short():
    # Long entra al ask (+0.0001): high +0.0030 no llega al TP.
    e = etiquetar(barras(PLANO + [[BASE, BASE + 0.0030, BASE, BASE]] + PLANO * 12, spread=10))
    assert e["y_long"] == 0 and e["to_long"] and np.isclose(e["r_long"], -0.0001 / 0.0015)
    # Short: el SL se dispara con el ask (high + spread).
    e = etiquetar(barras(PLANO + [[BASE, BASE + 0.0014, BASE, BASE]] + PLANO * 12, spread=10))
    assert e["y_short"] == 0 and np.isclose(e["r_short"], -1.0) and not e["to_short"]


def test_comision_y_deslizamiento():
    d = barras(PLANO + [[BASE, BASE + 0.0035, BASE, BASE]] + PLANO * 12)
    assert np.isclose(etiquetar(d, comision_puntos=15)["r_long"], 1.9)          # 0.00015 / 0.0015
    assert np.isclose(etiquetar(d, slippage_puntos=7.5)["r_long"], 1.9)         # entrada + salida
    assert np.isclose(etiquetar(d, comision_puntos=15, slippage_puntos=7.5)["r_long"], 1.8)


def test_swap_triple_del_miercoles():
    # Entrada el miércoles a las 23:00, time stop el jueves: cruza el rollover triple.
    d = barras(PLANO * 15, t0=MIERCOLES_23H - 900)
    e = etiquetar(d, swap_long_puntos=-0.7, swap_short_puntos=-1.0, swap_3dias=3)
    assert np.isclose(e["r_long"], 3 * -0.7 * POINT / 0.0015)
    assert np.isclose(e["r_short"], 3 * -1.0 * POINT / 0.0015)


@pytest.mark.parametrize("entrada, salida, noches", [
    ("2026-10-28 23:00", "2026-10-29 01:00", 3),   # miércoles -> jueves (triple)
    ("2026-10-26 22:00", "2026-10-27 02:00", 1),   # lunes -> martes
    ("2026-10-30 20:00", "2026-11-02 02:00", 1),   # viernes -> lunes: solo cobra la noche del viernes
    ("2026-10-27 01:00", "2026-10-27 23:00", 0),   # mismo día
    ("2026-10-26 22:00", "2026-10-27 00:00", 1),   # salida justo a las 00:00: paga el rollover
])
def test_noches_de_swap(entrada, salida, noches):
    segundos = lambda s: int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())  # noqa: E731
    assert tm.noches_de_swap(segundos(entrada), segundos(salida), 3) == noches


# ---------------- walk-forward, simulación y preparación ----------------
def _filas(velas_de_cada_fila, horizonte):
    """Filas ya preparadas: índice = vela de la fila, y su etiqueta llega hasta `horizonte` velas después."""
    velas_de_cada_fila = np.asarray(velas_de_cada_fila)
    return pd.DataFrame({"fin_etiqueta": velas_de_cada_fila + horizonte}, index=velas_de_cada_fila)


def test_ventanas_walk_forward():
    vs = list(tm.ventanas_walk_forward(_filas(np.arange(1000), 13), 5))   # 8 segmentos de 125; N=12 -> t+13
    tr, va, te = vs[0]
    assert (tr[0], tr[-1], va[0], va[-1], te[0], te[-1]) == (0, 236, 250, 361, 375, 499)   # purga de 13, no 12
    tr, va, te = vs[-1]
    assert (tr[-1], va[0], va[-1], te[0], te[-1]) == (736, 750, 861, 875, 999)
    assert np.array_equal(np.concatenate([v[2] for v in vs]), np.arange(375, 1000))


def test_la_purga_cuenta_velas_y_no_filas():
    # Los filtros quitan filas: aquí solo queda una vela de cada dos, y la etiqueta llega 13 velas más allá.
    datos = _filas(np.arange(0, 2000, 2), 13)
    vela, fin = datos.index.to_numpy(), datos["fin_etiqueta"].to_numpy()
    for tr, va, te in tm.ventanas_walk_forward(datos, 5):
        assert (fin[tr] < vela[va[0]]).all() and (fin[va] < vela[te[0]]).all()
        assert fin[tr[-1] + 1] >= vela[va[0]] and fin[va[-1] + 1] >= vela[te[0]]   # y no se purga de más


ETIQUETAS = ["y_long", "y_short", "r_long", "r_short", "k_long", "k_short", "to_long", "to_short"]


def _perturbar_vela(velas, posicion):
    velas = velas.copy()
    velas.loc[posicion, ["open", "high", "low", "close"]] += 0.01
    return velas


@pytest.mark.parametrize("generar", [velas_planas, velas_con_senal], ids=["todo-time-stop", "senal"])
def test_ninguna_etiqueta_de_train_o_validation_depende_del_test(generar):
    """Regresión: con el cierre por tiempo en la apertura t+N+1, purgar N filas dejaba que la última operación
    de validation dependiera de la primera vela del test (y esa operación interviene al elegir el umbral).
    Con precio plano todas las operaciones cierran por tiempo, así que la fuga aparece seguro si existe."""
    velas = generar()
    datos = tm.preparar_datos(velas, POINT)
    assert generar is not velas_planas or datos["to_long"].all()
    for tr, va, te in tm.ventanas_walk_forward(datos, config.WF_VENTANAS):
        perturbado = tm.preparar_datos(_perturbar_vela(velas, datos.index[te[0]]), POINT)
        for filas in (tr, va):
            indice = datos.index[filas]
            pd.testing.assert_frame_equal(datos.loc[indice, ETIQUETAS], perturbado.loc[indice, ETIQUETAS])


@pytest.mark.parametrize("generar, corte", [(velas_planas, "2023-12-10"), (velas_con_senal, "2024-03-25")],
                         ids=["todo-time-stop", "senal"])
def test_ninguna_etiqueta_del_walk_forward_depende_del_holdout(monkeypatch, generar, corte):
    monkeypatch.setattr(config, "HOLDOUT_DESDE", corte)
    velas = generar()
    wf, holdout = tm.separar_holdout(tm.preparar_datos(velas, POINT))
    primera_del_holdout = int(velas.index[velas["time"] >= holdout["time"].iloc[0]][0])
    wf_perturbado, _ = tm.separar_holdout(tm.preparar_datos(_perturbar_vela(velas, primera_del_holdout), POINT))
    pd.testing.assert_frame_equal(wf[ETIQUETAS], wf_perturbado.loc[wf.index, ETIQUETAS])


def test_simular_mono_posicion():
    filas = pd.DataFrame({"r_long": [2.0, 9, 9, -1.0, 9, 0.3, 9, 9], "k_long": [2, 0, 0, 3, 0, 6, 0, 0],
                          "to_long": [False] * 8, "r_short": [0.0] * 8, "k_short": [0] * 8, "to_short": [False] * 8})
    unos, ceros = np.ones(8), np.zeros(8)
    assert list(tm.simular(filas, unos, ceros, 0.5, None)[:2]) == [2.0, 9.0]   # tras SL/TP en k, libre desde k
    filas.loc[0, ["r_long", "k_long", "to_long"]] = [0.3, 4, True]
    assert list(tm.simular(filas, unos, ceros, 0.5, None)[:2]) == [0.3, 0.3]   # tras time stop, libre desde k+1
    assert len(tm.simular(filas, unos, unos, 0.5, 0.5)) == 0                   # ambas señales: no opera


def test_preparar_datos_quita_entradas_con_spread_alto():
    velas = velas_con_senal()
    caras = [500, 501, 2000, 7777]
    velas.loc[caras, "spread"] = config.MAX_SPREAD_PUNTOS + 1
    velas.loc[3000, "spread"] = config.MAX_SPREAD_PUNTOS                    # en el límite: se permite
    prep, completo = tm.preparar_datos(velas, POINT), tm.preparar_datos(velas_con_senal(), POINT)
    assert all(i - 1 not in prep.index for i in caras)
    assert all(i in prep.index for i in (501, 2000, 7777))
    assert 2999 in prep.index and len(completo) - len(prep) == len(caras)
    assert prep.index[0] >= config.WARMUP


def test_preparar_datos_quita_entradas_bajo_el_nivel_de_stops():
    velas = velas_con_senal()
    assert len(tm.preparar_datos(velas, POINT, {"stops_level": 10**6})) == 0
    assert len(tm.preparar_datos(velas, POINT, {"stops_level": 0})) == len(tm.preparar_datos(velas, POINT))


def test_separar_holdout(monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_DESDE", "2024-03-25")
    datos = tm.preparar_datos(velas_con_senal(), POINT)
    wf, holdout = tm.separar_holdout(datos)
    corte = int(datetime(2024, 3, 25, tzinfo=timezone.utc).timestamp())
    assert (wf["time"] < corte).all() and (holdout["time"] >= corte).all() and len(holdout) > 0
    assert (wf["t_fin_etiqueta"] < corte).all()   # purga: ninguna etiqueta mira dentro (las que acabaron pronto
                                                  # por SL/TP antes del corte se conservan, con razón)
    monkeypatch.setattr(config, "HOLDOUT_DESDE", None)
    wf, holdout = tm.separar_holdout(datos)
    assert len(wf) == len(datos) and len(holdout) == 0


def test_resumen_t_stat():
    r = tm.resumen(np.array([2.0, -1.0, 2.0, -1.0]))
    assert np.isclose(r["error_estandar_R"], np.sqrt(3) / 2) and np.isclose(r["t_stat"], 0.5 / (np.sqrt(3) / 2))
    assert tm.resumen(np.array([2.0, 2.0, 2.0]))["t_stat"] == np.inf
    assert tm.resumen(np.array([])) == {"trades": 0}


# ---------------- pipeline completo (lento: 3 semillas x 5 ventanas) ----------------
@pytest.fixture
def holdout_sintetico(monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_DESDE", "2024-03-25")


def test_senal_plantada_pasa_walk_forward_y_holdout(holdout_sintetico):
    bundle = tm.entrenar(velas_con_senal(), POINT)
    informe = bundle["informe"]
    assert bundle["umbral_long"] is not None
    assert [r["semilla"] for r in informe["robustez"]] == list(config.SEMILLAS_ROBUSTEZ)
    assert informe["holdout"]["trades"] >= config.MIN_TRADES_TEST and informe["holdout"]["t_stat"] > 2
    assert tm.ventaja_significativa(informe)


def test_ruido_no_pasa(holdout_sintetico):
    ruido = velas_aleatorias(15_000, semilla=7)
    informe = tm.entrenar(ruido, POINT)["informe"]
    assert not tm.ventaja_significativa(informe)
