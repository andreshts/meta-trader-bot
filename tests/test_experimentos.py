"""El banco de experimentos: ninguna feature nueva mira al futuro, y evaluar una variante no deja rastro en train_model."""
import numpy as np
import pandas as pd
import pytest

import config
import experimentos as ex
import train_model as tm
from features import FEATURES, build_features

LUNES = 1_794_182_400  # 2026-11-09 00:00 UTC


def velas(n=400, semilla=0, inicio=LUNES) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    df = pd.DataFrame({"time": inicio + 900 * np.arange(n), "open": close, "close": close,
                       "high": close * (1 + rng.uniform(0, 0.001, n)), "low": close * (1 - rng.uniform(0, 0.001, n)),
                       "tick_volume": rng.integers(50, 500, n), "spread": 10})
    return df.assign(time_utc=df["time"])


def con_features(df: pd.DataFrame, extra, contexto=None) -> pd.DataFrame:
    return extra(build_features(df, 0.01), contexto or {})


@pytest.mark.parametrize("nombre", ["sesiones", "squeeze", "regimen_feature"])
def test_cambiar_el_futuro_no_cambia_las_features_de_una_vela(nombre):
    variante, corte = ex.VARIANTES[nombre], 300
    original = velas()
    alterado = original.copy()
    alterado.loc[corte + 1:, ["open", "high", "low", "close"]] *= 1.5
    columnas = variante.columnas({})
    antes = con_features(original, variante.extra)[columnas].iloc[:corte + 1]
    despues = con_features(alterado, variante.extra)[columnas].iloc[:corte + 1]
    pd.testing.assert_frame_equal(antes, despues)
    assert antes.iloc[60:].notna().all().all()


def test_las_sesiones_siguen_la_hora_local_de_cada_plaza():
    def banderas(texto_utc: str) -> dict:
        df = pd.DataFrame({"time_utc": [int(pd.Timestamp(texto_utc, tz="UTC").timestamp())]})
        fila = ex.sesiones(df, {}).iloc[0]
        return {c[4:] for c in ("ses_asia", "ses_londres", "ses_ny", "ses_solape") if fila[c]}

    assert banderas("2026-11-09 01:00") == {"asia"}           # 10:00 en Tokio
    assert banderas("2026-11-09 09:00") == {"londres"}
    assert banderas("2026-11-09 15:00") == {"londres", "ny", "solape"}  # 10:00 en Nueva York, horario de invierno
    assert banderas("2026-11-09 13:45") == {"londres"}        # en invierno Nueva York abre a las 14:30 UTC
    assert banderas("2026-07-06 13:45") == {"londres", "ny", "solape"}  # en verano, a las 13:30 UTC
    assert banderas("2026-11-09 22:00") == set()
    assert banderas("2026-11-07 15:00") == set()              # sábado


def test_el_squeeze_se_activa_cuando_el_precio_se_comprime():
    df = velas()
    df.loc[200:, ["open", "close"]] = df.loc[200, "close"]  # cierres planos con el mismo rango de vela
    resultado = con_features(df, ex.squeeze)
    assert resultado["squeeze_activo"].iloc[260:].eq(1).all() and resultado["squeeze_ratio"].iloc[260:].lt(1).all()


def test_el_regimen_distingue_tendencia_de_rango():
    df = velas()
    subida = 100 * 1.002 ** np.arange(len(df))
    for columna, factor in (("open", 1), ("close", 1), ("high", 1.0005), ("low", 0.9995)):
        df[columna] = subida * factor
    resultado = con_features(df, ex.regimen)
    assert resultado["regimen"].iloc[100:].eq(1).all() and resultado["adx"].iloc[100:].gt(ex.ADX_TENDENCIA).all()
    assert set(con_features(velas(), ex.regimen)["regimen"].unique()) <= {-1.0, 0.0, 1.0}


def test_los_otros_activos_solo_aportan_velas_que_ya_cerraron():
    propio, otro = velas(), velas(semilla=1)[["time", "close"]]
    contexto = {"otros": {"OTRO": otro}}
    columnas = ex.VARIANTES["otros_activos"].columnas(contexto)
    assert columnas == ["rel_OTRO_1h", "rel_OTRO_4h"]
    antes = con_features(propio, ex.otros_activos, contexto)[columnas]
    assert antes["rel_OTRO_1h"].iloc[100] == pytest.approx(otro["close"].iloc[100] / otro["close"].iloc[96] - 1)

    corte = 300
    alterado = otro.copy()
    alterado.loc[corte + 1:, "close"] *= 1.5
    despues = con_features(propio, ex.otros_activos, {"otros": {"OTRO": alterado}})[columnas]
    pd.testing.assert_frame_equal(antes.iloc[:corte + 1], despues.iloc[:corte + 1])
    assert not antes.iloc[corte + 1:].equals(despues.iloc[corte + 1:])


def test_un_dato_viejo_o_inexistente_de_otro_activo_vale_cero():
    propio = velas()
    cerrado = velas(semilla=1)[["time", "close"]].iloc[:100]  # el otro activo dejó de cotizar en la vela 99
    resultado = con_features(propio, ex.otros_activos, {"otros": {"OTRO": cerrado}})
    assert resultado["rel_OTRO_1h"].iloc[120] != 0          # hace 5 horas: todavía sirve
    assert resultado["rel_OTRO_1h"].iloc[99 + 97:].eq(0).all()  # más de un día: ya no
    tarde = velas(semilla=1, inicio=LUNES + 900 * 200)[["time", "close"]]
    assert con_features(propio, ex.otros_activos, {"otros": {"OTRO": tarde}})["rel_OTRO_4h"].iloc[:200].eq(0).all()


def test_una_variante_no_deja_rastro_en_train_model():
    originales = list(FEATURES), tm.build_features, tm.ventanas_walk_forward
    with pytest.raises(RuntimeError):
        with ex.con_variante(ex.VARIANTES["sesiones"], {}):
            assert FEATURES[-4:] == ["ses_asia", "ses_londres", "ses_ny", "ses_solape"] and tm.build_features is not originales[1]
            raise RuntimeError("fallo a mitad de la evaluación")
    assert (list(FEATURES), tm.build_features, tm.ventanas_walk_forward) == originales


def test_la_ventana_deslizante_recorta_solo_el_entrenamiento():
    datos = pd.DataFrame({"fin_etiqueta": np.arange(800)})
    ancladas = list(tm.ventanas_walk_forward(datos, 5))
    with ex.con_variante(ex.VARIANTES["deslizante"], {}):
        deslizantes = list(tm.ventanas_walk_forward(datos, 5))
    for (train_a, val_a, test_a), (train_d, val_d, test_d) in zip(ancladas, deslizantes):
        assert len(train_d) == min(len(train_a), 200) and train_d[-1] == train_a[-1]
        assert np.array_equal(val_a, val_d) and np.array_equal(test_a, test_d)


def test_evaluar_usa_solo_lo_anterior_al_holdout_y_restaura_la_configuracion(monkeypatch):
    vistos = {}

    def walk_forward_falso(datos, semilla):
        vistos["hasta"], vistos["columnas"], vistos["holdout"] = datos["time"].max(), list(FEATURES), config.HOLDOUT_DESDE
        trades = np.array([1.0, -1.0, 2.0])
        lados = {lado: {"auc_test": 0.55} for lado in tm.LADOS}
        return [{"trades": trades, "lados": lados, "resumen": tm.resumen(trades)}]

    monkeypatch.setattr(tm, "walk_forward", walk_forward_falso)
    antes = config.HOLDOUT_DESDE, config.SYMBOL, config.MAX_SPREAD_PUNTOS
    fin_de_2025 = int(pd.Timestamp("2025-12-31", tz="UTC").timestamp())
    paquete = {"velas": velas(600, inicio=fin_de_2025 - 900 * 300).drop(columns="time_utc"), "point": 0.01, "costes": {}}
    filas = list(ex.evaluar("squeeze", "XAUUSDm", paquete, {}, semillas=(42, 1)))
    assert [f["semilla"] for f in filas] == [42, 1] and filas[0]["trades"] == 3 and filas[0]["auc_long"] == 0.55
    assert filas[0]["features"] == len(FEATURES) + 2 and vistos["columnas"][-2:] == ["squeeze_ratio", "squeeze_activo"]
    assert vistos["holdout"] == ex.HOLDOUT_DESDE and vistos["hasta"] < int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
    assert (config.HOLDOUT_DESDE, config.SYMBOL, config.MAX_SPREAD_PUNTOS) == antes


def test_el_registro_se_acumula_y_la_tabla_compara_con_la_base(tmp_path):
    ruta = tmp_path / "experimentos" / "registro.csv"
    assert ex.leer_registro(ruta) == []

    def fila(variante, semilla, t):
        return {"fecha": "2026-11-10T10:00:00", "variante": variante, "simbolo": "XAUUSDm", "semilla": semilla, "features": 11,
                "filas": 1000, "trades": 100, "esperanza_R": 0.1, "t": t, "profit_factor": 1.1, "ventanas_positivas": 0.6,
                "auc_long": 0.55, "auc_short": 0.56}

    for variante, ts in ((ex.BASE, (-1.0, -2.0, 0.0)), ("squeeze", (2.5, 2.1, 0.0)), ("squeeze", (2.5, 2.1, 2.3))):
        for semilla, t in zip((42, 1, 2), ts):
            ex.apuntar(fila(variante, semilla, t), ruta)
    filas = ex.leer_registro(ruta)
    assert len(filas) == 9 and filas[0]["variante"] == ex.BASE
    lineas = ex.tabla(filas).splitlines()
    base = next(l for l in lineas if l.startswith("base"))
    assert "-1.00 / -2.00 / +0.00" in base and base.split()[-1] == "no"
    nueva = next(l for l in lineas if l.startswith("squeeze"))  # cuenta la última evaluación de cada semilla
    assert "+2.50 / +2.10 / +2.30" in nueva and "+3.30" in nueva and nueva.split()[-1] == "sí"
    assert "en 1 de 1 activos" in lineas[-1]
