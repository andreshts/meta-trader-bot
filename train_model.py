"""Entrenamiento de los modelos Long y Short con validación walk-forward (ver Entrenamiento.md).

Uso: python train_model.py
"""
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import config
from features import FEATURES, build_features

LADOS = ("long", "short")
GRID_UMBRALES = np.round(np.arange(0.20, 0.901, 0.01), 2)


def noches_de_swap(t_entrada: int, t_salida: int, dia_triple: int) -> int:
    """Rollovers (00:00 del servidor) entre la entrada y la salida, contando 3 el que cierra `dia_triple`.

    Solo cobran los rollovers que cierran un día de lunes a viernes. Los días se numeran como en MT5
    (0 = domingo, 3 = miércoles). Una salida justo a las 00:00 paga ese rollover (criterio conservador).
    """
    noches = 0
    for dia in range(t_entrada // 86400 + 1, t_salida // 86400 + 1):
        anterior = (dia - 1 + 4) % 7  # 1970-01-01 fue jueves (4)
        if 1 <= anterior <= 5:
            noches += 3 if anterior == dia_triple else 1
    return noches


def etiquetar_triple_barrera(df: pd.DataFrame, point: float, sl_mult: float, tp_mult: float, n: int,
                             comision_puntos: float = 0.0, slippage_puntos: float = 0.0,
                             swap_long_puntos: float = 0.0, swap_short_puntos: float = 0.0,
                             swap_3dias: int = 3) -> pd.DataFrame:
    """Etiqueta cada fila `t` simulando una entrada en la apertura de `t+1`.

    Por lado devuelve:
      y_<lado>  1 si toca TP antes que SL dentro de `n` velas, si no 0 (NaN sin horizonte completo)
      r_<lado>  resultado en múltiplos del riesgo (R), neto de costes
      k_<lado>  posición de la última vela con el trade abierto
      to_<lado> True si se cerró por time stop
    Las velas son bid: la compra entra al ask y sale al bid; la venta entra al bid y sale al ask.
    Si una vela toca TP y SL a la vez, cuenta como SL. El time stop cierra en la apertura de la vela
    `t+n+1`, como el bot, que cierra al detectar esa vela. Costes: comisión y deslizamiento (en la entrada
    y en la salida), en puntos, y el swap de cada rollover según `noches_de_swap`.
    """
    o, h, l = (df[col].to_numpy(dtype=float) for col in ("open", "high", "low"))
    spr = df["spread"].to_numpy(dtype=float) * point
    atr = df["atr"].to_numpy(dtype=float)
    tiempos = df["time"].to_numpy(dtype=np.int64)
    m = len(df)

    res = {f"{campo}_{lado}": np.full(m, np.nan) for lado in LADOS for campo in ("y", "r", "k")}
    res.update({f"to_{lado}": np.zeros(m, dtype=bool) for lado in LADOS})

    for t in range(m - n - 1):
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        e = t + 1
        cierre = e + n  # vela en cuya apertura cierra el time stop
        sl_dist, tp_dist = sl_mult * atr[t], tp_mult * atr[t]
        coste_r = (comision_puntos + 2 * slippage_puntos) * point / sl_dist

        # Long: entra al ask, SL/TP se disparan con el bid.
        entrada = o[e] + spr[e]
        sl, tp = entrada - sl_dist, entrada + tp_dist
        y, salida, k, timeout = 0, o[cierre], cierre - 1, True
        for j in range(e, cierre):
            if l[j] <= sl:
                y, salida, k, timeout = 0, min(o[j], sl), j, False
                break
            if h[j] >= tp:
                y, salida, k, timeout = 1, tp, j, False
                break
        noches = noches_de_swap(tiempos[e], tiempos[cierre if timeout else k], swap_3dias)
        res["y_long"][t], res["k_long"][t], res["to_long"][t] = y, k, timeout
        res["r_long"][t] = (salida - entrada + noches * swap_long_puntos * point) / sl_dist - coste_r

        # Short: entra al bid, SL/TP se disparan con el ask (bid + spread).
        entrada = o[e]
        sl, tp = entrada + sl_dist, entrada - tp_dist
        y, salida, k, timeout = 0, o[cierre] + spr[cierre], cierre - 1, True
        for j in range(e, cierre):
            if h[j] + spr[j] >= sl:
                y, salida, k, timeout = 0, max(o[j] + spr[j], sl), j, False
                break
            if l[j] + spr[j] <= tp:
                y, salida, k, timeout = 1, tp, j, False
                break
        noches = noches_de_swap(tiempos[e], tiempos[cierre if timeout else k], swap_3dias)
        res["y_short"][t], res["k_short"][t], res["to_short"][t] = y, k, timeout
        res["r_short"][t] = (entrada - salida + noches * swap_short_puntos * point) / sl_dist - coste_r

    return pd.DataFrame(res, index=df.index)


def ventanas_walk_forward(n_filas: int, n_ventanas: int, purga: int):
    """Ventanas ancladas sobre `n_ventanas + 3` segmentos iguales del histórico.

    La ventana `k` entrena con los segmentos `[0, k+2)`, valida con el `k+2` y testea con el `k+3`
    (la última absorbe el resto). Se purgan `purga` filas al final de train y de validation.
    """
    seg = n_filas // (n_ventanas + 3)
    for k in range(n_ventanas):
        ini_val, ini_test = (k + 2) * seg, (k + 3) * seg
        fin_test = n_filas if k == n_ventanas - 1 else (k + 4) * seg
        yield (np.arange(0, ini_val - purga),
               np.arange(ini_val, ini_test - purga),
               np.arange(ini_test, fin_test))


def nuevo_modelo(semilla: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=25,
        max_features="sqrt",
        random_state=semilla,
        n_jobs=-1,
    )


def calibrar(modelo, X_val, y_val):
    """Calibración sigmoide sobre validation, sin reentrenar el bosque."""
    try:
        from sklearn.frozen import FrozenEstimator
        calibrado = CalibratedClassifierCV(FrozenEstimator(modelo), method="sigmoid")
    except ImportError:  # scikit-learn < 1.6
        calibrado = CalibratedClassifierCV(modelo, method="sigmoid", cv="prefit")
    return calibrado.fit(X_val, y_val)


def simular(filas: pd.DataFrame, p_long, p_short, umbral_long, umbral_short) -> np.ndarray:
    """Backtest mono-posición que reproduce el bucle de trading_bot.py; devuelve los R de cada trade.

    Tras SL/TP en la vela `k` se puede volver a entrar con la señal de la fila `k`; tras un time stop,
    el bot dedica la vela siguiente a cerrar, así que la siguiente señal válida es la de `k + 1`.
    """
    pos = filas.index.to_numpy()
    r = {lado: filas[f"r_{lado}"].to_numpy() for lado in LADOS}
    k = {lado: filas[f"k_{lado}"].to_numpy() for lado in LADOS}
    to = {lado: filas[f"to_{lado}"].to_numpy() for lado in LADOS}
    sin_señal = np.zeros(len(pos), dtype=bool)
    señal_long = np.asarray(p_long) >= umbral_long if umbral_long is not None else sin_señal
    señal_short = np.asarray(p_short) >= umbral_short if umbral_short is not None else sin_señal

    resultados = []
    libre = -1
    for i in np.flatnonzero(señal_long != señal_short):
        if pos[i] < libre:
            continue
        lado = "long" if señal_long[i] else "short"
        resultados.append(r[lado][i])
        libre = int(k[lado][i]) + (1 if to[lado][i] else 0)
    return np.array(resultados)


def resumen(trades: np.ndarray) -> dict:
    if len(trades) == 0:
        return {"trades": 0}
    curva = np.cumsum(trades)
    ganancias, perdidas = trades[trades > 0].sum(), -trades[trades < 0].sum()
    error_estandar = trades.std(ddof=1) / np.sqrt(len(trades)) if len(trades) > 1 else np.inf
    if error_estandar > 0:
        t_stat = trades.mean() / error_estandar
    else:  # todos los trades idénticos
        t_stat = np.inf if trades.mean() > 0 else 0.0
    return {
        "trades": len(trades),
        "acierto": float((trades > 0).mean()),
        "esperanza_R": float(trades.mean()),
        "error_estandar_R": float(error_estandar),
        "t_stat": float(t_stat),
        "profit_factor": float(ganancias / perdidas) if perdidas > 0 else float("inf"),
        "max_drawdown_R": float((np.maximum.accumulate(np.maximum(curva, 0)) - curva).max()),
        "total_R": float(curva[-1]),
    }


def _proporcion_positivas(walk_forward: dict) -> float:
    return float(np.mean([v.get("esperanza_R", 0.0) > 0 for v in walk_forward["ventanas"]]))


def proporcion_ventanas_positivas(informe: dict) -> float:
    return _proporcion_positivas(informe["walk_forward"])


def pasa_walk_forward(walk_forward: dict) -> bool:
    """Un walk-forward pasa si, sobre todos sus trades fuera de muestra, hay un mínimo de trades y una
    esperanza de al menos T_MINIMO_TEST errores estándar sobre 0, y además es positiva en suficientes ventanas."""
    agregado = walk_forward["agregado"]
    return (agregado.get("trades", 0) >= config.MIN_TRADES_TEST
            and agregado.get("t_stat", 0.0) >= config.T_MINIMO_TEST
            and _proporcion_positivas(walk_forward) >= config.WF_MIN_PROP_VENTANAS_POSITIVAS)


def pasa_holdout(holdout: dict) -> bool:
    return (holdout.get("trades", 0) >= config.MIN_TRADES_TEST
            and holdout.get("t_stat", 0.0) >= config.T_MINIMO_TEST)


def ventaja_significativa(informe: dict) -> bool:
    """Criterio de despliegue: el walk-forward debe pasar con todas las semillas de SEMILLAS_ROBUSTEZ, y el
    modelo guardado debe pasar también en el holdout, que no se usó para elegir nada.

    Un único entrenamiento cerca del umbral puede pasar por suerte (la semilla del bosque cambia qué trades
    superan el umbral), y el walk-forward se ha reutilizado en muchos experimentos. Los modelos sin la
    comprobación de robustez o sin holdout, o con menos semillas, no pasan.
    """
    if "robustez" not in informe or "holdout" not in informe:
        return False
    por_semilla = {r["semilla"]: r["walk_forward"] for r in informe["robustez"]}
    return (all(s in por_semilla and pasa_walk_forward(por_semilla[s]) for s in config.SEMILLAS_ROBUSTEZ)
            and pasa_holdout(informe["holdout"]))


def elegir_umbral(filas: pd.DataFrame, proba: np.ndarray, lado: str, min_trades: int):
    """Umbral del grid con mayor esperanza en un backtest de un solo lado; None si ninguno es rentable."""
    mejor_umbral, mejor_esperanza = None, 0.0
    ceros = np.zeros(len(proba))
    for umbral in GRID_UMBRALES:
        if lado == "long":
            trades = simular(filas, proba, ceros, umbral, None)
        else:
            trades = simular(filas, ceros, proba, None, umbral)
        if len(trades) >= min_trades and trades.mean() > mejor_esperanza:
            mejor_umbral, mejor_esperanza = float(umbral), float(trades.mean())
    return mejor_umbral


COSTES_POR_DEFECTO = {"swap_long_puntos": 0.0, "swap_short_puntos": 0.0, "swap_3dias": 3, "stops_level": 0}


def costes_de_simbolo(info) -> dict:
    """Swap y nivel mínimo de stops del símbolo (symbol_info). Solo se modela el swap en puntos."""
    swap_en_puntos = info.swap_mode == 1  # SYMBOL_SWAP_MODE_POINTS
    if not swap_en_puntos and info.swap_mode != 0:
        print(f"AVISO: swap_mode {info.swap_mode} no soportado; el backtest ignora el swap")
    return {
        "swap_long_puntos": float(info.swap_long) if swap_en_puntos else 0.0,
        "swap_short_puntos": float(info.swap_short) if swap_en_puntos else 0.0,
        "swap_3dias": int(info.swap_rollover3days),
        "stops_level": int(info.trade_stops_level),
    }


def preparar_datos(velas: pd.DataFrame, point: float, costes: dict | None = None) -> pd.DataFrame:
    """Features y etiquetas, sin el calentamiento inicial ni las filas sin horizonte completo.

    También quita las entradas que los filtros del bot no permitirían, para que no entrenen ni cuenten en el
    backtest: spread de la vela de entrada (`t+1`) mayor que MAX_SPREAD_PUNTOS, o SL más cerca del precio que
    el nivel mínimo de stops más el spread. El spread de la vela es una aproximación del de la entrada real.
    """
    costes = {**COSTES_POR_DEFECTO, **(costes or {})}
    velas = velas.reset_index(drop=True)
    datos = build_features(velas, point)
    etiquetas = etiquetar_triple_barrera(datos, point, config.SL_MULTIPLICADOR_ATR, config.TP_MULTIPLICADOR_ATR,
                                         config.MAX_BARS_EN_TRADE, config.COMISION_PUNTOS, config.SLIPPAGE_PUNTOS,
                                         costes["swap_long_puntos"], costes["swap_short_puntos"], costes["swap_3dias"])
    datos = pd.concat([datos, etiquetas], axis=1)
    datos["spread_entrada"] = datos["spread"].shift(-1)
    datos = datos.iloc[config.WARMUP:]
    distancia_minima = (costes["stops_level"] + datos["spread_entrada"]) * point
    operable = ((datos["spread_entrada"] <= config.MAX_SPREAD_PUNTOS)
                & (config.SL_MULTIPLICADOR_ATR * datos["atr"] > distancia_minima))
    return datos[operable].dropna(subset=FEATURES + ["y_long", "y_short"])


def separar_holdout(datos: pd.DataFrame):
    """(walk-forward, holdout) según HOLDOUT_DESDE.

    Se purgan las N últimas filas antes del corte, para que ninguna etiqueta del walk-forward use precios del
    holdout. Con HOLDOUT_DESDE = None no hay holdout, y el criterio de despliegue no puede cumplirse.
    """
    if config.HOLDOUT_DESDE is None:
        return datos, datos.iloc[0:0]
    corte = int(datetime.strptime(config.HOLDOUT_DESDE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    previo = datos[datos["time"] < corte]
    return previo.iloc[:-config.MAX_BARS_EN_TRADE], datos[datos["time"] >= corte]


def entrenar_ventana(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, semilla: int = 42) -> dict:
    """Entrena en train, calibra y elige umbrales en validation, y evalúa en test."""
    modelos, umbrales, proba_test, lados = {}, {}, {}, {}
    for lado in LADOS:
        y = f"y_{lado}"
        bosque = nuevo_modelo(semilla).fit(train[FEATURES], train[y].astype(int))
        modelo = calibrar(bosque, val[FEATURES], val[y].astype(int))
        umbral = elegir_umbral(val, modelo.predict_proba(val[FEATURES])[:, 1], lado, config.MIN_TRADES_UMBRAL)
        p_test = modelo.predict_proba(test[FEATURES])[:, 1]

        y_test = test[y].astype(int).to_numpy()
        señales = p_test >= umbral if umbral is not None else np.zeros(len(p_test), dtype=bool)
        lados[lado] = {
            "tasa_base_train": float(train[y].mean()),
            "umbral": umbral,
            "auc_test": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) == 2 else None,
            "precision_test": float(y_test[señales].mean()) if señales.any() else None,
            "señales_test": int(señales.sum()),
        }
        modelos[lado], umbrales[lado], proba_test[lado] = modelo, umbral, p_test

    trades = simular(test, proba_test["long"], proba_test["short"], umbrales["long"], umbrales["short"])
    return {"modelos": modelos, "umbrales": umbrales, "lados": lados, "trades": trades}


def fecha(time_mt5) -> str:
    """Fecha de un `time` de MT5, sin convertir de zona: es la hora del servidor."""
    return datetime.fromtimestamp(int(time_mt5), timezone.utc).strftime("%Y-%m-%d")


def walk_forward(datos: pd.DataFrame, semilla: int, mostrar: bool = False) -> list[dict]:
    """Las ventanas del walk-forward con una semilla del Random Forest."""
    ventanas = []
    particiones = ventanas_walk_forward(len(datos), config.WF_VENTANAS, config.MAX_BARS_EN_TRADE)
    for n, (i_train, i_val, i_test) in enumerate(particiones, start=1):
        train, val, test = datos.iloc[i_train], datos.iloc[i_val], datos.iloc[i_test]
        ventana = entrenar_ventana(train, val, test, semilla)
        ventana["resumen"] = {
            "test_desde": fecha(test["time"].iloc[0]),
            "test_hasta": fecha(test["time"].iloc[-1]),
            "filas_train": len(train),
            "umbral_long": ventana["umbrales"]["long"],
            "umbral_short": ventana["umbrales"]["short"],
            **resumen(ventana["trades"]),
        }
        if mostrar:
            print(f"Ventana {n}/{config.WF_VENTANAS}: train={len(train)} validation={len(val)} test={len(test)} "
                  f"({ventana['resumen']['test_desde']} -> {ventana['resumen']['test_hasta']})")
        ventanas.append(ventana)
    return ventanas


def _resumen_walk_forward(ventanas: list[dict]) -> dict:
    return {
        "ventanas": [v["resumen"] for v in ventanas],
        "agregado": resumen(np.concatenate([v["trades"] for v in ventanas])),
    }


def entrenar(velas: pd.DataFrame, point: float, costes: dict | None = None) -> dict:
    """Walk-forward con cada semilla de SEMILLAS_ROBUSTEZ y evaluación en el holdout, sin dependencia de MT5.

    El modelo que se guarda es el de la última ventana con la primera semilla; las demás solo sirven
    para comprobar que el resultado no depende de la suerte del bosque. Ese modelo, entrenado solo con datos
    anteriores al holdout, es el que se evalúa en el holdout.
    """
    datos, holdout = separar_holdout(preparar_datos(velas, point, costes))
    print(f"Walk-forward: {len(datos)} filas hasta {fecha(datos['time'].iloc[-1])}; "
          f"holdout desde {config.HOLDOUT_DESDE}: {len(holdout)} filas")
    principal, robustez = None, []
    for semilla in config.SEMILLAS_ROBUSTEZ:
        ventanas = walk_forward(datos, semilla, mostrar=principal is None)
        robustez.append({"semilla": semilla, "walk_forward": _resumen_walk_forward(ventanas)})
        print(f"Semilla {semilla}: {_texto_backtest(robustez[-1]['walk_forward']['agregado'])}")
        if principal is None:
            principal = ventanas

    ultima = principal[-1]
    trades_holdout = np.array([])
    if len(holdout):
        p = {lado: ultima["modelos"][lado].predict_proba(holdout[FEATURES])[:, 1] for lado in LADOS}
        trades_holdout = simular(holdout, p["long"], p["short"], ultima["umbrales"]["long"], ultima["umbrales"]["short"])
    informe = {
        **ultima["lados"],
        "backtest_test": resumen(ultima["trades"]),
        "walk_forward": robustez[0]["walk_forward"],
        "robustez": robustez,
        "holdout": {"desde": config.HOLDOUT_DESDE, "filas": len(holdout), **resumen(trades_holdout)},
        "costes": {**COSTES_POR_DEFECTO, **(costes or {}),
                   "comision_puntos": config.COMISION_PUNTOS, "slippage_puntos": config.SLIPPAGE_PUNTOS},
    }
    return {
        "model_long": ultima["modelos"]["long"],
        "model_short": ultima["modelos"]["short"],
        "umbral_long": ultima["umbrales"]["long"],
        "umbral_short": ultima["umbrales"]["short"],
        "features": list(FEATURES),
        "sl_mult": config.SL_MULTIPLICADOR_ATR,
        "tp_mult": config.TP_MULTIPLICADOR_ATR,
        "n_barras": config.MAX_BARS_EN_TRADE,
        "symbol": config.SYMBOL,
        "timeframe": config.TIMEFRAME,
        "entrenado": datetime.now().isoformat(timespec="seconds"),
        "informe": informe,
    }


def _texto_backtest(bt: dict) -> str:
    if bt["trades"] == 0:
        return "sin operaciones"
    return (f"{bt['trades']} trades  acierto {bt['acierto']:.1%}  esperanza {bt['esperanza_R']:+.3f} R "
            f"± {bt['error_estandar_R']:.3f} (t = {bt['t_stat']:.2f})")


def imprimir_informe(informe: dict) -> None:
    breakeven = 1 / (1 + config.TP_MULTIPLICADOR_ATR / config.SL_MULTIPLICADOR_ATR)
    print(f"\nPunto de equilibrio de acierto: {breakeven:.1%} (los costes ya van descontados en cada R)")

    print(f"\n[WALK-FORWARD, test fuera de muestra de cada ventana, semilla {informe['robustez'][0]['semilla']}]")
    for v in informe["walk_forward"]["ventanas"]:
        umbrales = "/".join("off" if u is None else f"{u:.2f}" for u in (v["umbral_long"], v["umbral_short"]))
        print(f"  {v['test_desde']} -> {v['test_hasta']}  umbrales L/S {umbrales:<9}  {_texto_backtest(v)}")
    agregado = informe["walk_forward"]["agregado"]
    print(f"  AGREGADO: {_texto_backtest(agregado)}")
    if agregado["trades"] > 0:
        print(f"            profit factor {agregado['profit_factor']:.2f}  max drawdown {agregado['max_drawdown_R']:.1f} R  "
              f"total {agregado['total_R']:+.1f} R")
    print(f"  Ventanas con esperanza > 0: {proporcion_ventanas_positivas(informe):.0%}")

    print("\n[ROBUSTEZ, el mismo walk-forward con cada semilla del Random Forest]")
    for r in informe["robustez"]:
        wf = r["walk_forward"]
        print(f"  semilla {r['semilla']:>3}: {_texto_backtest(wf['agregado'])}  ventanas+ {_proporcion_positivas(wf):.0%}  "
              f"{'pasa' if pasa_walk_forward(wf) else 'no pasa'}")

    print("\n[ÚLTIMA VENTANA, el modelo que se guarda]")
    for lado in LADOS:
        r = informe[lado]
        estado = f"umbral {r['umbral']:.2f}" if r["umbral"] is not None else "DESACTIVADO (sin esperanza positiva en validation)"
        auc = f"{r['auc_test']:.3f}" if r["auc_test"] is not None else "n/a"
        precision = f", precision {r['precision_test']:.1%} en {r['señales_test']} señales" if r["precision_test"] is not None else ""
        print(f"  {lado.upper():<5} {estado}; tasa base train {r['tasa_base_train']:.1%}, ROC-AUC test {auc}{precision}")

    h = informe["holdout"]
    print(f"\n[HOLDOUT desde {h['desde']}, reservado: {h['filas']} filas]")
    print(f"  {_texto_backtest(h)}  {'pasa' if pasa_holdout(h) else 'no pasa'}")

    criterio = (f"t agregado >= {config.T_MINIMO_TEST} con >= {config.MIN_TRADES_TEST} trades y esperanza > 0 "
                f"en >= {config.WF_MIN_PROP_VENTANAS_POSITIVAS:.0%} de las ventanas, con todas las semillas "
                f"{list(config.SEMILLAS_ROBUSTEZ)}; y t >= {config.T_MINIMO_TEST} con >= {config.MIN_TRADES_TEST} "
                "trades en el holdout")
    if ventaja_significativa(informe):
        print(f"\nVEREDICTO: ventaja significativa ({criterio})")
    else:
        print(f"\nVEREDICTO: SIN ventaja significativa ({criterio}); trading_bot.py no arrancará con este modelo")


def main() -> None:
    import MetaTrader5 as mt5
    import mt5_utils

    if not mt5_utils.conectar():
        raise SystemExit(f"No se pudo conectar con MT5: {mt5.last_error()}")
    try:
        info = mt5.symbol_info(config.SYMBOL)
        velas = mt5_utils.descargar_velas(1, config.VELAS_HISTORICO)
    finally:
        mt5.shutdown()
    costes = costes_de_simbolo(info)
    print(f"{len(velas)} velas cerradas de {config.SYMBOL} {config.TIMEFRAME} | costes: comisión "
          f"{config.COMISION_PUNTOS} pts, deslizamiento {config.SLIPPAGE_PUNTOS} pts por ejecución, swap long/short "
          f"{costes['swap_long_puntos']}/{costes['swap_short_puntos']} pts por noche, stops_level {costes['stops_level']}")

    bundle = entrenar(velas, point=info.point, costes=costes)
    imprimir_informe(bundle["informe"])

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    config.MODELOS_HISTORICO_DIR.mkdir(parents=True, exist_ok=True)
    copia = config.MODELOS_HISTORICO_DIR / f"{config.MODEL_PATH.stem}_{datetime.now():%Y%m%d_%H%M%S}.pkl"
    joblib.dump(bundle, copia)
    print(f"\nModelo guardado en {config.MODEL_PATH} (copia en {copia})")


if __name__ == "__main__":
    main()
