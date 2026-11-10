"""Entrena un modelo por activo con velas de Exness, con el método de train_model.py, y lo guarda en models/exness/.

Uso: python train_exness.py [SIMBOLO ...] [--comision 7]   (sin símbolos, todos)
--comision es la comisión de ida y vuelta por lote, en la moneda de la cuenta: 0 en una cuenta Standard (la de la demo);
en una Raw Spread o Zero, la que cobre Exness por ese instrumento.
Solo lee velas: no envía órdenes y puede ejecutarse con el bot en marcha.
"""
import argparse
import ctypes
import ctypes.wintypes
import json
import math
import time

import joblib
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import config
import train_model as tm
from activos import ACTIVOS, Activo, hora_modelo
from cuenta import CUENTA
from features import FEATURES

ENTRENABLES = list(ACTIVOS)
VELAS = 99_999  # el máximo que entrega la terminal con su límite de 100.000 barras
SEÑALES_POR_SEMANA = 5  # frecuencia, por lado, con la que se fija el umbral inicial
BELOW_NORMAL_PRIORITY = 0x4000
# Misma proporción que tienen en EUR/USD respecto a su spread mediano de 8 puntos.
FILTRO_SPREAD_SOBRE_MEDIANA, DESLIZAMIENTO_SOBRE_MEDIANA = 25 / 8, 1 / 8


def descargar(simbolo: str, intentos: int = 10) -> pd.DataFrame:
    """Velas M15 cerradas. Con un símbolo recién añadido, la terminal tarda unos segundos en entregar el histórico."""
    if not mt5.symbol_select(simbolo, True):
        raise SystemExit(f"No se pudo seleccionar {simbolo}: {mt5.last_error()}")
    velas = None
    for _ in range(intentos):
        velas = mt5.copy_rates_from_pos(simbolo, mt5.TIMEFRAME_M15, 1, VELAS)
        if velas is not None and len(velas) >= VELAS:
            break
        time.sleep(2)
    if velas is None or len(velas) == 0:
        raise SystemExit(f"Sin velas de {simbolo}: {mt5.last_error()}")
    return pd.DataFrame(velas)


def umbral_por_frecuencia(proba: np.ndarray, semanas: float) -> float:
    """Umbral, a dos decimales y redondeado a la baja, que `proba` supera al menos SEÑALES_POR_SEMANA veces por semana."""
    return math.floor(float(np.quantile(proba, max(0.0, 1 - SEÑALES_POR_SEMANA * semanas / len(proba)))) * 100) / 100


def comision_en_puntos(comision_por_lote: float, info) -> float:
    """Pasa una comisión por lote (moneda de la cuenta) a puntos de precio, que es como la descuenta el backtest."""
    valor_del_punto = info.trade_tick_value / info.trade_tick_size * info.point  # de un lote
    return comision_por_lote / valor_del_punto


def entrenar_activo(activo: Activo, velas: pd.DataFrame, info, comision_por_lote: float = 0.0) -> dict:
    """El bundle listo para el bot, con los umbrales iniciales fijados por frecuencia sobre las velas más recientes."""
    velas = velas.assign(time=hora_modelo(velas["time"]))
    mediana = float(velas["spread"].median())
    config.SYMBOL = activo.simbolo
    config.MAX_SPREAD_PUNTOS = mediana * FILTRO_SPREAD_SOBRE_MEDIANA
    config.SLIPPAGE_PUNTOS = mediana * DESLIZAMIENTO_SOBRE_MEDIANA
    config.COMISION_PUNTOS = comision_en_puntos(comision_por_lote, info)
    costes = tm.costes_de_simbolo(info)

    bundle = tm.entrenar(velas, info.point, costes)
    datos, _ = tm.separar_holdout(tm.preparar_datos(velas, info.point, costes))
    recientes = datos.iloc[len(datos) - len(datos) // (config.WF_VENTANAS + 3):]  # el test de la última ventana
    semanas = (int(recientes["time"].iloc[-1]) - int(recientes["time"].iloc[0])) / (7 * 86400)
    for lado in tm.LADOS:
        proba = bundle[f"model_{lado}"].predict_proba(recientes[FEATURES])[:, 1]
        bundle[f"umbral_{lado}"] = umbral_por_frecuencia(proba, semanas)
    return bundle


def guardar(activo: Activo, bundle: dict) -> None:
    activo.modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, activo.modelo)
    resumen = {clave: bundle[clave] for clave in ("umbral_long", "umbral_short", "entrenado")}
    activo.modelo.with_suffix(".json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")  # para la ventana


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simbolos", nargs="*", help=f"entre: {', '.join(ENTRENABLES)}")
    parser.add_argument("--comision", type=float, default=0.0, help="comisión de ida y vuelta por lote, en la moneda de la cuenta")
    args = parser.parse_args()
    simbolos = args.simbolos or ENTRENABLES
    if not 0 <= args.comision < float("inf"):
        parser.error("--comision no puede ser negativa")
    if desconocidos := set(simbolos) - set(ENTRENABLES):
        parser.error(f"símbolos no válidos: {sorted(desconocidos)}")
    ctypes.windll.kernel32.SetPriorityClass(ctypes.wintypes.HANDLE(-1), BELOW_NORMAL_PRIORITY)  # el bot va primero

    if not mt5.initialize(CUENTA.terminal):
        raise SystemExit(f"No se pudo conectar con MT5: {mt5.last_error()}")
    try:
        descargas = {s: (descargar(s), mt5.symbol_info(s)) for s in simbolos}
    finally:
        mt5.shutdown()

    for simbolo, (velas, info) in descargas.items():
        print(f"\n=== {simbolo}: {len(velas)} velas", flush=True)
        activo = ACTIVOS[simbolo]
        bundle = entrenar_activo(activo, velas, info, args.comision)
        guardar(activo, bundle)
        print(f"Guardado {activo.modelo}: umbral compra {bundle['umbral_long']}, venta {bundle['umbral_short']}", flush=True)


if __name__ == "__main__":
    main()
