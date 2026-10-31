"""Cálculo único de features, compartido por train_model.py y trading_bot.py.

La fila `t` solo usa información disponible al cierre de la vela `t` (sin `shift(1)` extra).
"""
import numpy as np
import pandas as pd

FEATURES = [
    "returns",
    "dist_sma10",
    "dist_sma50",
    "volatilidad",
    "atr_rel",
    "rsi",
    "volume_ratio",
    "spread_ratio",
    "hora_sin",
    "hora_cos",
    "dia_semana",
]

PERIODO_ATR = 14
PERIODO_RSI = 14


def atr_wilder(df: pd.DataFrame, periodo: int = PERIODO_ATR) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / periodo, adjust=False).mean()


def rsi_wilder(close: pd.Series, periodo: int = PERIODO_RSI) -> pd.Series:
    """RSI entre 0 y 1: media de subidas / (media de subidas + media de bajadas)."""
    delta = close.diff()
    subidas = delta.clip(lower=0).ewm(alpha=1 / periodo, adjust=False).mean()
    bajadas = (-delta.clip(upper=0)).ewm(alpha=1 / periodo, adjust=False).mean()
    return subidas / (subidas + bajadas)


def build_features(df: pd.DataFrame, point: float) -> pd.DataFrame:
    """Devuelve una copia de `df` (velas de MT5) con las columnas de FEATURES y `atr` (absoluto).

    `atr` no es una feature: lo usan el etiquetado y el cálculo de SL/TP.
    Las primeras filas quedan con NaN o con valores de calentamiento; quien llama descarta
    las primeras `config.WARMUP` filas.
    """
    out = df.copy()
    close = out["close"]
    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()

    out["returns"] = close.pct_change()
    out["dist_sma10"] = (close - sma10) / sma10
    out["dist_sma50"] = (close - sma50) / sma50
    out["volatilidad"] = out["returns"].rolling(10).std()
    out["atr"] = atr_wilder(out)
    out["atr_rel"] = out["atr"] / close
    out["rsi"] = rsi_wilder(close)
    out["volume_ratio"] = out["tick_volume"] / out["tick_volume"].rolling(20).mean()
    out["spread_ratio"] = (out["spread"] * point) / close

    # `time` está en hora del servidor del broker: las sesiones se aprenden en esa misma hora.
    hora = (out["time"] // 3600) % 24
    out["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    out["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    out["dia_semana"] = ((out["time"] // 86400) + 3) % 7  # 1970-01-01 fue jueves -> lunes = 0
    return out
