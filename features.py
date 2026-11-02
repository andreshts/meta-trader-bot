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
PERIODO_ADX = 14
PERIODO_SMA_MTF = 20
PASO_BASE_S = 900  # M15 en segundos
SEGUNDOS_H1 = 3600
SEGUNDOS_H4 = 14400


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


def adx_wilder(df: pd.DataFrame, periodo: int = PERIODO_ADX,
               atr: pd.Series | None = None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX(14), +DI y -DI de Wilder acotados entre 0 y 1."""
    if atr is None:
        atr = atr_wilder(df, periodo)
    subida = df["high"].diff()
    bajada = -df["low"].diff()
    plus_dm = pd.Series(np.where((subida > bajada) & (subida > 0), subida, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((bajada > subida) & (bajada > 0), bajada, 0.0), index=df.index)
    plus_di = plus_dm.ewm(alpha=1 / periodo, adjust=False).mean() / atr
    minus_di = minus_dm.ewm(alpha=1 / periodo, adjust=False).mean() / atr
    suma_di = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / suma_di).fillna(0.0)
    adx = dx.ewm(alpha=1 / periodo, adjust=False).mean()
    return adx, plus_di, minus_di


def mtf_cerrado(df: pd.DataFrame, periodo_s: int, paso_base_s: int = PASO_BASE_S,
                periodo_rsi: int = PERIODO_RSI,
                periodo_sma: int = PERIODO_SMA_MTF) -> tuple[pd.Series, pd.Series]:
    """RSI y distancia a la SMA de temporalidad superior (`periodo_s`) usando solo velas superiores YA cerradas.

    Para la vela base `t` (que cierra en `time[t] + paso_base_s`), un bloque superior `b = time // periodo_s`
    solo está cerrado si `(b + 1) * periodo_s <= time[t] + paso_base_s`. Así ninguna vela en formación ni
    futura interviene jamás en la fila `t`.
    """
    tiempos = df["time"].to_numpy(dtype=np.int64)
    bloques = tiempos // periodo_s
    cierre_htf = df.groupby(bloques, sort=True)["close"].last()
    rsi_htf = rsi_wilder(cierre_htf, periodo_rsi).to_numpy(dtype=float)
    sma_htf = cierre_htf.rolling(periodo_sma).mean()
    dist_sma_htf = ((cierre_htf - sma_htf) / sma_htf).to_numpy(dtype=float)

    fin_bloque = (cierre_htf.index.to_numpy(dtype=np.int64) + 1) * periodo_s
    cierre_base = tiempos + paso_base_s
    idx = np.searchsorted(fin_bloque, cierre_base, side="right") - 1
    valido = idx >= 0
    idx_seguro = np.maximum(idx, 0)

    rsi_alineado = np.where(valido, rsi_htf[idx_seguro], np.nan)
    dist_alineada = np.where(valido, dist_sma_htf[idx_seguro], np.nan)
    return pd.Series(rsi_alineado, index=df.index), pd.Series(dist_alineada, index=df.index)


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
