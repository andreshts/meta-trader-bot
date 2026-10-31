"""Fixtures comunes: velas sintéticas y config aislada en un directorio temporal."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

POINT = 0.00001
T0 = 1_700_000_000  # 2023-11-14, hora del servidor


def velas_aleatorias(n: int, semilla: int = 0, spread: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    close = 1.1 + np.cumsum(rng.normal(0, 0.0004, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "time": T0 + 900 * np.arange(n),
        "open": open_,
        "close": close,
        "high": np.maximum(open_, close) + 0.0002,
        "low": np.minimum(open_, close) - 0.0002,
        "tick_volume": rng.integers(200, 600, n).astype(float),
        "spread": spread,
        "real_volume": 0,
    })


def velas_con_senal(n: int = 15_000, semilla: int = 1) -> pd.DataFrame:
    """Picos de volumen seguidos de 12 velas de deriva alcista: una ventaja real que el modelo debe encontrar."""
    rng = np.random.default_rng(semilla)
    ret = rng.normal(0, 0.0004, n)
    vol = rng.integers(200, 600, n).astype(float)
    for s in rng.choice(np.arange(100, n - 20), size=600, replace=False):
        vol[s] = 3000
        ret[s + 1:s + 13] += 0.0004
    close = 1.1 + np.cumsum(ret)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "time": T0 + 900 * np.arange(n),
        "open": open_,
        "close": close,
        "high": np.maximum(open_, close) + np.abs(rng.normal(0, 0.0002, n)),
        "low": np.minimum(open_, close) - np.abs(rng.normal(0, 0.0002, n)),
        "tick_volume": vol,
        "spread": 8,
        "real_volume": 0,
    })


@pytest.fixture(autouse=True)
def config_aislado(tmp_path, monkeypatch):
    """Ningún test escribe en logs/ ni en models/ del proyecto, ni espera de verdad."""
    for nombre, archivo in {
        "LOG_PATH": "trading_bot.log",
        "OPERACIONES_PATH": "operaciones.csv",
        "ESTADO_PATH": "estado_bot.json",
        "LOCK_PATH": "trading_bot.lock",
        "PAUSA_PATH": "PAUSA",
        "MODEL_PATH": "modelo.pkl",
    }.items():
        monkeypatch.setattr(config, nombre, tmp_path / archivo)
    monkeypatch.setattr(config, "MODELOS_HISTORICO_DIR", tmp_path / "historico")
    monkeypatch.setattr(config, "ESPERA_RECONCILIACION_S", 0)
    monkeypatch.setattr(config, "REINTENTO_CONEXION_S", 0)
    return tmp_path
