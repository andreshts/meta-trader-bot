"""Activos que el bot opera en la cuenta de Exness (cuenta.py): lo que cambia de uno a otro y la hora común de sus modelos."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
import cuenta
import panel_mt5

LOGS = cuenta.LOGS  # por cuenta: el estado de riesgo de una no vale para otra
MODELOS = config.BASE_DIR / "models" / "exness"
CIERRE_DIARIO_MAX_S = 3 * 3600  # entre la última vela antes del cierre y la primera después


@dataclass(frozen=True)
class Activo:
    simbolo: str
    nombre: str
    magic: int
    lote_min: float
    max_spread: int  # en puntos
    simbolo_modelo: str
    modelo: Path
    carpeta: Path  # registro, operaciones y estado de riesgo
    bloqueo: Path  # impide dos bots sobre el mismo activo
    panel: Path  # lo que leen el indicador de MT5 y la ventana de configuración
    cierra_a_diario: bool = True
    umbrales: tuple | None = None  # (compra, venta) iniciales; si es None están en el .json junto al modelo
    # Sin entradas en las velas que abren en esta franja, en hora de los modelos (el cierre de Nueva York es a las 00:00):
    # desde media hora antes del cierre hasta pasada la reapertura. En las velas de Exness, el spread del EUR/USD es de 2,5
    # a 6 veces el normal entre las 00:00 y la 01:00, y el oro y los índices no cotizan en esa hora. None = sin franja.
    sin_entradas: tuple | None = ("23:30", "01:15")

    @property
    def detener(self) -> Path:
        return self.carpeta / "DETENER"

    @property
    def salida(self) -> Path:
        return self.carpeta / "salida.txt"


def _activo(simbolo: str, nombre: str, magic: int, lote_min: float, max_spread: int) -> Activo:
    carpeta = LOGS / simbolo
    return Activo(simbolo, nombre, magic, lote_min, max_spread, simbolo, MODELOS / f"{simbolo}.pkl", carpeta,
                  carpeta / "trading_bot.lock", panel_mt5.RUTA.with_name(f"rf_bot_panel_{simbolo}.txt"))


# El EUR/USD conserva sus rutas de siempre para el estado de riesgo y, en la demo original, el bloqueo que comparte con
# demo_exness_probe.py. Su modelo es, como los demás, el entrenado con velas de Exness (train_exness.py): el de
# config.MODEL_PATH se entrenó con otro servidor, cuyo volumen de ticks y spread no son comparables.
EURUSD = Activo("EURUSDm", "EUR/USD", config.MAGIC_NUMBER, 0.01, config.MAX_SPREAD_PUNTOS, "EURUSDm",
                MODELOS / "EURUSDm.pkl", LOGS, config.LOCK_PATH if cuenta.CUENTA.login == cuenta.DEMO.login else LOGS / "trading_bot.lock",
                panel_mt5.RUTA, cierra_a_diario=False)
ACTIVOS = {a.simbolo: a for a in (
    EURUSD,
    _activo("XAUUSDm", "Oro", 202602, 0.01, 750),
    _activo("US500m", "S&P 500", 202603, 0.14, 200),
    _activo("USTECm", "Nasdaq 100", 202604, 0.05, 600),
    _activo("DE30m", "DAX", 202605, 0.07, 60),
)}


# Activos que suelen moverse juntos: una posición del mismo lado en dos de ellos es casi la misma apuesta.
GRUPOS_CORRELACION = (("US500m", "USTECm", "DE30m"),)


def grupo_de(simbolo: str) -> tuple:
    """Los símbolos del grupo de correlación de ese activo, él incluido; él solo si no está en ninguno."""
    return next((grupo for grupo in GRUPOS_CORRELACION if simbolo in grupo), (simbolo,))


def hora_modelo(tiempos) -> np.ndarray:
    """Pasa la hora de Exness (UTC) a la de los modelos: UTC+2 o UTC+3 según el horario de EE. UU.

    Es la del servidor con el que se entrenó el EUR/USD. Con ella, el cierre de Nueva York cae siempre a las 00:00 y las
    sesiones no se mueven una hora al cambiar el horario.
    """
    utc = pd.to_datetime(np.asarray(tiempos, dtype=np.int64), unit="s", utc=True)
    nueva_york = utc.tz_convert("America/New_York").tz_localize(None)
    return ((nueva_york - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64) + 7 * 3600


def segundos_del_dia(hora: str) -> int:
    """"HH:MM" en segundos desde las 00:00. ValueError si no es una hora."""
    horas, _, minutos = hora.partition(":")
    if not (len(horas) == len(minutos) == 2 and horas.isdigit() and minutos.isdigit() and int(horas) < 24 and int(minutos) < 60):
        raise ValueError(f"{hora!r} no es una hora HH:MM")
    return int(horas) * 3600 + int(minutos) * 60


def en_franja(tiempo_servidor: int, franja: tuple) -> bool:
    """True si esa hora del servidor cae en la franja (inicio, fin), dada en hora de los modelos. El inicio entra y el
    fin no; una franja puede cruzar la medianoche."""
    inicio, fin = (segundos_del_dia(h) for h in franja)
    t = int(hora_modelo([tiempo_servidor])[0]) % 86400
    return inicio <= t < fin if inicio <= fin else t >= inicio or t < fin


def es_cierre_diario(anterior: int, siguiente: int) -> bool:
    """True si el salto entre dos velas es la pausa diaria: corto y cruza las 00:00 de la hora de los modelos."""
    if not 0 < siguiente - anterior <= CIERRE_DIARIO_MAX_S:
        return False
    desde, hasta = hora_modelo([anterior, siguiente])
    return bool(desde // 86400 != hasta // 86400)
