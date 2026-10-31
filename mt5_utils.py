"""Conexión con la terminal de MT5 y descarga de velas, compartidas por los dos scripts."""
import time

import MetaTrader5 as mt5
import pandas as pd

import config

TIMEFRAME = getattr(mt5, f"TIMEFRAME_{config.TIMEFRAME}")


def conectar() -> bool:
    """Inicializa la terminal y añade el símbolo al Market Watch. Devuelve False si falla."""
    credenciales = {}
    if config.MT5_LOGIN:
        credenciales = {"login": config.MT5_LOGIN, "password": config.MT5_PASSWORD, "server": config.MT5_SERVER}
    ok = mt5.initialize(config.MT5_PATH, **credenciales) if config.MT5_PATH else mt5.initialize(**credenciales)
    return bool(ok) and mt5.symbol_select(config.SYMBOL, True)


def descargar_velas(desde_pos: int, cantidad: int) -> pd.DataFrame:
    """Velas desde la posición `desde_pos` (0 = vela en formación, 1 = última cerrada).

    `time` queda como entero en hora del servidor del broker; no se convierte a UTC.
    """
    rates = mt5.copy_rates_from_pos(config.SYMBOL, TIMEFRAME, desde_pos, cantidad)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos falló: {mt5.last_error()}")
    return pd.DataFrame(rates)


def descargar_cerradas_al_dia(cantidad: int, intentos: int = 20, espera_s: float = 3) -> pd.DataFrame:
    """Las últimas `cantidad` velas cerradas, esperando a que la terminal termine de sincronizarlas.

    Justo después de conectar, la terminal puede devolver solo las velas que tenía en caché (por ejemplo,
    hasta el último día en que estuvo abierta) y completar el resto en segundo plano. Se da por al día cuando
    la última vela cerrada llega hasta el último tick: su cierre es la apertura de la vela en curso.
    """
    for _ in range(intentos):
        velas = descargar_velas(1, cantidad)
        tick = mt5.symbol_info_tick(config.SYMBOL)
        if tick is not None and int(velas["time"].iloc[-1]) + 2 * config.TIMEFRAME_SEGUNDOS > tick.time:
            return velas
        time.sleep(espera_s)
    raise RuntimeError(f"Las velas de {config.SYMBOL} no se sincronizaron: la última cerrada es "
                       f"{int(velas['time'].iloc[-1])} y el último tick {getattr(tick, 'time', None)}")
