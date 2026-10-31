"""Conexión con la terminal de MT5 y descarga de velas, compartidas por los dos scripts."""
import time
from datetime import datetime, timedelta, timezone

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


def descargar_cerradas_desde(desde: str, intentos: int = 20, espera_s: float = 3) -> pd.DataFrame:
    """Todas las velas cerradas desde la fecha `desde` (AAAA-MM-DD, hora del servidor), al día.

    Justo después de conectar, la terminal puede devolver solo las velas que tenía en caché (por ejemplo,
    hasta el último día en que estuvo abierta) y completar el resto en segundo plano. Se repite la descarga
    hasta que la última vela cerrada llega hasta el último tick: su cierre es la apertura de la vela en curso.

    Si el histórico empieza más de una semana después de `desde`, la terminal lo está recortando (límite de
    barras) y se detiene con las instrucciones para ampliarlo.
    """
    inicio = datetime.strptime(desde, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    velas, tick = None, None
    for _ in range(intentos):
        rates = mt5.copy_rates_range(config.SYMBOL, TIMEFRAME, inicio, datetime.now(timezone.utc) + timedelta(days=2))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_range falló: {mt5.last_error()}")
        en_formacion = int(descargar_velas(0, 1)["time"].iloc[0])
        velas = pd.DataFrame(rates)
        velas = velas[velas["time"] < en_formacion].reset_index(drop=True)
        tick = mt5.symbol_info_tick(config.SYMBOL)
        if len(velas) and tick is not None and int(velas["time"].iloc[-1]) + 2 * config.TIMEFRAME_SEGUNDOS > tick.time:
            break
        time.sleep(espera_s)
    else:
        ultima = int(velas["time"].iloc[-1]) if velas is not None and len(velas) else None
        raise RuntimeError(f"Las velas de {config.SYMBOL} no se sincronizaron: la última cerrada es {ultima} y el "
                           f"último tick {getattr(tick, 'time', None)}")
    primera = int(velas["time"].iloc[0])
    if primera > inicio.timestamp() + 7 * 86400:
        empieza = datetime.fromtimestamp(primera, timezone.utc).strftime("%Y-%m-%d")
        raise RuntimeError(f"El histórico de {config.SYMBOL} empieza el {empieza}, no el {desde}: la terminal lo "
                           "recorta. Pon 'Máx. barras en la ventana' en Ilimitado (Herramientas → Opciones → "
                           "Gráficos) y reinicia MT5.")
    return velas
