"""Ejecuta trading_bot.py durante un tiempo limitado, exclusivamente en la demo Exness autorizada.

Uso: python demo_exness_bot.py --horas 5   (crear logs/PAUSA bloquea nuevas entradas)

Los ajustes solo existen en este proceso: config.py sigue exigiendo ventaja significativa y apuntando a EURUSD. El
modelo no tiene ventaja demostrada (ver README.md), así que esto solo sirve para observar la mecánica en Demo.
"""
import argparse
import ctypes
import time

import MetaTrader5 as mt5

import config
import trading_bot
from demo_exness_probe import LOGIN, SERVER, SYMBOL, TERMINAL
from features import build_features

SIMBOLO_MODELO = config.SYMBOL
# El modelo aprendió las sesiones en hora de MetaQuotes-Demo (UTC+2 en horario de invierno de EE. UU.); Exness usa UTC.
DESFASE_MODELO_S = 2 * 3600
# El estado de riesgo es por cuenta: el de otra cuenta falsearía el drawdown y el historial de velas.
LOGS_DIR = config.BASE_DIR / "logs" / "exness"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

_verificar_cuenta = trading_bot.verificar_cuenta
_cargar_modelo = trading_bot.cargar_modelo
_ciclo = trading_bot.ciclo


def verificar_cuenta(login_esperado=None) -> int:
    """Además de lo que comprueba el bot, se detiene si no es exactamente la cuenta demo autorizada."""
    cuenta = mt5.account_info()
    if cuenta is not None and (cuenta.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO
                               or cuenta.login != LOGIN or cuenta.server != SERVER):
        raise SystemExit(f"Cuenta inesperada ({cuenta.login}, {cuenta.server}); el bot se detiene")
    return _verificar_cuenta(login_esperado)


def cargar_modelo() -> dict:
    """Valida el modelo contra el símbolo con el que se entrenó; se opera con el nombre que le da Exness."""
    config.SYMBOL = SIMBOLO_MODELO
    try:
        return _cargar_modelo()
    finally:
        config.SYMBOL = SYMBOL


def features_en_hora_del_modelo(velas, point):
    """Solo cambia la hora que ve el modelo; las velas y las órdenes siguen en hora del servidor."""
    return build_features(velas.assign(time=velas["time"] + DESFASE_MODELO_S), point)


def ciclo_con_limite(fin: float):
    def ciclo(bundle: dict, estado: dict) -> None:
        if time.time() >= fin:
            trading_bot.log.info("Fin de la sesión programada")
            raise SystemExit(0)  # no es una Exception: sale del bucle y cierra la conexión con MT5
        _ciclo(bundle, estado)
    return ciclo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horas", type=float, required=True)
    args = parser.parse_args()

    config.SYMBOL = SYMBOL
    config.MT5_PATH = TERMINAL
    config.EXIGIR_VENTAJA_SIGNIFICATIVA = False
    config.PERMITIR_CUENTA_REAL = False
    config.LOG_PATH = LOGS_DIR / "trading_bot.log"
    config.OPERACIONES_PATH = LOGS_DIR / "operaciones.csv"
    config.ESTADO_PATH = LOGS_DIR / "estado_bot.json"
    trading_bot.verificar_cuenta = verificar_cuenta
    trading_bot.cargar_modelo = cargar_modelo
    trading_bot.build_features = features_en_hora_del_modelo
    trading_bot.ciclo = ciclo_con_limite(time.time() + args.horas * 3600)

    # Evita la suspensión por inactividad mientras dure este proceso; no cambia ningún ajuste de Windows.
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    trading_bot.main()


if __name__ == "__main__":
    main()
