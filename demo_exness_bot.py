"""Ejecuta trading_bot.py durante un tiempo limitado, exclusivamente en la cuenta de Exness configurada (cuenta.py).

Sin configuración es la demo de siempre. Una cuenta real exige EXNESS_CUENTA_REAL=1 y, salvo que se desactive a
propósito, un modelo con ventaja significativa.

Uso: python demo_exness_bot.py [--horas 5] [--simbolo XAUUSDm] [--umbral 0.30]   (crear logs/PAUSA bloquea nuevas entradas)
Sin --horas corre hasta que se detenga. vigilante.py relanza los bots que la ventana dejó en marcha si se caen.
La ventana de panel_control.py lanza un proceso de este script por cada activo y arma estos argumentos.
El estado se ve dentro de MT5 con el indicador mql5/RF_Bot_Panel.mq5.
--umbral sustituye, solo en esa ejecución, los umbrales del modelo: da más operaciones, no más ventaja.

Los ajustes solo existen en este proceso: config.py sigue exigiendo ventaja significativa y apuntando a EURUSD. El
modelo no tiene ventaja demostrada (ver README.md), así que esto solo sirve para observar la mecánica en Demo.
"""
import argparse
import ctypes
import math
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np

import alertas
import config
import trading_bot
from activos import ACTIVOS, EURUSD, Activo, en_franja, es_cierre_diario, hora_modelo, segundos_del_dia
from cuenta import CUENTA, LOGS
from features import build_features
from panel_mt5 import PanelMT5

LOGIN, SERVER, TERMINAL = CUENTA.login, CUENTA.server, CUENTA.terminal
DEL_MODELO = object()  # valor por defecto de --umbral-long y --umbral-short: se conserva el del modelo
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

_verificar_cuenta = trading_bot.verificar_cuenta
_cargar_modelo = trading_bot.cargar_modelo
_ciclo = trading_bot.ciclo
_es_salto_de_fin_de_semana = trading_bot.es_salto_de_fin_de_semana
_evaluar_entrada = trading_bot.evaluar_entrada


def verificar_cuenta(login_esperado=None) -> int:
    """Además de lo que comprueba el bot, se detiene si no es exactamente la cuenta configurada: mismo número,
    mismo servidor y mismo tipo (una cuenta real solo vale si se configuró como real, y al revés)."""
    cuenta = mt5.account_info()
    modo = mt5.ACCOUNT_TRADE_MODE_REAL if CUENTA.real else mt5.ACCOUNT_TRADE_MODE_DEMO
    if cuenta is not None and (cuenta.trade_mode != modo or cuenta.login != CUENTA.login
                               or cuenta.server != CUENTA.server):
        raise SystemExit(f"Cuenta inesperada ({cuenta.login}, {cuenta.server}, modo {cuenta.trade_mode}): se esperaba "
                         f"la {CUENTA.tipo} {CUENTA.login} en {CUENTA.server}; el bot se detiene")
    return _verificar_cuenta(login_esperado)


def cargar_modelo(activo: Activo = EURUSD) -> dict:
    """Valida el modelo contra el símbolo con el que se entrenó; se opera con el nombre que le da Exness."""
    config.SYMBOL = activo.simbolo_modelo
    try:
        return _cargar_modelo()
    finally:
        config.SYMBOL = activo.simbolo


def aplicar_umbral(bundle: dict, umbral: float | None) -> None:
    """Sustituye los umbrales solo en memoria; un lado que el entrenamiento desactivó sigue desactivado."""
    if umbral is None:
        return
    originales = bundle["umbral_long"], bundle["umbral_short"]
    for clave in ("umbral_long", "umbral_short"):
        if bundle[clave] is not None:
            bundle[clave] = umbral
    trading_bot.log.warning("Umbral manual %.2f en esta ejecución; los del modelo son long=%s short=%s", umbral, *originales)


def umbral_valido(texto: str) -> float:
    valor = float(texto)
    if not 0 < valor < 1:
        raise argparse.ArgumentTypeError(f"{texto} no está entre 0 y 1")
    return valor


def umbral_o_apagado(texto: str) -> float | None:
    """`off` no opera ese lado."""
    return None if texto == "off" else umbral_valido(texto)


def franja_o_apagada(texto: str) -> tuple | None:
    """`HH:MM-HH:MM` en hora de los modelos; `off` no aplica ninguna franja."""
    if texto == "off":
        return None
    inicio, _, fin = texto.partition("-")
    try:
        segundos_del_dia(inicio), segundos_del_dia(fin)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{texto}: {e}; usa HH:MM-HH:MM u off") from None
    return inicio, fin


def evaluar_fuera_de(franja: tuple):
    """La evaluación de entradas del bot, salvo en las velas que abren dentro de la franja."""
    def evaluar_entrada(bundle: dict, info, bar_time: int, estado: dict) -> None:
        if en_franja(bar_time, franja):
            trading_bot.log.info("Vela dentro de la franja sin entradas (%s a %s, hora de los modelos); no se evalúa", *franja)
            return
        _evaluar_entrada(bundle, info, bar_time, estado)
    return evaluar_entrada


def numero_positivo(texto: str) -> float:
    valor = float(texto)
    if not 0 < valor < math.inf:
        raise argparse.ArgumentTypeError(f"{texto} no es un número positivo")
    return valor


def entero_positivo(texto: str) -> int:
    valor = int(texto)
    if valor < 1:
        raise argparse.ArgumentTypeError(f"{texto} no es un entero positivo")
    return valor


def porcentaje(texto: str) -> float:
    valor = numero_positivo(texto)
    if valor > 100:
        raise argparse.ArgumentTypeError(f"{texto} supera el 100 %")
    return valor / 100


def features_en_hora_del_modelo(velas, point):
    """Solo cambia la hora que ve el modelo; las velas y las órdenes siguen en hora del servidor."""
    return build_features(velas.assign(time=hora_modelo(velas["time"])), point)


def salto_admitido(anterior: int, siguiente: int) -> bool:
    """Para los activos con pausa diaria, esa pausa no cuenta como velas que faltan."""
    return _es_salto_de_fin_de_semana(anterior, siguiente) or es_cierre_diario(anterior, siguiente)


def resultado_del_dia() -> tuple:
    """(realizado, flotante) de todos los activos del bot desde las 00:00 del servidor: el límite diario es de la cuenta."""
    inicio_dia = mt5.symbol_info_tick(config.SYMBOL).time // 86400 * 86400
    ahora = datetime.now(timezone.utc)
    deals, posiciones = mt5.history_deals_get(ahora - timedelta(days=7), ahora + timedelta(days=2)), mt5.positions_get()
    if deals is None or posiciones is None:
        raise RuntimeError(f"No se pudo leer el resultado del día: {mt5.last_error()}")
    magics = {a.magic for a in ACTIVOS.values()}
    realizado = sum(trading_bot.importe_deal(d) for d in deals if d.magic in magics and d.time >= inicio_dia)
    flotante = sum(getattr(p, "profit", 0.0) + getattr(p, "swap", 0.0) for p in posiciones if p.magic in magics)
    return realizado, flotante


def datos_de_la_cuenta() -> dict | None:
    """Balance, equity y moneda para el panel; None si la terminal no los entrega ahora."""
    cuenta = mt5.account_info()
    if cuenta is None:
        return None
    return {"balance": f"{cuenta.balance:.2f}", "equity": f"{cuenta.equity:.2f}", "moneda": cuenta.currency}


def ciclo_con_limite(fin: float, panel: PanelMT5, detener=None):
    """El ciclo del bot con fin de sesión (`fin` infinito = sin límite), orden de detener y latido.

    `ciclo.al_esperar` hace lo mismo mientras el bot espera a que vuelva la conexión, para que siga
    respondiendo a la ventana y al vigilante.
    """
    def parar_si_toca() -> None:
        if time.time() >= fin:
            trading_bot.log.info("Fin de la sesión programada")
            raise SystemExit(0)  # no es una Exception: sale del bucle y cierra la conexión con MT5
        if detener is not None and detener.exists():
            trading_bot.log.info("Detenido desde la ventana de configuración")
            detener.unlink(missing_ok=True)
            raise SystemExit(0)

    def ciclo(bundle: dict, estado: dict) -> None:
        parar_si_toca()
        _ciclo(bundle, estado)
        panel.latido(datos_de_la_cuenta)

    def al_esperar() -> None:
        parar_si_toca()
        panel.latido()

    ciclo.al_esperar = al_esperar
    return ciclo


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horas", type=numero_positivo, help="duración de la sesión; sin él, hasta que se detenga")
    parser.add_argument("--simbolo", choices=ACTIVOS, default=EURUSD.simbolo)
    parser.add_argument("--umbral", type=umbral_valido, help="umbral de long y short solo para esta ejecución, ej. 0.30")
    parser.add_argument("--umbral-long", type=umbral_o_apagado, default=DEL_MODELO, help="umbral de compra, u off")
    parser.add_argument("--umbral-short", type=umbral_o_apagado, default=DEL_MODELO, help="umbral de venta, u off")
    lote = parser.add_mutually_exclusive_group()
    lote.add_argument("--volumen", type=numero_positivo, help="lote fijo de cada operación")
    lote.add_argument("--riesgo", type=porcentaje, help="lote calculado para arriesgar este %% de la equity por operación")
    parser.add_argument("--max-spread", type=numero_positivo, help="spread máximo para entrar, en puntos")
    parser.add_argument("--sin-entradas", type=franja_o_apagada, default=DEL_MODELO,
                        help="franja sin entradas HH:MM-HH:MM, en hora de los modelos (cierre de Nueva York = 00:00), u off")
    parser.add_argument("--perdida-diaria", type=porcentaje, help="pérdida máxima del día, en %% del balance")
    parser.add_argument("--drawdown", type=porcentaje, help="caída máxima desde el máximo de equity, en %%")
    parser.add_argument("--riesgo-abierto", type=porcentaje,
                        help="pérdida máxima, en %% de la equity, si todas las posiciones de la cuenta tocan su stop")
    parser.add_argument("--max-posiciones", type=entero_positivo, help="posiciones abiertas a la vez entre todos los activos")
    return parser.parse_args()


def configurar_activo(activo: Activo, args: argparse.Namespace) -> None:
    config.SYMBOL = activo.simbolo
    config.MODEL_PATH = activo.modelo
    config.MAGIC_NUMBER = activo.magic
    config.VOLUMEN = args.volumen or activo.lote_min
    config.RIESGO_PCT = args.riesgo  # None = lote fijo
    config.MAX_SPREAD_PUNTOS = args.max_spread or activo.max_spread
    # El deslizamiento que se tolera crece con el spread: 20 puntos de EUR/USD serían nada en el oro.
    config.DESVIACION_SLIPPAGE = max(1, round(config.DESVIACION_SLIPPAGE * activo.max_spread / EURUSD.max_spread))
    config.MAX_PERDIDA_DIARIA_PCT = args.perdida_diaria or config.MAX_PERDIDA_DIARIA_PCT
    config.MAX_DRAWDOWN_PCT = args.drawdown or config.MAX_DRAWDOWN_PCT
    config.MT5_PATH = CUENTA.terminal
    config.EXIGIR_VENTAJA_SIGNIFICATIVA = CUENTA.exigir_ventaja
    config.PERMITIR_CUENTA_REAL = CUENTA.real
    # Lo que comparten los bots de todos los activos: el máximo de equity, el turno para abrir y los topes de exposición.
    config.MAGICS_CUENTA = {a.magic for a in ACTIVOS.values()}
    config.ESTADO_CUENTA_PATH = LOGS / "riesgo_cuenta.json"
    config.BLOQUEO_CUENTA_PATH = LOGS / "riesgo_cuenta.lock"
    config.MAX_RIESGO_ABIERTO_PCT = args.riesgo_abierto
    config.MAX_POSICIONES_CUENTA = args.max_posiciones
    # El estado de riesgo es por cuenta y activo: el de otra cuenta falsearía el drawdown y el historial de velas.
    config.LOG_PATH = activo.carpeta / "trading_bot.log"
    config.OPERACIONES_PATH = activo.carpeta / "operaciones.csv"
    config.ESTADO_PATH = activo.carpeta / "estado_bot.json"
    config.LOCK_PATH = activo.bloqueo


def main() -> None:
    args = leer_argumentos()
    activo = ACTIVOS[args.simbolo]
    fin = time.time() + args.horas * 3600 if args.horas else math.inf
    panel = PanelMT5(activo.simbolo, CUENTA.login, activo.magic, fin, activo.panel)

    def cargar_modelo_y_publicar() -> dict:
        bundle = cargar_modelo(activo)
        aplicar_umbral(bundle, args.umbral)
        for clave, valor in (("umbral_long", args.umbral_long), ("umbral_short", args.umbral_short)):
            if valor is not DEL_MODELO:
                bundle[clave] = valor
        panel.fijar_umbrales(bundle["umbral_long"], bundle["umbral_short"])
        return bundle

    configurar_activo(activo, args)
    activo.detener.unlink(missing_ok=True)  # una orden de detener de una sesión anterior no cuenta
    trading_bot.verificar_cuenta = verificar_cuenta
    trading_bot.cargar_modelo = cargar_modelo_y_publicar
    trading_bot.build_features = features_en_hora_del_modelo
    trading_bot.resultado_del_dia = resultado_del_dia
    trading_bot.ciclo = ciclo_con_limite(fin, panel, activo.detener)
    trading_bot.al_esperar = trading_bot.ciclo.al_esperar
    if activo.cierra_a_diario:
        trading_bot.es_salto_de_fin_de_semana = salto_admitido
    franja = activo.sin_entradas if args.sin_entradas is DEL_MODELO else args.sin_entradas
    if franja is not None:
        trading_bot.evaluar_entrada = evaluar_fuera_de(franja)
    trading_bot.log.addHandler(panel)
    avisos = alertas.crear(f"{activo.simbolo} · {CUENTA.tipo} {CUENTA.login}")  # None sin Telegram configurado
    if avisos is not None:
        trading_bot.log.addHandler(avisos)

    # Evita la suspensión por inactividad mientras dure este proceso; no cambia ningún ajuste de Windows.
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        trading_bot.main()
    except SystemExit as e:
        if avisos is not None and e.code not in (None, 0):  # las salidas normales ya dejan su línea en el log
            avisos.avisar(f"🛑 El bot se detuvo: {e.code}")
        raise
    except BaseException as e:
        if avisos is not None:
            avisos.avisar(f"🛑 El bot se cayó: {type(e).__name__}: {e}")
        raise
    finally:
        panel.detener()
        if avisos is not None:
            avisos.cerrar()


if __name__ == "__main__":
    main()
