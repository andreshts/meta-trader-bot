"""Prueba técnica de una apertura y su cierre, exclusivamente en la demo Exness autorizada.

Sin --execute solo inspecciona. No usa señales del modelo ni habilita cuentas reales.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

import config
from features import build_features
from trading_bot import InstanciaUnica

LOGIN = 480569059
SERVER = "Exness-MT5Trial11"
TERMINAL = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
SYMBOL = "EURUSDm"
MAGIC = 20261106
MAX_RISK_USD = 1.25  # Exclusivamente dinero virtual para la prueba de ejecución.
REPORT = config.BASE_DIR / "logs" / "exness_demo_probe.jsonl"


def audit(event, **data):
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **data}
    print(json.dumps(row, default=str), flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def guard():
    a, terminal = mt5.account_info(), mt5.terminal_info()
    if (a is None or a.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO
            or a.login != LOGIN or a.server != SERVER or a.currency != "USD"):
        raise RuntimeError("Cuenta inesperada: se prohíbe enviar órdenes")
    if terminal is None or not terminal.connected:
        raise RuntimeError("Terminal desconectada")
    return a, terminal


def positions():
    result = mt5.positions_get(symbol=SYMBOL)
    if result is None:
        raise RuntimeError(f"No se pueden consultar posiciones: {mt5.last_error()}")
    return [p for p in result if p.magic == MAGIC]


def orders():
    result = mt5.orders_get(symbol=SYMBOL)
    if result is None:
        raise RuntimeError(f"No se pueden consultar órdenes: {mt5.last_error()}")
    return [o for o in result if o.magic == MAGIC]


def send(request):
    a, terminal = guard()
    if not terminal.trade_allowed or terminal.tradeapi_disabled or not a.trade_allowed or not a.trade_expert:
        raise RuntimeError("Trading algorítmico no habilitado")
    if request.get("symbol") != SYMBOL or request.get("magic") != MAGIC:
        raise RuntimeError("Solicitud ajena a la prueba")
    check = mt5.order_check(request)
    if check is None or check.retcode != 0:
        raise RuntimeError(f"order_check rechazó la solicitud: {check}; {mt5.last_error()}")
    guard()  # Revalidar inmediatamente antes del efecto externo.
    result = mt5.order_send(request)
    audit("order_result", request=request, result=None if result is None else result._asdict())
    return result


def fresh_tick(attempts=1):
    for attempt in range(attempts):
        tick = mt5.symbol_info_tick(SYMBOL)
        # Se observó un desfase de ~3 s entre el reloj local y el servidor.
        if tick is not None and -5 <= time.time() - tick.time <= 15 and 0 < tick.bid <= tick.ask:
            return tick
        if attempt + 1 < attempts:
            time.sleep(0.5)  # La terminal puede devolver su caché justo después de initialize.
    raise RuntimeError("Cotización inválida o desactualizada")


def inspect():
    a, terminal = guard()
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError("No se puede seleccionar el símbolo")
    info, tick = mt5.symbol_info(SYMBOL), fresh_tick(attempts=10)
    if info is None or not info.filling_mode & 1:
        raise RuntimeError("La prueba exige llenado FOK")
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, config.BUFFER_VELAS + 1)
    if rates is None or len(rates) < config.BUFFER_VELAS + 1:
        raise RuntimeError("Histórico insuficiente")
    if not int(rates[-1]["time"]) <= tick.time < int(rates[-1]["time"]) + 900:
        raise RuntimeError("La vela actual no corresponde a la cotización")
    if int(rates[-1]["time"]) - int(rates[-2]["time"]) != 900:
        raise RuntimeError("Última vela cerrada desactualizada")
    atr = float(build_features(pd.DataFrame(rates[:-1]), info.point).atr.iloc[-1])
    sl = round(tick.ask - config.SL_MULTIPLICADOR_ATR * atr, info.digits)
    tp = round(tick.ask + config.TP_MULTIPLICADOR_ATR * atr, info.digits)
    volume = info.volume_min
    risk = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, SYMBOL, volume, tick.ask, sl)
    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, SYMBOL, volume, tick.ask)
    if risk is None or not 0 < -risk <= MAX_RISK_USD or margin is None or margin > a.margin_free:
        raise RuntimeError(f"Riesgo o margen fuera del límite de demo: {risk=}, {margin=}")
    if (tick.ask - tick.bid) / info.point > config.MAX_SPREAD_PUNTOS:
        raise RuntimeError("Spread fuera de límite")
    audit("preflight", login=a.login, server=a.server, demo=True, balance=a.balance,
          algo_trading=terminal.trade_allowed, symbol=SYMBOL, volume=volume,
          risk_usd=-risk, margin_usd=margin, sl=sl, tp=tp)
    return {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": volume,
            "type": mt5.ORDER_TYPE_BUY, "price": tick.ask, "sl": sl, "tp": tp,
            "deviation": 10, "magic": MAGIC, "comment": "demo execution probe",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK}


def close_probe():
    """Reconcilia y liquida solo esta prueba; nunca repite una apertura."""
    for _ in range(10):
        guard()
        active, opened = orders(), positions()
        if not active and not opened:
            return
        for o in active:
            send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket, "symbol": SYMBOL, "magic": MAGIC})
        for p in positions():
            tick = fresh_tick()
            buy = p.type == mt5.POSITION_TYPE_BUY
            send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "position": p.ticket,
                  "volume": p.volume, "type": mt5.ORDER_TYPE_SELL if buy else mt5.ORDER_TYPE_BUY,
                  "price": tick.bid if buy else tick.ask, "deviation": 10, "magic": MAGIC,
                  "comment": "demo probe close", "type_time": mt5.ORDER_TIME_GTC,
                  "type_filling": mt5.ORDER_FILLING_FOK})
        time.sleep(1)
    raise RuntimeError("Quedan posiciones u órdenes de la prueba: requiere atención")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    with InstanciaUnica(config.LOCK_PATH):
        if not mt5.initialize(TERMINAL):
            raise RuntimeError(str(mt5.last_error()))
        try:
            request = inspect()
            if not args.execute:
                return
            guard()
            existing_positions, existing_orders = mt5.positions_get(), mt5.orders_get()
            if existing_positions is None or existing_orders is None or existing_positions or existing_orders:
                raise RuntimeError("Se requiere la cuenta sin posiciones ni órdenes para iniciar la prueba")
            # Registrar antes de enviar. Si no se puede escribir, no se envía.
            audit("intent", request=request)
            try:
                result = send(request)
                if result is None or result.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                    raise RuntimeError("Apertura no confirmada; reconciliar sin repetir")
                time.sleep(3)
            finally:
                close_probe()
            audit("finished", positions_remaining=len(positions()), orders_remaining=len(orders()))
        finally:
            mt5.shutdown()


if __name__ == "__main__":
    main()
