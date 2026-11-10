"""Informe de rendimiento de las operaciones cerradas: esperanza en R, profit factor, drawdown y desgloses por activo,
lado, salida y sesión, con la esperanza del backtest de cada activo al lado.

Uso: python informe.py [--dias 7 | --todo] [--telegram]
Lee el historial de MT5 (no envía órdenes) y, para la referencia del backtest, experimentos/registro.csv. El
vigilante lo lanza con --telegram al cierre de cada viernes.
"""
import argparse
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import MetaTrader5 as mt5

import alertas
import experimentos
from activos import ACTIVOS
from operaciones import leer_activos

POCAS = 30  # con menos trades, la comparación con el backtest todavía no dice nada
SESIONES = (("Asia", 0, 7), ("Europa", 7, 13), ("América", 13, 21), ("Cierre", 21, 24))  # hora UTC de la entrada


@dataclass(frozen=True)
class Trade:
    simbolo: str
    compra: bool
    apertura: int  # hora del servidor de Exness, que es UTC
    cierre: int
    motivo: str
    resultado: float  # neto, en la moneda de la cuenta
    r: float | None  # resultado / lo que se arriesgaba hasta el stop; None si no se pudo calcular


def sesion_de(apertura: int) -> str:
    hora = apertura // 3600 % 24
    return next(f"{nombre} ({desde:02d}-{hasta:02d})" for nombre, desde, hasta in SESIONES if desde <= hora < hasta)


def estadisticas(trades: list) -> dict:
    """Resumen de una lista de trades, en orden de cierre. La esperanza, el t y el profit factor van en R cuando
    todos los trades la tienen; el drawdown, en dinero."""
    trades = sorted(trades, key=lambda t: t.cierre)
    resultados = [t.resultado for t in trades]
    rs = [t.r for t in trades if t.r is not None]
    curva = pico = caida = 0.0
    for resultado in resultados:
        curva += resultado
        pico = max(pico, curva)
        caida = max(caida, pico - curva)
    ganado, perdido = sum(x for x in resultados if x > 0), -sum(x for x in resultados if x < 0)
    resumen = {"trades": len(trades), "acierto": sum(x > 0 for x in resultados) / len(trades) if trades else 0.0,
               "neto": sum(resultados), "drawdown": caida,
               "profit_factor": ganado / perdido if perdido > 0 else math.inf if ganado > 0 else 0.0,
               "esperanza_R": statistics.fmean(rs) if rs else None, "t": None}
    if len(rs) > 1 and (desviacion := statistics.stdev(rs)) > 0:
        resumen["t"] = resumen["esperanza_R"] / (desviacion / math.sqrt(len(rs)))
    return resumen


def desglose(trades: list, clave) -> dict:
    """{valor de la clave: estadísticas}, en el orden en que aparece cada valor."""
    grupos = {}
    for trade in trades:
        grupos.setdefault(clave(trade), []).append(trade)
    return {valor: estadisticas(grupo) for valor, grupo in grupos.items()}


def referencia(filas_registro: list) -> dict:
    """{símbolo: esperanza en R del backtest}: la media de las semillas de la variante base en el banco de experimentos."""
    por_simbolo = {}
    for fila in filas_registro:
        if fila["variante"] == experimentos.BASE:
            por_simbolo.setdefault(fila["simbolo"], {})[fila["semilla"]] = float(fila["esperanza_R"])
    return {simbolo: statistics.fmean(semillas.values()) for simbolo, semillas in por_simbolo.items()}


def _r(valor: float | None) -> str:
    return "s/d" if valor is None else f"{valor:+.2f} R"


def _pf(valor: float) -> str:
    return "∞" if math.isinf(valor) else f"{valor:.2f}"


def _trades(n: int) -> str:
    return "1 trade" if n == 1 else f"{n} trades"


def _linea(nombre: str, e: dict, moneda: str, extra: str = "") -> str:
    return (f"{nombre}: {_trades(e['trades'])} · acierto {e['acierto']:.0%} · {e['neto']:+.2f} {moneda} · {_r(e['esperanza_R'])}{extra}"
            f" · PF {_pf(e['profit_factor'])}")


def texto(trades: list, desde: int | None, hasta: int, moneda: str, backtest: dict, nombres: dict) -> str:
    dia = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%d/%m/%Y")  # noqa: E731
    titulo = f"📈 Informe de rendimiento {'hasta el ' + dia(hasta) if desde is None else f'del {dia(desde)} al {dia(hasta)}'}"
    if not trades:
        return f"{titulo}\nSin operaciones cerradas en el periodo"
    total = estadisticas(trades)
    lineas = [titulo, _linea("Total", total, moneda) + f" · drawdown {total['drawdown']:.2f} {moneda}"
              + (f" · t = {total['t']:+.2f}" if total["t"] is not None else ""), "", "Por activo (esperanza en vivo / backtest):"]
    for simbolo, e in desglose(trades, lambda t: t.simbolo).items():
        lineas.append("  " + _linea(nombres.get(simbolo, simbolo), e, moneda, f" / {_r(backtest.get(simbolo))}"))
    for encabezado, clave in (("Por lado:", lambda t: "Compra" if t.compra else "Venta"), ("Por salida:", lambda t: t.motivo),
                              ("Por sesión (hora UTC de la entrada):", lambda t: sesion_de(t.apertura))):
        lineas += ["", encabezado, *("  " + _linea(str(valor), e, moneda) for valor, e in desglose(trades, clave).items())]
    if total["trades"] < POCAS:
        lineas += ["", f"Con {_trades(total['trades'])} el resultado es orientativo; a partir de {POCAS} se puede comparar con el backtest."]
    return "\n".join(lineas)


def riesgo_de(simbolo: str, op) -> float | None:
    """Lo que la operación perdía si tocaba su stop, en la moneda de la cuenta; None si no se puede calcular."""
    if not op.sl:
        return None
    perdida = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY if op.compra else mt5.ORDER_TYPE_SELL, simbolo, op.lote, op.entrada, op.sl)
    return -perdida if perdida is not None and perdida < 0 else None


def leer_trades(activos: dict, desde: int | None, riesgo_de=riesgo_de, leer=leer_activos) -> tuple:
    """(trades cerrados desde `desde`, moneda de la cuenta). RuntimeError si no se pudo leer MT5."""
    trades, moneda = [], ""
    for simbolo, datos in leer(list(activos.values())).items():
        if isinstance(datos, str):
            raise RuntimeError(datos)
        moneda = datos.moneda
        for op in datos.operaciones:
            if op.cierre is None or (desde is not None and op.cierre < desde):
                continue
            riesgo = riesgo_de(simbolo, op)
            trades.append(Trade(simbolo, op.compra, op.apertura, op.cierre, op.motivo, op.resultado, op.resultado / riesgo if riesgo else None))
    return trades, moneda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    periodo = parser.add_mutually_exclusive_group()
    periodo.add_argument("--dias", type=int, default=7, help="días hacia atrás (por defecto 7)")
    periodo.add_argument("--todo", action="store_true", help="todas las operaciones de la cuenta")
    parser.add_argument("--telegram", action="store_true", help="además de mostrarlo, lo envía por Telegram")
    args = parser.parse_args()
    ahora = int(time.time())
    desde = None if args.todo else ahora - args.dias * 86400
    try:
        trades, moneda = leer_trades(ACTIVOS, desde)
    except RuntimeError as e:
        raise SystemExit(f"No se pudo leer MT5: {e}") from None
    finally:
        mt5.shutdown()
    informe = texto(trades, desde, ahora, moneda, referencia(experimentos.leer_registro()), {s: a.nombre for s, a in ACTIVOS.items()})
    print(informe)
    if args.telegram:
        token, chat = alertas.credenciales()
        if token is None:
            raise SystemExit("Telegram no está configurado (.env)")
        try:
            alertas.enviar_telegram(informe, token, chat)
        except alertas.ErrorTelegram as e:
            raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
