"""Las operaciones de cada activo, reconstruidas desde el historial de MT5. Solo lee: nunca envía órdenes."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

from activos import Activo
from cuenta import CUENTA
from trading_bot import CIERRE_POR_LIMITE, importe_deal

LOGIN, TERMINAL = CUENTA.login, CUENTA.terminal

DESDE = datetime(2020, 1, 1, tzinfo=timezone.utc)  # anterior a cualquier operación de la cuenta
TIMEOUT_CONEXION_MS = 5000
ABIERTA = "Abierta"
MOTIVOS = {mt5.DEAL_REASON_SL: "Stop loss", mt5.DEAL_REASON_TP: "Take profit", mt5.DEAL_REASON_SO: "Stop out",
           mt5.DEAL_REASON_CLIENT: "Manual", mt5.DEAL_REASON_MOBILE: "Manual", mt5.DEAL_REASON_WEB: "Manual"}


@dataclass(frozen=True)
class Operacion:
    ticket: int
    compra: bool
    lote: float
    apertura: int
    entrada: float
    sl: float
    tp: float
    cierre: int | None  # None mientras siga abierta
    salida: float  # precio de cierre o, si sigue abierta, el actual
    motivo: str
    resultado: float  # neto de swap y comisiones; incluye el flotante si sigue abierta


@dataclass(frozen=True)
class Instantanea:
    operaciones: list  # la más reciente primero
    digitos: int
    moneda: str


def _motivo(salida) -> str:
    if "time stop" in salida.comment:
        return "Por tiempo"
    if CIERRE_POR_LIMITE in salida.comment:
        return "Por límite"
    return MOTIVOS.get(salida.reason, "Bot")


def reconstruir(deals, ordenes, posiciones, activo: Activo) -> list[Operacion]:
    """Une entrada, salidas y niveles de cada posición del activo.

    Las salidas se buscan por posición y no por magic: lo que importa es que la posición la abriera este bot.
    """
    entradas = {d.position_id: d for d in deals
                if d.entry == mt5.DEAL_ENTRY_IN and d.magic == activo.magic and d.symbol == activo.simbolo}
    salidas = defaultdict(list)
    for d in deals:
        if d.entry != mt5.DEAL_ENTRY_IN and d.position_id in entradas:
            salidas[d.position_id].append(d)
    niveles = {o.ticket: o for o in ordenes}

    def realizado(ticket: int) -> float:
        entrada = importe_deal(entradas[ticket]) if ticket in entradas else 0.0
        return entrada + sum(importe_deal(d) for d in salidas[ticket])

    abiertas = {p.ticket: p for p in posiciones if p.magic == activo.magic and p.symbol == activo.simbolo}
    lista = [Operacion(p.ticket, p.type == mt5.POSITION_TYPE_BUY, p.volume, p.time, p.price_open, p.sl, p.tp, None,
                       p.price_current, ABIERTA, realizado(p.ticket) + p.profit + p.swap) for p in abiertas.values()]
    for ticket, e in entradas.items():
        if ticket in abiertas or not salidas[ticket]:  # sin salida ni posición: se acaba de cerrar y llega en la próxima lectura
            continue
        ultima = max(salidas[ticket], key=lambda d: d.time)
        orden = niveles.get(e.order)
        lista.append(Operacion(ticket, e.type == mt5.DEAL_TYPE_BUY, e.volume, e.time, e.price, getattr(orden, "sl", 0.0),
                               getattr(orden, "tp", 0.0), ultima.time, ultima.price, _motivo(ultima), realizado(ticket)))
    return sorted(lista, key=lambda o: o.apertura, reverse=True)


def leer_activos(activos: list[Activo]) -> dict:
    """{símbolo: Instantanea}, o {símbolo: motivo} si no se pudo leer.

    Llamar siempre desde un mismo hilo: la librería de MT5 mantiene una única conexión por proceso.
    """
    try:
        if mt5.terminal_info() is None and not mt5.initialize(TERMINAL, timeout=TIMEOUT_CONEXION_MS):
            raise RuntimeError(f"no se pudo conectar con la terminal {mt5.last_error()}")
        cuenta = mt5.account_info()
        if cuenta is None or cuenta.login != LOGIN:
            raise RuntimeError(f"la terminal no está en la cuenta configurada ({LOGIN})")
        hasta = datetime.now(timezone.utc) + timedelta(days=2)
        deals, ordenes, posiciones = mt5.history_deals_get(DESDE, hasta), mt5.history_orders_get(DESDE, hasta), mt5.positions_get()
        if deals is None or ordenes is None or posiciones is None:
            raise RuntimeError(f"no se pudo leer el historial {mt5.last_error()}")
        return {a.simbolo: Instantanea(reconstruir(deals, ordenes, posiciones, a),
                                       getattr(mt5.symbol_info(a.simbolo), "digits", 5), cuenta.currency) for a in activos}
    except RuntimeError as e:
        return {a.simbolo: str(e) for a in activos}
