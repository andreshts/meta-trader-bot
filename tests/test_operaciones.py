"""Reconstrucción de las operaciones de un activo a partir de lo que entrega MT5."""
from types import SimpleNamespace as N

import MetaTrader5 as real
import pytest

import operaciones as op
from activos import ACTIVOS

ORO = ACTIVOS["XAUUSDm"]


def deal(posicion, entry, t, precio, profit=0.0, swap=0.0, comision=0.0, tipo=real.DEAL_TYPE_BUY, magic=ORO.magic, symbol=ORO.simbolo,
         razon=real.DEAL_REASON_EXPERT, comentario="", orden=None, volumen=0.01):
    return N(position_id=posicion, entry=entry, time=t, price=precio, profit=profit, swap=swap, commission=comision, fee=0.0, type=tipo,
             magic=magic, symbol=symbol, reason=razon, comment=comentario, order=orden or posicion, volume=volumen)


def entrada(posicion=1, t=1000, precio=4000.0, **kw):
    return deal(posicion, real.DEAL_ENTRY_IN, t, precio, **kw)


def salida(posicion=1, t=2000, precio=4010.0, **kw):
    return deal(posicion, real.DEAL_ENTRY_OUT, t, precio, **kw)


def orden(ticket=1, sl=3990.0, tp=4020.0):
    return N(ticket=ticket, sl=sl, tp=tp)


def posicion(ticket=7, magic=ORO.magic, symbol=ORO.simbolo, tipo=real.POSITION_TYPE_SELL):
    return N(ticket=ticket, magic=magic, symbol=symbol, type=tipo, volume=0.01, time=3000, price_open=4005.0, sl=4015.0, tp=3985.0,
             price_current=4001.0, profit=4.0, swap=-0.5)


def test_una_operacion_cerrada_une_entrada_salida_y_niveles():
    deals = [entrada(comision=-0.1), salida(profit=20.0, swap=-1.0, comision=-0.1, razon=real.DEAL_REASON_TP)]
    [o] = op.reconstruir(deals, [orden()], [], ORO)
    assert (o.ticket, o.compra, o.lote, o.apertura, o.entrada, o.sl, o.tp) == (1, True, 0.01, 1000, 4000.0, 3990.0, 4020.0)
    assert (o.cierre, o.salida, o.motivo) == (2000, 4010.0, "Take profit")
    assert o.resultado == pytest.approx(18.8)


@pytest.mark.parametrize("razon,comentario,motivo", [
    (real.DEAL_REASON_SL, "", "Stop loss"), (real.DEAL_REASON_TP, "", "Take profit"),
    (real.DEAL_REASON_EXPERT, "rf_mt5 time stop", "Por tiempo"), (real.DEAL_REASON_CLIENT, "", "Manual"),
])
def test_motivo_de_cierre(razon, comentario, motivo):
    [o] = op.reconstruir([entrada(), salida(razon=razon, comentario=comentario)], [], [], ORO)
    assert o.motivo == motivo


def test_una_venta_se_identifica_como_venta():
    [o] = op.reconstruir([entrada(tipo=real.DEAL_TYPE_SELL), salida()], [], [], ORO)
    assert o.compra is False


def test_una_posicion_abierta_lleva_flotante_y_precio_actual():
    [o] = op.reconstruir([entrada(posicion=7, comision=-0.2)], [], [posicion()], ORO)
    assert (o.cierre, o.salida, o.motivo, o.compra, o.sl) == (None, 4001.0, "Abierta", False, 4015.0)
    assert o.resultado == pytest.approx(3.3)


def test_la_salida_cuenta_aunque_el_broker_le_ponga_otro_magic():
    [o] = op.reconstruir([entrada(), salida(magic=0, profit=5.0)], [], [], ORO)
    assert o.resultado == 5.0


def test_lo_ajeno_no_aparece():
    deals = [entrada(posicion=1, magic=999), salida(posicion=1, magic=999), entrada(posicion=2, symbol="EURUSDm"), salida(posicion=2)]
    posiciones = [posicion(ticket=8, magic=999), posicion(ticket=9, symbol="EURUSDm")]
    assert op.reconstruir(deals, [], posiciones, ORO) == []


def test_una_entrada_sin_salida_ni_posicion_espera_a_la_siguiente_lectura():
    assert op.reconstruir([entrada()], [], [], ORO) == []


def test_la_mas_reciente_va_primero():
    deals = [entrada(1, t=100), salida(1, t=200), entrada(2, t=300), salida(2, t=400)]
    assert [o.ticket for o in op.reconstruir(deals, [], [posicion(ticket=7)], ORO)] == [7, 2, 1]


def falso_mt5(conectada=True, inicializa=True, login=op.LOGIN, deals=()):
    return N(terminal_info=lambda: N() if conectada else None, initialize=lambda *_a, **_k: inicializa, last_error=lambda: (1, "x"),
             account_info=lambda: N(login=login, currency="USD"), history_deals_get=lambda *_a: deals,
             history_orders_get=lambda *_a: (), positions_get=lambda: (), symbol_info=lambda _s: N(digits=3),
             DEAL_ENTRY_IN=real.DEAL_ENTRY_IN, POSITION_TYPE_BUY=real.POSITION_TYPE_BUY, DEAL_TYPE_BUY=real.DEAL_TYPE_BUY)


def test_leer_activos_devuelve_una_instantanea_por_simbolo(monkeypatch):
    monkeypatch.setattr(op, "mt5", falso_mt5(deals=(entrada(), salida(profit=2.0))))
    datos = op.leer_activos([ORO])["XAUUSDm"]
    assert (len(datos.operaciones), datos.digitos, datos.moneda) == (1, 3, "USD")


@pytest.mark.parametrize("falso,mensaje", [
    (falso_mt5(conectada=False, inicializa=False), "conectar"), (falso_mt5(login=123), "cuenta configurada"),
])
def test_leer_activos_explica_por_que_no_pudo(monkeypatch, falso, mensaje):
    monkeypatch.setattr(op, "mt5", falso)
    assert mensaje in op.leer_activos([ORO])["XAUUSDm"]
