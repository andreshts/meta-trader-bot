"""MT5 simulado para probar trading_bot.py sin terminal: cuenta, posiciones, órdenes, historial y velas en memoria."""
from types import SimpleNamespace

import MetaTrader5 as _mt5
import numpy as np
import pandas as pd

import config

SIN_RESPUESTA = "sin respuesta"  # en `respuestas`: order_send devuelve None
INCIERTOS = (_mt5.TRADE_RETCODE_PLACED, _mt5.TRADE_RETCODE_TIMEOUT, _mt5.TRADE_RETCODE_CONNECTION)


class FakeMT5:
    def __init__(self):
        self.conectado = True
        self.login_tras_reconectar = None
        self.cuenta = SimpleNamespace(login=1001, trade_mode=0, equity=10_000.0, balance=10_000.0)
        self.info = SimpleNamespace(point=0.00001, digits=5, filling_mode=1, trade_stops_level=0, volume_min=0.01,
                                    volume_step=0.01, volume_max=100.0, trade_tick_size=0.00001, trade_tick_value=1.0)
        self.bar_time = 1_793_318_400  # apertura de la vela en formación (hora del servidor)
        self.tick = SimpleNamespace(time=self.bar_time + 5, bid=1.10000, ask=1.10002)
        self.posiciones, self.deals, self.enviadas = [], [], []
        self.ordenes, self.historial_ordenes = [], []   # órdenes activas y terminadas
        self.respuestas = []      # retcodes forzados para los próximos order_send
        # Qué pasa con una orden de resultado incierto: "ejecuta" (enseguida), "coloca" (queda activa en el
        # servidor hasta ejecutar_colocadas() o cancelar_colocadas()) o "nada" (nunca llegó al servidor).
        self.modo_incierto = "nada"
        self.fraccion_parcial = 0.5
        self._ticket = 5000

    def __getattr__(self, nombre):
        if nombre.isupper():  # constantes reales: ORDER_TYPE_BUY, TRADE_RETCODE_DONE...
            return getattr(_mt5, nombre)
        raise AttributeError(nombre)

    def _nuevo_ticket(self) -> int:
        self._ticket += 1
        return self._ticket

    # --- reloj de velas ---
    def avanzar_vela(self, n: int = 1) -> None:
        self.bar_time += n * config.TIMEFRAME_SEGUNDOS
        self.tick.time = self.bar_time + 5

    def descargar_velas(self, desde_pos: int, cantidad: int) -> pd.DataFrame:
        """Sustituye a mt5_utils.descargar_velas: desde 0 es la vela en formación, desde 1 las cerradas."""
        fin = self.bar_time - (desde_pos - 1) * config.TIMEFRAME_SEGUNDOS
        tiempos = fin - config.TIMEFRAME_SEGUNDOS * np.arange(cantidad, 0, -1)
        rng = np.random.default_rng(0)
        close = 1.1 + np.cumsum(rng.normal(0, 0.0004, cantidad))
        open_ = np.r_[close[0], close[:-1]]
        return pd.DataFrame({"time": tiempos, "open": open_, "close": close, "high": np.maximum(open_, close) + 0.0002,
                             "low": np.minimum(open_, close) - 0.0002, "tick_volume": 300.0, "spread": 2,
                             "real_volume": 0})

    # --- conexión y cuenta ---
    def terminal_info(self):
        return SimpleNamespace(connected=True, trade_allowed=True) if self.conectado else None

    def reconectar(self) -> bool:
        self.conectado = True
        if self.login_tras_reconectar is not None:
            self.cuenta.login = self.login_tras_reconectar
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (1, "fake")

    def account_info(self):
        return self.cuenta

    def symbol_info(self, _simbolo):
        return self.info

    def symbol_info_tick(self, _simbolo):
        return self.tick

    def positions_get(self, symbol=None):
        return tuple(self.posiciones)

    def orders_get(self, symbol=None):
        return tuple(self.ordenes)

    def history_orders_get(self, _desde, _hasta):
        return tuple(self.historial_ordenes)

    def history_deals_get(self, _desde, _hasta):
        return tuple(self.deals)

    # --- órdenes ---
    def abrir_posicion(self, tipo=None, volumen=0.01, time=None) -> SimpleNamespace:
        pos = SimpleNamespace(ticket=self._nuevo_ticket(), type=_mt5.POSITION_TYPE_BUY if tipo is None else tipo,
                              volume=volumen, time=self.tick.time if time is None else time,
                              magic=config.MAGIC_NUMBER, price_open=self.tick.ask, sl=1.09, tp=1.12)
        self.posiciones.append(pos)
        return pos

    def cerrar_por_sl(self, pos) -> None:
        """El servidor cierra la posición por su SL (sin que el bot envíe nada)."""
        self.posiciones.remove(pos)
        self.deals.append(self._deal(_mt5.DEAL_ENTRY_OUT, self._nuevo_ticket(), pos.ticket, pos.volume, pos.sl, "sl"))

    def order_check(self, _request):
        return SimpleNamespace(retcode=0, comment="Done")

    def order_send(self, request):
        self.enviadas.append(request)
        retcode = self.respuestas.pop(0) if self.respuestas else _mt5.TRADE_RETCODE_DONE
        orden = self._nuevo_ticket()
        incierto = retcode == SIN_RESPUESTA or retcode in INCIERTOS
        volumen = round(request["volume"] * (self.fraccion_parcial if retcode == _mt5.TRADE_RETCODE_DONE_PARTIAL else 1), 8)
        ejecutada = False
        if retcode in (_mt5.TRADE_RETCODE_DONE, _mt5.TRADE_RETCODE_DONE_PARTIAL) or (incierto and self.modo_incierto == "ejecuta"):
            self._ejecutar(request, volumen, orden)
            ejecutada = True
        elif incierto and self.modo_incierto == "coloca":
            self.ordenes.append(SimpleNamespace(ticket=orden, time_setup=self.tick.time, state=_mt5.ORDER_STATE_PLACED,
                                                magic=request["magic"], symbol=request["symbol"], request=request))
        if retcode == SIN_RESPUESTA:
            return None
        return SimpleNamespace(retcode=retcode, order=orden if self.modo_incierto != "nada" or not incierto else 0,
                               volume=volumen if ejecutada else 0.0, price=request["price"], comment="fake")

    def ejecutar_colocadas(self) -> None:
        """Las órdenes que quedaron colocadas se ejecutan tarde."""
        for orden in list(self.ordenes):
            self.ordenes.remove(orden)
            self._ejecutar(orden.request, orden.request["volume"], orden.ticket)

    def cancelar_colocadas(self) -> None:
        for orden in list(self.ordenes):
            self.ordenes.remove(orden)
            self.historial_ordenes.append(SimpleNamespace(**{**vars(orden), "state": _mt5.ORDER_STATE_CANCELED}))

    def _deal(self, entry, orden, posicion, volumen, precio, comentario):
        return SimpleNamespace(ticket=self._nuevo_ticket(), order=orden, position_id=posicion, entry=entry,
                               magic=config.MAGIC_NUMBER, symbol=config.SYMBOL, time=self.tick.time, volume=volumen,
                               price=precio, profit=-1.5 if entry == _mt5.DEAL_ENTRY_OUT else 0.0,
                               swap=-0.1 if entry == _mt5.DEAL_ENTRY_OUT else 0.0, commission=0.0, fee=0.0,
                               comment=comentario)

    def _ejecutar(self, request, volumen, orden):
        self.historial_ordenes.append(SimpleNamespace(ticket=orden, time_setup=self.tick.time, magic=request["magic"],
                                                      symbol=request["symbol"], state=_mt5.ORDER_STATE_FILLED))
        if "position" in request:  # cierre total o parcial
            pos = next(p for p in self.posiciones if p.ticket == request["position"])
            self.posiciones.remove(pos)
            restante = round(pos.volume - volumen, 8)
            if restante > 0:
                self.posiciones.append(SimpleNamespace(**{**vars(pos), "volume": restante}))
            self.deals.append(self._deal(_mt5.DEAL_ENTRY_OUT, orden, pos.ticket, volumen, request["price"],
                                         request["comment"]))
        else:  # en MT5 el ticket de la posición es el de la orden que la abrió
            self.posiciones.append(SimpleNamespace(
                ticket=orden, volume=volumen, time=self.tick.time, magic=request["magic"],
                type=_mt5.POSITION_TYPE_BUY if request["type"] == _mt5.ORDER_TYPE_BUY else _mt5.POSITION_TYPE_SELL,
                price_open=request["price"], sl=request["sl"], tp=request["tp"]))
            self.deals.append(self._deal(_mt5.DEAL_ENTRY_IN, orden, orden, volumen, request["price"], request["comment"]))
