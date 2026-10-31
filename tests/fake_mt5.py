"""MT5 simulado para probar trading_bot.py sin terminal: cuenta, posiciones, órdenes, historial y velas en memoria."""
from types import SimpleNamespace

import MetaTrader5 as _mt5
import numpy as np
import pandas as pd

import config

SIN_RESPUESTA = "sin respuesta"  # en `respuestas`: order_send devuelve None


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
        self.respuestas = []             # retcodes forzados para los próximos order_send
        self.ejecutar_si_incierto = False
        self.fraccion_parcial = 0.5
        self._ticket = 5000

    def __getattr__(self, nombre):
        if nombre.isupper():  # constantes reales: ORDER_TYPE_BUY, TRADE_RETCODE_DONE...
            return getattr(_mt5, nombre)
        raise AttributeError(nombre)

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

    def history_deals_get(self, _desde, _hasta):
        return tuple(self.deals)

    # --- órdenes ---
    def abrir_posicion(self, tipo=None, volumen=0.01, time=None) -> SimpleNamespace:
        self._ticket += 1
        pos = SimpleNamespace(ticket=self._ticket, type=_mt5.POSITION_TYPE_BUY if tipo is None else tipo,
                              volume=volumen, time=self.tick.time if time is None else time,
                              magic=config.MAGIC_NUMBER, price_open=self.tick.ask, sl=1.09, tp=1.12)
        self.posiciones.append(pos)
        return pos

    def order_check(self, _request):
        return SimpleNamespace(retcode=0, comment="Done")

    def order_send(self, request):
        self.enviadas.append(request)
        retcode = self.respuestas.pop(0) if self.respuestas else _mt5.TRADE_RETCODE_DONE
        incierto = retcode == SIN_RESPUESTA or retcode in (_mt5.TRADE_RETCODE_PLACED, _mt5.TRADE_RETCODE_TIMEOUT,
                                                          _mt5.TRADE_RETCODE_CONNECTION)
        ejecuta = retcode in (_mt5.TRADE_RETCODE_DONE, _mt5.TRADE_RETCODE_DONE_PARTIAL) or (
            incierto and self.ejecutar_si_incierto)
        volumen = round(request["volume"] * (self.fraccion_parcial if retcode == _mt5.TRADE_RETCODE_DONE_PARTIAL else 1), 8)
        if ejecuta:
            self._aplicar(request, volumen)
        if retcode == SIN_RESPUESTA:
            return None
        return SimpleNamespace(retcode=retcode, order=self._ticket, volume=volumen if ejecuta else 0.0,
                               price=request["price"], comment="fake")

    def _aplicar(self, request, volumen):
        self._ticket += 1
        if "position" in request:  # cierre total o parcial
            pos = next(p for p in self.posiciones if p.ticket == request["position"])
            self.posiciones.remove(pos)
            restante = round(pos.volume - volumen, 8)
            if restante > 0:
                self.posiciones.append(SimpleNamespace(**{**vars(pos), "volume": restante}))
            self.deals.append(SimpleNamespace(ticket=self._ticket, position_id=pos.ticket, entry=_mt5.DEAL_ENTRY_OUT,
                                              magic=request["magic"], symbol=request["symbol"], time=self.tick.time,
                                              volume=volumen, price=request["price"], profit=-1.5, swap=-0.1,
                                              commission=0.0, fee=0.0, comment=request["comment"]))
        else:
            self.posiciones.append(SimpleNamespace(
                ticket=self._ticket, volume=volumen, time=self.tick.time, magic=request["magic"],
                type=_mt5.POSITION_TYPE_BUY if request["type"] == _mt5.ORDER_TYPE_BUY else _mt5.POSITION_TYPE_SELL,
                price_open=request["price"], sl=request["sl"], tp=request["tp"]))
