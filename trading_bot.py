"""Bucle de ejecución en tiempo real (ver Ejecucion.md).

Uso: python trading_bot.py   (Ctrl+C para detenerlo; crear logs/PAUSA bloquea nuevas entradas)
"""
import csv
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import joblib
import MetaTrader5 as mt5
import numpy as np

import config
import mt5_utils
import riesgo_cuenta
from features import build_features
from train_model import ventaja_significativa

log = logging.getLogger("trading_bot")

ACCOUNT_TRADE_MODE_REAL = getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)
FLAG_FILLING_FOK, FLAG_FILLING_IOC = 1, 2  # bits de symbol_info.filling_mode
RETCODES_EJECUTADA = {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL}
RETCODES_INCIERTOS = {mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_TIMEOUT, mt5.TRADE_RETCODE_CONNECTION}
ESTADOS_ORDEN_EJECUTADA = {mt5.ORDER_STATE_FILLED, mt5.ORDER_STATE_PARTIAL}
ESTADOS_ORDEN_DESCARTADA = {mt5.ORDER_STATE_CANCELED, mt5.ORDER_STATE_REJECTED, mt5.ORDER_STATE_EXPIRED}
EJECUTADA, RECHAZADA, INCIERTA = "ejecutada", "rechazada", "incierta"  # resultados de enviar()
CIERRE_POR_LIMITE = "limite"  # sin tilde: va también en el comentario de la orden
CAMPOS_OPERACIONES = ["fecha", "evento", "ticket", "tipo", "volumen_pedido", "volumen", "precio_pedido", "precio",
                      "slippage_puntos", "spread_puntos", "sl", "tp", "retcode", "comentario",
                      "beneficio", "swap", "comision", "tasas"]


def configurar_logs() -> None:
    config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(config.LOG_PATH, encoding="utf-8")],
    )


class InstanciaUnica:
    """Bloqueo exclusivo de un archivo: una segunda instancia del bot no puede arrancar.

    El sistema operativo libera el bloqueo si el proceso muere, así que no quedan bloqueos huérfanos.
    """

    def __init__(self, ruta):
        self.ruta = ruta
        self._archivo = None

    def __enter__(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._archivo = open(self.ruta, "a+")
        try:
            self._archivo.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._archivo.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._archivo, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._archivo.close()
            raise SystemExit(f"Ya hay otra instancia de trading_bot.py en marcha ({self.ruta})")
        return self

    def __exit__(self, *_):
        try:
            self._archivo.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._archivo.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._archivo.close()


def cargar_modelo() -> dict:
    """Carga el bundle y se detiene si fue entrenado con parámetros distintos a config.py."""
    if not config.MODEL_PATH.exists():
        raise SystemExit(f"No existe {config.MODEL_PATH}; ejecuta antes python train_model.py")
    bundle = joblib.load(config.MODEL_PATH)
    esperado = {
        "sl_mult": config.SL_MULTIPLICADOR_ATR,
        "tp_mult": config.TP_MULTIPLICADOR_ATR,
        "n_barras": config.MAX_BARS_EN_TRADE,
        "symbol": config.SYMBOL,
        "timeframe": config.TIMEFRAME,
    }
    distintos = {k: (bundle.get(k), v) for k, v in esperado.items() if bundle.get(k) != v}
    if distintos:
        raise SystemExit(f"El modelo no coincide con config.py, (modelo, config): {distintos}")
    if bundle["umbral_long"] is None and bundle["umbral_short"] is None:
        raise SystemExit("Ambos lados están desactivados en el modelo: no hay nada que operar")
    if config.EXIGIR_VENTAJA_SIGNIFICATIVA and not ventaja_significativa(bundle["informe"]):
        informe = bundle["informe"]
        if "robustez" not in informe or "holdout" not in informe:
            detalle = "el modelo es anterior a la comprobación de robustez y holdout; reentrénalo"
        else:
            holdout = informe["holdout"]
            detalle = ", ".join(f"semilla {r['semilla']}: trades={r['walk_forward']['agregado'].get('trades')} "
                                f"t={r['walk_forward']['agregado'].get('t_stat', 0.0):.2f}" for r in informe["robustez"])
            detalle += (f"; holdout desde {holdout.get('desde')}: trades={holdout.get('trades')} "
                        f"t={holdout.get('t_stat', 0.0):.2f}")
        raise SystemExit(
            f"El modelo no tiene ventaja significativa y robusta ({detalle}). Se exige t >= {config.T_MINIMO_TEST} "
            f"con >= {config.MIN_TRADES_TEST} trades y esperanza > 0 en >= {config.WF_MIN_PROP_VENTANAS_POSITIVAS:.0%} "
            f"de las ventanas con todas las semillas {list(config.SEMILLAS_ROBUSTEZ)}, y t >= {config.T_MINIMO_TEST} "
            f"con >= {config.MIN_TRADES_TEST} trades en el holdout. Para probar solo la mecánica en Demo: "
            "EXIGIR_VENTAJA_SIGNIFICATIVA = False")
    return bundle


def verificar_cuenta(login_esperado=None) -> int:
    """Comprueba la cuenta antes de operar y tras reconectar. Se detiene si cambió o si es real sin permiso."""
    cuenta = mt5.account_info()
    if cuenta is None:
        raise RuntimeError(f"No se pudo leer la cuenta: {mt5.last_error()}")
    if login_esperado is not None and cuenta.login != login_esperado:
        raise SystemExit(f"La terminal cambió de cuenta ({login_esperado} -> {cuenta.login}); el bot se detiene")
    if cuenta.trade_mode == ACCOUNT_TRADE_MODE_REAL and not config.PERMITIR_CUENTA_REAL:
        raise SystemExit("Cuenta REAL detectada. El bot solo opera en Demo salvo PERMITIR_CUENTA_REAL = True")
    return cuenta.login


al_esperar = None  # si se define, esperar_conexion la llama en cada reintento (el lanzador: latido, Detener, fin de sesión)


def esperar_conexion(login) -> None:
    """Espera a que la terminal esté conectada; si hubo que reconectar, vuelve a verificar la cuenta.

    Un corte más corto que AVISO_SIN_CONEXION_S solo queda en el log como información; a partir de ahí es un aviso.
    """
    inicio = None
    while True:
        terminal = mt5.terminal_info()
        if terminal is not None and terminal.connected:
            if inicio is not None:
                verificar_cuenta(login)
                duracion = time.monotonic() - inicio
                if duracion >= config.AVISO_SIN_CONEXION_S:
                    log.info("Conexión recuperada tras %.0f s; cuenta %s verificada", duracion, login)
                else:
                    log.info("Corte breve de conexión (%.0f s); cuenta %s verificada", duracion, login)
            return
        if inicio is None:
            inicio = time.monotonic()
        nivel = logging.WARNING if time.monotonic() - inicio >= config.AVISO_SIN_CONEXION_S else logging.INFO
        log.log(nivel, "Sin conexión con MT5; reintento en %ss", config.REINTENTO_CONEXION_S)
        if terminal is None:
            mt5.shutdown()
            mt5_utils.conectar()
        if al_esperar is not None:
            al_esperar()
        time.sleep(config.REINTENTO_CONEXION_S)


def posiciones_del_bot() -> list:
    posiciones = mt5.positions_get(symbol=config.SYMBOL)
    if posiciones is None:
        raise RuntimeError(f"positions_get falló: {mt5.last_error()}")
    return [p for p in posiciones if p.magic == config.MAGIC_NUMBER]


def posicion_abierta():
    propias = posiciones_del_bot()
    return propias[0] if propias else None


def ordenes_activas_del_bot() -> list:
    """Órdenes del bot colocadas en el servidor y todavía sin ejecutar ni cancelar."""
    ordenes = mt5.orders_get(symbol=config.SYMBOL)
    if ordenes is None:
        raise RuntimeError(f"orders_get falló: {mt5.last_error()}")
    return [o for o in ordenes if o.magic == config.MAGIC_NUMBER]


def historial_ordenes_del_bot() -> list:
    """Órdenes del bot de los últimos 7 días ya terminadas (ejecutadas, canceladas, rechazadas o expiradas).

    Justo después de conectar, el historial puede llegar incompleto: que una orden no aparezca no prueba nada.
    """
    ahora = datetime.now(timezone.utc)
    ordenes = mt5.history_orders_get(ahora - timedelta(days=7), ahora + timedelta(days=2))
    if ordenes is None:
        raise RuntimeError(f"history_orders_get falló: {mt5.last_error()}")
    return [o for o in ordenes if o.magic == config.MAGIC_NUMBER and o.symbol == config.SYMBOL]


def velas_en_trade(pos) -> int:
    """Velas cerradas que terminan después de la apertura (hora del servidor en ambos lados)."""
    cerradas = mt5_utils.descargar_velas(1, config.MAX_BARS_EN_TRADE + 5)
    return int(np.count_nonzero(cerradas["time"].to_numpy() + config.TIMEFRAME_SEGUNDOS > pos.time))


def modo_llenado(info) -> int:
    if info.filling_mode & FLAG_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if info.filling_mode & FLAG_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def calcular_volumen(info, sl_dist: float):
    """Lote fijo o por riesgo, ajustado a volume_step y limitado por MAX_LOTES. None si no cabe en volume_min."""
    if config.RIESGO_PCT is None:
        lotes = config.VOLUMEN
    else:
        # En símbolos que no cotizan en la moneda de la cuenta, el valor del tick al perder difiere del de ganar.
        valor_tick = getattr(info, "trade_tick_value_loss", 0.0) or info.trade_tick_value
        perdida_por_lote = sl_dist / info.trade_tick_size * valor_tick
        lotes = mt5.account_info().equity * config.RIESGO_PCT / perdida_por_lote
    lotes = min(lotes, info.volume_max, config.MAX_LOTES)
    lotes = round(math.floor(lotes / info.volume_step + 1e-9) * info.volume_step, 8)
    if lotes < info.volume_min:
        return None
    return lotes


def perdida_en_sl(tipo: int, simbolo: str, volumen: float, precio: float, sl: float) -> float:
    """Pérdida, en la moneda de la cuenta, si la operación toca su SL. Sin SL no tiene tope: infinito."""
    if not sl:
        return math.inf
    resultado = mt5.order_calc_profit(tipo, simbolo, volumen, precio, sl)
    if resultado is None:
        raise RuntimeError(f"order_calc_profit falló en {simbolo}: {mt5.last_error()}")
    return max(0.0, -resultado)  # un SL que ya asegura beneficio no añade riesgo


def exposicion_de_la_cuenta() -> tuple:
    """(posiciones abiertas, pérdida total si todas tocan su SL) de todos los bots de la cuenta."""
    posiciones = mt5.positions_get()
    if posiciones is None:
        raise RuntimeError(f"positions_get falló: {mt5.last_error()}")
    magics = config.MAGICS_CUENTA or {config.MAGIC_NUMBER}
    propias = [p for p in posiciones if p.magic in magics]
    riesgo = sum(perdida_en_sl(mt5.ORDER_TYPE_BUY if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_SELL,
                               p.symbol, p.volume, p.price_open, p.sl) for p in propias)
    return len(propias), riesgo


def motivo_exposicion(tipo: int, precio: float, sl: float, volumen: float):
    """Motivo por el que esta entrada no cabe en los límites de la cuenta, o None."""
    if config.MAX_POSICIONES_CUENTA is None and config.MAX_RIESGO_ABIERTO_PCT is None:
        return None
    abiertas, riesgo = exposicion_de_la_cuenta()
    if config.MAX_POSICIONES_CUENTA is not None and abiertas >= config.MAX_POSICIONES_CUENTA:
        return f"la cuenta ya tiene {abiertas} posiciones del bot (máximo {config.MAX_POSICIONES_CUENTA})"
    if config.MAX_RIESGO_ABIERTO_PCT is not None:
        total = riesgo + perdida_en_sl(tipo, config.SYMBOL, volumen, precio, sl)
        limite = config.MAX_RIESGO_ABIERTO_PCT * mt5.account_info().equity
        if total > limite:
            return (f"el riesgo abierto de la cuenta pasaría a {total:.2f} (límite "
                    f"{config.MAX_RIESGO_ABIERTO_PCT:.1%} de la equity = {limite:.2f})")
    return None


@contextmanager
def turno_de_la_cuenta():
    """Entre comprobar la exposición y enviar la orden no se cuela otro bot de la misma cuenta."""
    if config.BLOQUEO_CUENTA_PATH is None:
        yield
        return
    with riesgo_cuenta.turno(config.BLOQUEO_CUENTA_PATH, config.ESPERA_TURNO_CUENTA_S):
        yield


def registrar_operacion(fila: dict) -> None:
    """Añade una fila a logs/operaciones.csv (órdenes enviadas y cierres con su resultado)."""
    config.OPERACIONES_PATH.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not config.OPERACIONES_PATH.exists()
    with open(config.OPERACIONES_PATH, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_OPERACIONES)
        if nuevo:
            escritor.writeheader()
        escritor.writerow({"fecha": datetime.now().isoformat(timespec="seconds"), **fila})


def anotar(fila) -> None:
    """Registro de auditoría que nunca interrumpe el flujo de órdenes ni su reconciliación.

    `fila` es el dict o una función que lo construye (así también se capturan los fallos al construirlo). Si
    algo falla (disco, permisos, un dato inesperado), se avisa en el log y se sigue.
    """
    try:
        registrar_operacion(fila() if callable(fila) else fila)
    except Exception:
        log.exception("No se pudo escribir en %s; la operación sigue su curso", config.OPERACIONES_PATH)


def _fila_orden(request: dict, result, descripcion: str) -> dict:
    info, tick = mt5.symbol_info(config.SYMBOL), mt5.symbol_info_tick(config.SYMBOL)
    compra = request["type"] == mt5.ORDER_TYPE_BUY
    precio = getattr(result, "price", None) or None
    slippage = None
    if precio:
        slippage = round(((precio - request["price"]) if compra else (request["price"] - precio)) / info.point, 1)
    return {
        "evento": descripcion,
        "ticket": request.get("position") or getattr(result, "order", None),
        "tipo": "BUY" if compra else "SELL",
        "volumen_pedido": request["volume"],
        "volumen": getattr(result, "volume", None),
        "precio_pedido": request["price"],
        "precio": precio,
        "slippage_puntos": slippage,
        "spread_puntos": round((tick.ask - tick.bid) / info.point) if tick else None,
        "sl": request.get("sl"),
        "tp": request.get("tp"),
        "retcode": getattr(result, "retcode", None),
        "comentario": getattr(result, "comment", None),
    }


def enviar(request: dict, descripcion: str, confirmar) -> tuple:
    """order_check + order_send, sin reintentos. Devuelve (EJECUTADA | RECHAZADA | INCIERTA, result).

    DONE y DONE_PARTIAL cuentan como ejecutadas (una parcial se avisa en el log). Si el resultado es incierto
    (sin respuesta, PLACED, TIMEOUT o CONNECTION) se mira `confirmar()`, que comprueba las posiciones. Si aún
    no se confirma, queda INCIERTA: que no haya posición todavía no significa que la orden no vaya a ejecutarse.
    """
    check = mt5.order_check(request)
    if check is None:
        log.error("%s: order_check devolvió None: %s", descripcion, mt5.last_error())
        return RECHAZADA, None
    if check.retcode != 0:
        log.error("%s rechazada por order_check: retcode=%s comment=%s", descripcion, check.retcode, check.comment)
        return RECHAZADA, None
    result = mt5.order_send(request)
    retcode = None if result is None else result.retcode

    if retcode in RETCODES_EJECUTADA:
        if retcode == mt5.TRADE_RETCODE_DONE_PARTIAL:
            log.warning("%s ejecutada PARCIALMENTE: volumen=%s de %s", descripcion, result.volume, request["volume"])
        else:
            log.info("%s ejecutada: orden=%s volumen=%s precio=%s", descripcion, result.order, result.volume, result.price)
        anotar(lambda: _fila_orden(request, result, descripcion))
        return EJECUTADA, result

    if result is None or retcode in RETCODES_INCIERTOS:
        motivo = mt5.last_error() if result is None else f"retcode={retcode} comment={result.comment}"
        log.warning("%s con resultado incierto (%s); se reconcilia con las posiciones", descripcion, motivo)
        time.sleep(config.ESPERA_RECONCILIACION_S)
        if confirmar():
            log.info("%s confirmada al reconciliar", descripcion)
            anotar(lambda: _fila_orden(request, result, f"{descripcion} (reconciliada)"))
            return EJECUTADA, result
        log.warning("%s sin confirmar todavía; queda pendiente de resolver", descripcion)
        anotar(lambda: _fila_orden(request, result, f"{descripcion} (pendiente)"))
        return INCIERTA, result

    log.error("%s fallida: retcode=%s comment=%s", descripcion, retcode, result.comment)
    anotar(lambda: _fila_orden(request, result, f"{descripcion} (fallida)"))
    return RECHAZADA, result


def abrir(lado: str, atr: float, info, tick, estado: dict) -> None:
    verificar_cuenta(estado["login"])
    sl_dist = config.SL_MULTIPLICADOR_ATR * atr
    tp_dist = config.TP_MULTIPLICADOR_ATR * atr
    distancia_minima = info.trade_stops_level * info.point + (tick.ask - tick.bid)
    if sl_dist <= distancia_minima:
        log.warning("SL de %.5f por debajo de la distancia mínima %.5f; no se opera", sl_dist, distancia_minima)
        return
    volumen = calcular_volumen(info, sl_dist)
    if volumen is None:
        log.warning("El lote mínimo %s supera el riesgo permitido; no se opera", info.volume_min)
        return

    if lado == "long":
        tipo, precio, sl, tp = mt5.ORDER_TYPE_BUY, tick.ask, tick.ask - sl_dist, tick.ask + tp_dist
    else:
        tipo, precio, sl, tp = mt5.ORDER_TYPE_SELL, tick.bid, tick.bid + sl_dist, tick.bid - tp_dist
    sl, tp = round(sl, info.digits), round(tp, info.digits)

    try:
        with turno_de_la_cuenta():
            motivo = motivo_exposicion(tipo, precio, sl, volumen)
            if motivo:
                log.info("Sin entrada: %s", motivo)
                return
            _enviar_entrada(lado, tipo, precio, sl, tp, volumen, info, tick, estado)
    except riesgo_cuenta.CuentaOcupada as e:
        log.warning("Sin entrada: %s", e)


def _enviar_entrada(lado: str, tipo: int, precio: float, sl: float, tp: float, volumen: float, info, tick,
                    estado: dict) -> None:
    # La intención se persiste ANTES de enviar: si cualquier cosa falla a partir de aquí (una excepción, un
    # corte, un reinicio), queda constancia de que pudo haber una orden y no se abre otra hasta resolverla.
    estado["pendiente"] = {"orden": None, "lado": lado, "enviada_servidor": int(tick.time),
                           "enviada_local": time.time()}
    guardar_estado(estado)
    if estado.get("error_estado") or estado.get("estado_irrecuperable"):
        log.error("No se envía la entrada: no se pudo guardar la intención (%s)",
                  estado.get("error_estado") or "estado irrecuperable")
        return
    resultado, result = enviar({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": config.SYMBOL,
        "volume": volumen,
        "type": tipo,
        "price": precio,
        "sl": sl,
        "tp": tp,
        "deviation": config.DESVIACION_SLIPPAGE,
        "magic": config.MAGIC_NUMBER,
        "comment": "rf_mt5",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": modo_llenado(info),
    }, f"Apertura {lado.upper()}", confirmar=lambda: posicion_abierta() is not None)
    if resultado == INCIERTA:
        estado["pendiente"]["orden"] = getattr(result, "order", 0) or None  # hasta resolverla no se abre nada más
    else:
        estado["pendiente"] = None  # ejecutada (la posición existe) o rechazada (no hay orden)
    guardar_estado(estado)
    if resultado == EJECUTADA and config.AJUSTAR_SL_TP_A_EJECUCION:
        try:
            recolocar_sl_tp(precio, sl, tp, info)
        except Exception:
            log.exception("No se pudieron recolocar el SL y el TP; la posición conserva los que se enviaron")


def recolocar_sl_tp(precio_pedido: float, sl: float, tp: float, info) -> None:
    """Si la orden se ejecutó a otro precio, mueve el SL y el TP para que queden a la distancia prevista del
    precio real, como en el entrenamiento. Con ejecución a mercado el broker no respeta `deviation`, así que el
    deslizamiento puede ser cualquiera. Si la modificación falla, la posición conserva los niveles enviados.
    """
    pos = posicion_abierta()
    if pos is None:
        return
    desvio = pos.price_open - precio_pedido
    if abs(desvio) < info.point / 2:
        return
    nuevo_sl, nuevo_tp = round(sl + desvio, info.digits), round(tp + desvio, info.digits)
    result = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": config.SYMBOL, "position": pos.ticket,
                             "sl": nuevo_sl, "tp": nuevo_tp, "magic": config.MAGIC_NUMBER})
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info("SL y TP recolocados por una ejecución a %.1f puntos del precio pedido: sl=%s tp=%s",
                 desvio / info.point, nuevo_sl, nuevo_tp)
    else:
        log.warning("No se pudieron recolocar el SL y el TP (%s); la posición conserva los que se enviaron",
                    mt5.last_error() if result is None else f"retcode={result.retcode} comment={result.comment}")


def resolver_pendiente(estado: dict) -> None:
    """Resuelve una entrada con resultado incierto mirando posiciones, órdenes activas e historial.

    Mientras siga sin resolver no se abren nuevas entradas. Solo se da por no ejecutada con una prueba (la orden
    consta como cancelada, rechazada o expirada) o si tras PENDIENTE_SIN_RASTRO_S no aparece en ningún sitio.
    """
    pendiente = estado.get("pendiente")
    if not pendiente:
        return
    desde = pendiente["enviada_servidor"] - 5

    def es_la_pendiente(orden) -> bool:
        return orden.ticket == pendiente["orden"] if pendiente["orden"] else orden.time_setup >= desde

    if posicion_abierta() is not None:
        motivo = "se ejecutó: la posición está abierta"
    elif any(es_la_pendiente(o) for o in ordenes_activas_del_bot()):
        return  # sigue colocada en el servidor: esperar
    else:
        historial = [o for o in historial_ordenes_del_bot() if es_la_pendiente(o)]
        entradas = [d for d in deals_del_bot() if d.entry == mt5.DEAL_ENTRY_IN
                    and (d.order == pendiente["orden"] if pendiente["orden"] else d.time >= desde)]
        espera = time.time() - pendiente["enviada_local"]
        if entradas or any(o.state in ESTADOS_ORDEN_EJECUTADA for o in historial):
            motivo = "se ejecutó y la posición ya se cerró (consta en el historial)"
        elif historial and all(o.state in ESTADOS_ORDEN_DESCARTADA for o in historial):
            motivo = "no se ejecutó: la orden consta como cancelada, rechazada o expirada"
        elif config.PENDIENTE_SIN_RASTRO_S is not None and espera > config.PENDIENTE_SIN_RASTRO_S:
            motivo = f"sin rastro en posiciones, órdenes ni historial tras {espera:.0f} s; se da por no enviada"
        else:
            return
    log.warning("Entrada pendiente (orden %s) resuelta: %s", pendiente["orden"], motivo)
    estado["pendiente"] = None
    guardar_estado(estado)


def cerrar(pos, info, login, causa: str = "time stop", volumen: float | None = None) -> None:
    """Cierra la posición a mercado, o solo `volumen` de ella."""
    verificar_cuenta(login)
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if pos.type == mt5.POSITION_TYPE_BUY:
        tipo, precio = mt5.ORDER_TYPE_SELL, tick.bid
    else:
        tipo, precio = mt5.ORDER_TYPE_BUY, tick.ask
    enviar({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": config.SYMBOL,
        "volume": pos.volume if volumen is None else volumen,
        "type": tipo,
        "position": pos.ticket,
        "price": precio,
        "deviation": config.DESVIACION_SLIPPAGE,
        "magic": config.MAGIC_NUMBER,
        "comment": f"rf_mt5 {causa}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": modo_llenado(info),
    }, f"Cierre por {causa} (ticket {pos.ticket})",
        confirmar=lambda: all(p.ticket != pos.ticket or p.volume < pos.volume for p in posiciones_del_bot()))


GESTIONES = ("breakeven", "trailing", "parcial")


def plan_de_gestion(compra: bool, entrada: float, sl: float, tp: float, cerradas, point: float, gestion: str) -> tuple:
    """(stop nuevo o None, si el precio ya avanzó GESTION_ACTIVACION_R) con las velas cerradas desde la entrada.

    Es la regla de experimentos.py: el stop solo se mueve al cerrar una vela, con lo visto hasta ella, y nunca se
    aleja. La distancia de riesgo sale del objetivo, que no se mueve. La venta sale al ask (bid + spread de la vela).
    """
    sl_dist = abs(tp - entrada) * config.SL_MULTIPLICADOR_ATR / config.TP_MULTIPLICADOR_ATR
    if cerradas.empty or sl_dist <= 0:
        return None, False
    if compra:
        mejor = float(cerradas["high"].max())
        activada = mejor >= entrada + config.GESTION_ACTIVACION_R * sl_dist
        nuevo = mejor - sl_dist if gestion == "trailing" else entrada if activada else None
        return (nuevo if nuevo is not None and nuevo > sl + point / 2 else None), activada
    mejor = float((cerradas["low"] + cerradas["spread"] * point).min())
    activada = mejor <= entrada - config.GESTION_ACTIVACION_R * sl_dist
    nuevo = mejor + sl_dist if gestion == "trailing" else entrada if activada else None
    return (nuevo if nuevo is not None and nuevo < sl - point / 2 else None), activada


def ajustar_salida(pos, info, login) -> None:
    """Con GESTION, al abrir cada vela: cierra media posición si toca y acerca el stop. Si el precio ya pasó el
    stop nuevo, o el broker lo rechaza, la posición conserva el que tenía y se vuelve a intentar en la vela siguiente."""
    if config.GESTION is None or not pos.sl or not pos.tp:
        return
    compra = pos.type == mt5.POSITION_TYPE_BUY
    cerradas = mt5_utils.descargar_velas(1, config.MAX_BARS_EN_TRADE + 5)
    cerradas = cerradas[cerradas["time"] + config.TIMEFRAME_SEGUNDOS > pos.time]  # desde la vela de la entrada
    nuevo_sl, activada = plan_de_gestion(compra, pos.price_open, pos.sl, pos.tp, cerradas, info.point, config.GESTION)
    if config.GESTION == "parcial" and activada and not any(
            d.position_id == pos.ticket and d.entry == mt5.DEAL_ENTRY_OUT for d in deals_del_bot()):
        mitad = round(math.floor(pos.volume / 2 / info.volume_step + 1e-9) * info.volume_step, 8)
        if mitad >= info.volume_min and not ordenes_activas_del_bot():
            cerrar(pos, info, login, "parcial", mitad)
    if nuevo_sl is None:
        return
    tick = mt5.symbol_info_tick(config.SYMBOL)
    margen = getattr(info, "trade_stops_level", 0) * info.point
    if tick is None or (compra and tick.bid - margen <= nuevo_sl) or (not compra and tick.ask + margen >= nuevo_sl):
        log.info("El stop no se puede mover a %s: el precio ya está ahí; se conserva %s", round(nuevo_sl, info.digits), pos.sl)
        return
    nuevo_sl = round(nuevo_sl, info.digits)
    result = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": config.SYMBOL, "position": pos.ticket,
                             "sl": nuevo_sl, "tp": pos.tp, "magic": config.MAGIC_NUMBER})
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info("Stop movido de %s a %s (%s)", pos.sl, nuevo_sl, config.GESTION)
    else:
        log.warning("No se pudo mover el stop a %s (%s); la posición conserva %s", nuevo_sl,
                    mt5.last_error() if result is None else f"retcode={result.retcode} comment={result.comment}", pos.sl)


def gestionar_posicion(pos, info, estado: dict, cuenta=None) -> None:
    """Time stop y, con CERRAR_AL_LIMITE, cierre al alcanzar el límite de drawdown o de pérdida diaria. Se
    comprueba en cada vuelta del bucle; si un cierre falla se reintenta tras REINTENTO_CIERRE_S."""
    ahora = time.monotonic()
    if ahora - estado.get("ultimo_intento_cierre", -math.inf) < config.REINTENTO_CIERRE_S:
        return
    causa = "time stop" if velas_en_trade(pos) >= config.MAX_BARS_EN_TRADE else None
    if causa is None and config.CERRAR_AL_LIMITE:
        limite = motivo_de_limite(estado, cuenta or mt5.account_info())
        if limite:
            log.warning("Límite alcanzado con una posición abierta (%s); se cierra", limite)
            causa = CIERRE_POR_LIMITE
    if causa is None or ordenes_activas_del_bot():  # un cierre anterior sigue colocado en el servidor: no se envía otro
        return
    estado["ultimo_intento_cierre"] = ahora
    cerrar(pos, info, estado["login"], causa)


def deals_del_bot() -> list:
    """Deals del bot de los últimos 7 días (margen amplio: el filtro por fecha de MT5 no es en hora del servidor)."""
    ahora = datetime.now(timezone.utc)
    deals = mt5.history_deals_get(ahora - timedelta(days=7), ahora + timedelta(days=2))
    if deals is None:
        raise RuntimeError(f"history_deals_get falló: {mt5.last_error()}")
    return [d for d in deals if d.magic == config.MAGIC_NUMBER and d.symbol == config.SYMBOL]


def registrar_deals(estado: dict) -> None:
    """Registra los deals nuevos del bot: entradas (con la comisión que se cobre al entrar) y cierres (por SL, TP
    o time stop) con su beneficio, swap, comisión y tasas."""
    for d in deals_del_bot():
        if d.ticket in estado["deals_vistos"]:
            continue
        estado["deals_vistos"].add(d.ticket)
        evento = "cierre" if d.entry == mt5.DEAL_ENTRY_OUT else "entrada"
        log.info("Deal de %s, posición %s: precio=%s beneficio=%s swap=%s comisión=%s tasas=%s (%s)", evento,
                 d.position_id, d.price, d.profit, d.swap, d.commission, getattr(d, "fee", 0.0), d.comment)
        anotar({"evento": evento, "ticket": d.position_id, "volumen": d.volume, "precio": d.price,
                "comentario": d.comment, "beneficio": d.profit, "swap": d.swap,
                "comision": d.commission, "tasas": getattr(d, "fee", 0.0)})


CLAVES_PERSISTIDAS = ("equity_max", "pendiente", "last_bar_time")
CLAVES_PENDIENTE = {"orden", "lado", "enviada_servidor", "enviada_local"}


def rutas_estado() -> tuple:
    """(principal, temporal, respaldo). El temporal solo existe a mitad de una escritura; el respaldo es la
    versión anterior a la última escritura."""
    ruta = config.ESTADO_PATH
    return ruta, ruta.with_name(ruta.name + ".tmp"), ruta.with_name(ruta.name + ".bak")


def validar_estado(datos) -> None:
    """ValueError si el contenido no es un estado de riesgo utilizable."""
    if not isinstance(datos, dict):
        raise ValueError("no es un objeto JSON")
    equity_max = datos.get("equity_max")
    if isinstance(equity_max, bool) or not isinstance(equity_max, (int, float)) or not math.isfinite(equity_max) \
            or equity_max <= 0:
        raise ValueError(f"equity_max inválido: {equity_max!r}")
    pendiente = datos.get("pendiente")
    if pendiente is not None and (not isinstance(pendiente, dict) or not CLAVES_PENDIENTE <= pendiente.keys()):
        raise ValueError(f"pendiente inválida: {pendiente!r}")
    last_bar_time = datos.get("last_bar_time")
    if last_bar_time is not None and (isinstance(last_bar_time, bool) or not isinstance(last_bar_time, int)):
        raise ValueError(f"last_bar_time inválido: {last_bar_time!r}")


def _leer_estado(ruta):
    """Contenido válido de `ruta`, None si no existe. ValueError u OSError si existe pero no se puede usar."""
    try:
        texto = ruta.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    datos = json.loads(texto)
    validar_estado(datos)
    return datos


def cargar_estado() -> dict:
    """Estado persistido. Prueba el archivo principal, luego el temporal (una escritura interrumpida justo antes
    de terminar) y luego el respaldo.

    Sin ningún archivo es la primera ejecución y devuelve {}. Si hay archivos pero ninguno es válido, devuelve
    {"estado_irrecuperable": motivo}: el máximo de equity se ha perdido y no se abren nuevas entradas hasta que
    alguien lo revise. Adoptar la equity actual como máximo borraría en silencio el drawdown acumulado.
    """
    problemas = []
    for ruta in rutas_estado():
        try:
            datos = _leer_estado(ruta)
        except (ValueError, OSError) as e:
            problemas.append(f"{ruta.name}: {e}")
            continue
        if datos is None:
            continue
        if ruta != config.ESTADO_PATH or problemas:
            log.warning("Estado recuperado de %s (%s)", ruta, "; ".join(problemas) or "falta el principal")
        return datos
    if problemas:
        motivo = "; ".join(problemas)
        log.error("Estado de riesgo irrecuperable: %s", motivo)
        return {"estado_irrecuperable": motivo}
    return {}


def guardar_estado(estado: dict) -> None:
    """Persiste lo que debe sobrevivir a un reinicio: el máximo de equity, la entrada pendiente y la última
    vela procesada (para no volver a evaluar entradas en la misma vela tras reiniciar).

    Escritura atómica: se escribe un temporal completo (con fsync) y se renombra encima del principal, que antes
    pasa a ser el respaldo. Un corte a mitad nunca deja un principal a medias. Si el estado era irrecuperable no
    se escribe nada, para no tapar la evidencia; si la escritura falla, se bloquean las entradas hasta que funcione.
    """
    if estado.get("estado_irrecuperable"):
        return
    principal, temporal, respaldo = rutas_estado()
    datos = {k: estado[k] for k in CLAVES_PERSISTIDAS if k in estado}
    try:
        principal.parent.mkdir(parents=True, exist_ok=True)
        with open(temporal, "w", encoding="utf-8") as f:
            json.dump(datos, f)
            f.flush()
            os.fsync(f.fileno())
        try:
            principal_valido = _leer_estado(principal) is not None
        except (ValueError, OSError):
            principal_valido = False  # no se rota un principal corrupto encima de un respaldo bueno
        if principal_valido:
            os.replace(principal, respaldo)
        os.replace(temporal, principal)
    except OSError as e:
        estado["error_estado"] = f"no se pudo guardar {principal}: {e}"
        log.error("%s", estado["error_estado"])
        return
    estado.pop("error_estado", None)


def actualizar_equity_max(estado: dict, equity: float) -> None:
    """Máximo de equity de la cuenta, persistido para que el límite de drawdown sobreviva a reinicios.

    Con ESTADO_CUENTA_PATH, el máximo es el del archivo que comparten todos los bots de la cuenta: borrarlo
    reinicia el drawdown de todos con la equity actual. Si está corrupto se bloquean las entradas; si solo no
    se pudo leer o escribir en esta vuelta, se sigue con el último máximo conocido y se reintenta en la siguiente.
    """
    if config.ESTADO_CUENTA_PATH is not None:
        try:
            compartido = riesgo_cuenta.subir_equity_max(config.ESTADO_CUENTA_PATH, config.BLOQUEO_CUENTA_PATH, equity)
        except ValueError as e:
            if "error_cuenta" not in estado:
                log.error("Máximo de equity de la cuenta ilegible: %s", e)
            estado["error_cuenta"] = (f"máximo de equity de la cuenta ilegible ({e}); revisa o borra "
                                      f"{config.ESTADO_CUENTA_PATH}")
        except (OSError, riesgo_cuenta.CuentaOcupada):
            log.debug("No se pudo sincronizar el máximo de equity de la cuenta", exc_info=True)
        else:
            estado.pop("error_cuenta", None)
            if compartido != estado.get("equity_max"):
                estado["equity_max"] = compartido
                guardar_estado(estado)
            return
    if equity > estado.get("equity_max", -math.inf):
        estado["equity_max"] = equity
        guardar_estado(estado)


def motivo_sin_entradas(estado: dict, cuenta):
    """Motivo para no abrir nuevas entradas, o None. La gestión de la posición abierta sigue igual."""
    if estado.get("estado_irrecuperable"):
        principal, _, respaldo = rutas_estado()
        return (f"estado de riesgo irrecuperable ({estado['estado_irrecuperable']}). Revisa {principal} y "
                f"{respaldo.name}; para empezar de cero con la equity actual como máximo, bórralos y reinicia")
    if estado.get("error_estado"):
        return estado["error_estado"]
    if estado.get("error_cuenta"):
        return estado["error_cuenta"]
    if estado.get("pendiente"):
        return f"entrada pendiente de resolver (orden {estado['pendiente']['orden']})"
    if ordenes_activas_del_bot():
        return "hay una orden del bot colocada en el servidor"
    if config.PAUSA_PATH.exists():
        return f"pausa manual (existe {config.PAUSA_PATH})"
    return motivo_de_limite(estado, cuenta)


def motivo_de_limite(estado: dict, cuenta):
    """El límite de drawdown o de pérdida diaria que se alcanzó, o None."""
    drawdown = (estado["equity_max"] - cuenta.equity) / estado["equity_max"]
    if drawdown >= config.MAX_DRAWDOWN_PCT:
        if config.ESTADO_CUENTA_PATH is not None:
            reanudar = f"borra {config.ESTADO_CUENTA_PATH} (reinicia el drawdown de todos los bots de la cuenta)"
        else:
            reanudar = f"borra {config.ESTADO_PATH} y {rutas_estado()[2].name}"
        return (f"drawdown de {drawdown:.1%} desde el máximo de equity {estado['equity_max']:.2f} "
                f"(límite {config.MAX_DRAWDOWN_PCT:.0%}); para reanudar, revisa y {reanudar}")
    realizado, flotante = resultado_del_dia()
    base = cuenta.balance - realizado  # balance al empezar el día
    perdida = -(realizado + flotante)
    if perdida >= config.MAX_PERDIDA_DIARIA_PCT * base:
        return (f"pérdida del día de {perdida:.2f} (límite {config.MAX_PERDIDA_DIARIA_PCT:.0%} de {base:.2f}; "
                f"realizado {realizado:.2f}, flotante {flotante:.2f})")
    return None


def importe_deal(d) -> float:
    """Todo lo que un deal mueve en el balance: beneficio, swap, comisión y tasas."""
    return d.profit + d.swap + d.commission + getattr(d, "fee", 0.0)


def resultado_del_dia() -> tuple:
    """(realizado, flotante) del bot desde las 00:00 del servidor.

    El realizado suma **todos** los deals del bot del día, de entrada y de salida: muchos brokers cobran la
    comisión al entrar. El flotante es el de sus posiciones abiertas. Los cargos de la cuenta que no llevan el
    magic del bot (comisiones mensuales, por ejemplo) no se le pueden atribuir; el límite de drawdown, que mira
    la equity de toda la cuenta, sí los cubre.
    """
    inicio_dia = mt5.symbol_info_tick(config.SYMBOL).time // 86400 * 86400
    realizado = sum(importe_deal(d) for d in deals_del_bot() if d.time >= inicio_dia)
    flotante = sum(getattr(p, "profit", 0.0) + getattr(p, "swap", 0.0) for p in posiciones_del_bot())
    return realizado, flotante


def actividad_en_la_vela(bar_time: int) -> bool:
    """True si el bot ya colocó alguna orden o tuvo algún deal desde la apertura de `bar_time` (hora del servidor).

    Segunda defensa frente a reentrar en la misma vela tras un reinicio, por si se perdió el estado persistido:
    una entrada que abrió y cerró por SL/TP en esta vela deja su rastro en el historial.
    """
    return (any(o.time_setup >= bar_time for o in ordenes_activas_del_bot())
            or any(o.time_setup >= bar_time for o in historial_ordenes_del_bot())
            or any(d.time >= bar_time for d in deals_del_bot()))


def esperar_historial(lecturas: int = 10) -> None:
    """Al conectar, la terminal puede estar todavía sincronizando el historial con el servidor y devolverlo
    incompleto. Se espera a que el número de órdenes y deals deje de cambiar (mitigación, no garantía)."""
    anterior = None
    for _ in range(lecturas):
        ahora = datetime.now(timezone.utc)
        desde, hasta = ahora - timedelta(days=7), ahora + timedelta(days=2)
        total = (mt5.history_orders_total(desde, hasta), mt5.history_deals_total(desde, hasta))
        if total == anterior:
            return
        anterior = total
        time.sleep(1)
    log.warning("El historial de MT5 no se estabilizó tras %s lecturas: %s", lecturas, anterior)


def es_salto_de_fin_de_semana(anterior: int, siguiente: int) -> bool:
    """Viernes -> sábado, domingo o lunes, en menos de 3 días (hora del servidor; lunes = 0, como en features)."""
    dia = lambda t: ((t // 86400) + 3) % 7  # noqa: E731
    return dia(anterior) == 4 and dia(siguiente) in (5, 6, 0) and siguiente - anterior <= 3 * 86400


def velas_para_la_senal(bar_time: int):
    """(velas cerradas, None) si son las que preceden a la vela `bar_time` que dispara la señal, o (None, motivo).

    Se descargan en una sola consulta la vela en formación y las cerradas anteriores, para que sean coherentes
    entre sí. Se exige que la vela en formación sea `bar_time`, que haya BUFFER_VELAS cerradas y que las últimas
    VELAS_CONTIGUAS_MIN sean consecutivas (solo se admite el salto del fin de semana). Un bloque atrasado o a
    medio sincronizar deja un hueco justo antes de la vela actual y no pasa. Los huecos más antiguos (festivos,
    cortes de datos) se aceptan, porque el entrenamiento también los tiene.
    """
    velas = mt5_utils.descargar_velas(0, config.BUFFER_VELAS + 1)
    tiempos = velas["time"].to_numpy(dtype=np.int64)
    if tiempos[-1] != bar_time:
        return None, f"la vela en formación ({tiempos[-1]}) no es la que disparó la señal ({bar_time})"
    if len(velas) - 1 < config.BUFFER_VELAS:
        return None, f"histórico insuficiente: {len(velas) - 1} velas cerradas de {config.BUFFER_VELAS}"
    recientes = tiempos[-(config.VELAS_CONTIGUAS_MIN + 1):]
    for anterior, siguiente in zip(recientes[:-1], recientes[1:]):
        if siguiente - anterior != config.TIMEFRAME_SEGUNDOS and not es_salto_de_fin_de_semana(anterior, siguiente):
            return None, (f"faltan velas entre {anterior} y {siguiente} (hora del servidor): datos atrasados o "
                          "incompletos")
    return velas.iloc[:-1].reset_index(drop=True), None


def evaluar_entrada(bundle: dict, info, bar_time: int, estado: dict) -> None:
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick falló: {mt5.last_error()}")
    retraso = tick.time - bar_time
    if retraso > config.MAX_RETRASO_ENTRADA_S:
        log.info("Vela nueva detectada %ss después de su apertura; se espera a la siguiente", retraso)
        return
    spread_puntos = round((tick.ask - tick.bid) / info.point)
    if spread_puntos > config.MAX_SPREAD_PUNTOS:
        log.warning("Spread de %s puntos > %s; no se opera", spread_puntos, config.MAX_SPREAD_PUNTOS)
        return

    cerradas, motivo = velas_para_la_senal(bar_time)
    if cerradas is None:
        log.warning("Velas no válidas para la señal (%s); no se opera", motivo)
        return
    datos = build_features(cerradas, info.point)
    x = datos[bundle["features"]].iloc[[-1]]
    atr = float(datos["atr"].iloc[-1])
    if not np.isfinite(x.to_numpy(dtype=float)).all() or not np.isfinite(atr) or atr <= 0:
        log.warning("Features no finitas o ATR no válido en la última vela cerrada; no se opera")
        return

    p_long = bundle["model_long"].predict_proba(x)[0, 1]
    p_short = bundle["model_short"].predict_proba(x)[0, 1]
    señal_long = bundle["umbral_long"] is not None and p_long >= bundle["umbral_long"]
    señal_short = bundle["umbral_short"] is not None and p_short >= bundle["umbral_short"]
    log.info("P_long=%.3f P_short=%.3f", p_long, p_short)
    if señal_long == señal_short:
        return
    abrir("long" if señal_long else "short", atr, info, tick, estado)


def ciclo(bundle: dict, estado: dict) -> None:
    """Una vuelta del bucle. La posición abierta se gestiona en cada vuelta; las entradas, solo con vela nueva."""
    esperar_conexion(estado["login"])
    cuenta = mt5.account_info()
    if cuenta is None:
        raise RuntimeError(f"account_info falló: {mt5.last_error()}")
    actualizar_equity_max(estado, cuenta.equity)
    resolver_pendiente(estado)
    info = mt5.symbol_info(config.SYMBOL)
    pos = posicion_abierta()
    if pos is not None:
        gestionar_posicion(pos, info, estado, cuenta)

    bar_time = int(mt5_utils.descargar_velas(0, 1)["time"].iloc[0])
    if bar_time == estado["last_bar_time"]:
        return
    # Se marca y se persiste antes de evaluar: una entrada que falla no se reintenta a mitad de vela (no
    # coincidiría con el entrenamiento), tampoco tras un reinicio. La gestión de la posición no depende de esto.
    estado["last_bar_time"] = bar_time
    guardar_estado(estado)
    try:
        registrar_deals(estado)
    except RuntimeError:
        log.warning("No se pudo leer el historial de deals", exc_info=True)
    if pos is not None:
        if posicion_abierta() is not None:  # sigue abierta tras gestionar_posicion
            ajustar_salida(pos, info, estado["login"])
        return
    motivo = motivo_sin_entradas(estado, cuenta)
    if motivo is None and actividad_en_la_vela(bar_time):
        motivo = "el bot ya operó en esta vela (consta en el historial)"
    if motivo:
        log.info("Sin nuevas entradas: %s", motivo)
        return
    evaluar_entrada(bundle, info, bar_time, estado)


def inicializar_ultima_vela(estado: dict) -> None:
    """En el primer arranque (sin vela guardada) no se evalúa la vela que ya estaba abierta.

    El control de retraso compara con el último tick, y con el mercado cerrado ese tick es tan viejo como la
    vela: sin esto, arrancar en fin de semana tomaría la última vela del viernes por una vela nueva.
    """
    if estado.get("last_bar_time") is None:
        estado["last_bar_time"] = int(mt5_utils.descargar_velas(0, 1)["time"].iloc[0])
        guardar_estado(estado)
        log.info("Primer arranque: la vela abierta (%s) no se evalúa; se espera a la siguiente", estado["last_bar_time"])


def main() -> None:
    configurar_logs()
    with InstanciaUnica(config.LOCK_PATH):
        bundle = cargar_modelo()
        if not config.EXIGIR_VENTAJA_SIGNIFICATIVA:
            log.warning("EXIGIR_VENTAJA_SIGNIFICATIVA = False: el modelo no ha demostrado ventaja. Solo para probar "
                        "la mecánica en una cuenta Demo")
        if not mt5_utils.conectar():
            raise SystemExit(f"No se pudo conectar con MT5: {mt5.last_error()}")
        try:
            estado = {"login": verificar_cuenta(), "last_bar_time": None, **cargar_estado()}
            esperar_historial()
            estado["deals_vistos"] = {d.ticket for d in deals_del_bot()}
            cuenta = mt5.account_info()
            actualizar_equity_max(estado, cuenta.equity)
            inicializar_ultima_vela(estado)
            if not mt5.terminal_info().trade_allowed:
                log.warning("El trading algorítmico está desactivado en la terminal (botón Algo Trading)")
            log.info("Bot iniciado: %s %s, cuenta %s, umbral_long=%s umbral_short=%s (modelo del %s)",
                     config.SYMBOL, config.TIMEFRAME, estado["login"], bundle["umbral_long"],
                     bundle["umbral_short"], bundle.get("entrenado"))
            while True:
                try:
                    ciclo(bundle, estado)
                except Exception:
                    log.exception("Error en el ciclo; se reintenta en la siguiente vuelta")
                time.sleep(config.SLEEP_S)
        except KeyboardInterrupt:
            log.info("Bot detenido por el usuario")
        finally:
            mt5.shutdown()


if __name__ == "__main__":
    main()
