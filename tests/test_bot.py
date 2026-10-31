"""Pruebas operativas de trading_bot.py con un MT5 simulado: fallos, reinicios, parciales, duplicados y límites."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

import config
import mt5_utils
import trading_bot as tb
from features import FEATURES
from fake_mt5 import SIN_RESPUESTA, FakeMT5

DONE, PARCIAL, RECHAZO = tb.mt5.TRADE_RETCODE_DONE, tb.mt5.TRADE_RETCODE_DONE_PARTIAL, tb.mt5.TRADE_RETCODE_REJECT
TIMEOUT, PLACED = tb.mt5.TRADE_RETCODE_TIMEOUT, tb.mt5.TRADE_RETCODE_PLACED


class ModeloFijo:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.array([[1 - self.p, self.p]] * len(x))


@pytest.fixture
def mt5(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setattr(tb, "mt5", fake)
    monkeypatch.setattr(mt5_utils, "descargar_velas", fake.descargar_velas)
    monkeypatch.setattr(mt5_utils, "conectar", fake.reconectar)
    monkeypatch.setattr(config, "REINTENTO_CIERRE_S", 0)
    return fake


@pytest.fixture
def bundle():
    """Señal de compra en cada vela."""
    return {"features": FEATURES, "model_long": ModeloFijo(0.9), "model_short": ModeloFijo(0.1),
            "umbral_long": 0.5, "umbral_short": 0.5}


def nuevo_estado(fake):
    return {"login": fake.cuenta.login, "last_bar_time": None, "deals_vistos": set()}


def aperturas(fake):
    return [r for r in fake.enviadas if "position" not in r]


def cierres(fake):
    return [r for r in fake.enviadas if "position" in r]


def operaciones():
    with open(config.OPERACIONES_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------- gestión de la posición ----------------
def test_cierre_temporal_fallido_se_reintenta_sin_esperar_otra_vela(mt5, bundle):
    estado = nuevo_estado(mt5)
    mt5.abrir_posicion()
    mt5.avanzar_vela(config.MAX_BARS_EN_TRADE)
    mt5.respuestas = [RECHAZO]
    tb.ciclo(bundle, estado)
    assert len(cierres(mt5)) == 1 and len(mt5.posiciones) == 1  # el cierre falló
    tb.ciclo(bundle, estado)                                      # misma vela: se reintenta igual
    assert len(cierres(mt5)) == 2 and mt5.posiciones == []
    assert aperturas(mt5) == []                                   # en la vela del time stop no se entra


def test_reintento_de_cierre_respeta_la_pausa(mt5, bundle, monkeypatch):
    monkeypatch.setattr(config, "REINTENTO_CIERRE_S", 3600)
    estado = nuevo_estado(mt5)
    mt5.abrir_posicion()
    mt5.avanzar_vela(config.MAX_BARS_EN_TRADE)
    mt5.respuestas = [RECHAZO]
    tb.ciclo(bundle, estado)
    tb.ciclo(bundle, estado)
    assert len(cierres(mt5)) == 1  # no se martillea el servidor cada 2 s


def test_no_cierra_antes_del_time_stop(mt5, bundle):
    estado = nuevo_estado(mt5)
    mt5.abrir_posicion()
    mt5.avanzar_vela(config.MAX_BARS_EN_TRADE - 1)
    tb.ciclo(bundle, estado)
    assert cierres(mt5) == [] and len(mt5.posiciones) == 1


def test_reinicio_con_posicion_vencida_la_cierra(mt5, bundle):
    mt5.abrir_posicion(time=mt5.tick.time - 100 * config.TIMEFRAME_SEGUNDOS)  # abierta antes del reinicio
    tb.ciclo(bundle, nuevo_estado(mt5))
    assert len(cierres(mt5)) == 1 and mt5.posiciones == []


def test_cierre_parcial_se_completa_en_la_siguiente_vuelta(mt5, bundle):
    estado = nuevo_estado(mt5)
    mt5.abrir_posicion(volumen=0.02)
    mt5.avanzar_vela(config.MAX_BARS_EN_TRADE)
    mt5.respuestas = [PARCIAL]
    tb.ciclo(bundle, estado)
    assert [p.volume for p in mt5.posiciones] == [0.01]
    tb.ciclo(bundle, estado)
    assert mt5.posiciones == [] and [r["volume"] for r in cierres(mt5)] == [0.02, 0.01]


# ---------------- entradas ----------------
def test_abre_en_vela_nueva_y_no_duplica(mt5, bundle):
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1 and len(mt5.posiciones) == 1
    tb.ciclo(bundle, estado)       # misma vela
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)       # vela nueva con posición abierta
    assert len(aperturas(mt5)) == 1


def test_no_entra_si_la_vela_se_detecta_tarde(mt5, bundle):
    mt5.tick.time = mt5.bar_time + config.MAX_RETRASO_ENTRADA_S + 1
    tb.ciclo(bundle, nuevo_estado(mt5))
    assert aperturas(mt5) == []


def test_apertura_parcial_cuenta_como_ejecutada(mt5, bundle):
    mt5.respuestas = [PARCIAL]
    tb.ciclo(bundle, nuevo_estado(mt5))
    assert [p.volume for p in mt5.posiciones] == [0.005]
    assert operaciones()[0]["volumen"] == "0.005"


def test_resultado_incierto_se_reconcilia_y_no_duplica(mt5, bundle):
    estado = nuevo_estado(mt5)
    mt5.respuestas, mt5.modo_incierto = [SIN_RESPUESTA], "ejecuta"
    tb.ciclo(bundle, estado)
    assert len(mt5.posiciones) == 1 and estado.get("pendiente") is None
    assert operaciones()[0]["evento"].endswith("(reconciliada)")
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1


def test_resultado_incierto_sin_posicion_no_se_da_por_fallido(mt5):
    request = {"type": tb.mt5.ORDER_TYPE_BUY, "volume": 0.01, "price": 1.1, "magic": config.MAGIC_NUMBER,
               "symbol": config.SYMBOL, "sl": 1.09, "tp": 1.12, "comment": "x"}
    mt5.respuestas = [TIMEOUT]
    resultado, _ = tb.enviar(request, "Prueba", confirmar=lambda: tb.posicion_abierta() is not None)
    assert resultado == tb.INCIERTA and mt5.posiciones == []


# ---------------- órdenes con resultado incierto (regresión: ejecución tardía) ----------------
def test_orden_colocada_que_se_ejecuta_tarde_no_duplica(mt5, bundle):
    """PLACED sin posición todavía: antes el bot lo daba por no ejecutado y abría otra entrada en la vela siguiente."""
    estado = nuevo_estado(mt5)
    mt5.respuestas, mt5.modo_incierto = [PLACED], "coloca"
    tb.ciclo(bundle, estado)
    assert mt5.posiciones == [] and len(mt5.ordenes) == 1 and estado["pendiente"]["orden"] == mt5.ordenes[0].ticket
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)                 # la orden sigue colocada: no se abre otra
    assert len(aperturas(mt5)) == 1
    mt5.ejecutar_colocadas()                 # ejecución tardía
    tb.ciclo(bundle, estado)
    assert estado["pendiente"] is None
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1 and len(mt5.posiciones) == 1


def test_orden_colocada_y_cancelada_libera_las_entradas(mt5, bundle):
    estado = nuevo_estado(mt5)
    mt5.respuestas, mt5.modo_incierto = [PLACED], "coloca"
    tb.ciclo(bundle, estado)
    mt5.cancelar_colocadas()
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)                 # consta como cancelada: se resuelve y se puede volver a entrar
    assert estado["pendiente"] is None and len(aperturas(mt5)) == 2 and len(mt5.posiciones) == 1


def test_pendiente_ejecutada_y_cerrada_por_sl_se_resuelve(mt5, bundle):
    estado = nuevo_estado(mt5)
    mt5.respuestas, mt5.modo_incierto = [PLACED], "coloca"
    tb.ciclo(bundle, estado)
    mt5.ejecutar_colocadas()
    mt5.cerrar_por_sl(mt5.posiciones[0])     # entra y sale antes de que el bot lo vea
    tb.ciclo(bundle, estado)
    assert estado["pendiente"] is None       # consta como ejecutada en el historial


def test_sin_respuesta_y_sin_rastro_bloquea_hasta_el_plazo(mt5, bundle, monkeypatch):
    estado = nuevo_estado(mt5)
    mt5.respuestas, mt5.modo_incierto = [SIN_RESPUESTA], "nada"
    tb.ciclo(bundle, estado)
    assert estado["pendiente"]["orden"] is None
    monkeypatch.setattr(config, "PENDIENTE_SIN_RASTRO_S", None)   # sin plazo: hace falta resolverlo a mano
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1
    monkeypatch.setattr(config, "PENDIENTE_SIN_RASTRO_S", 0)      # plazo vencido sin rastro: se da por no enviada
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert estado.get("pendiente") is None and len(aperturas(mt5)) == 2


def test_pendiente_sobrevive_a_un_reinicio(mt5, bundle):
    estado = nuevo_estado(mt5)
    mt5.respuestas, mt5.modo_incierto = [PLACED], "coloca"
    tb.ciclo(bundle, estado)
    assert json.loads(config.ESTADO_PATH.read_text())["pendiente"]["orden"] == mt5.ordenes[0].ticket
    reiniciado = {**nuevo_estado(mt5), **tb.cargar_estado()}
    mt5.avanzar_vela()
    tb.ciclo(bundle, reiniciado)
    assert len(aperturas(mt5)) == 1


def test_orden_activa_ajena_a_la_pendiente_tambien_bloquea(mt5, bundle):
    from types import SimpleNamespace
    mt5.ordenes.append(SimpleNamespace(ticket=1, time_setup=0, state=tb.mt5.ORDER_STATE_PLACED,
                                       magic=config.MAGIC_NUMBER, symbol=config.SYMBOL, request=None))
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)
    assert aperturas(mt5) == [] and "colocada" in tb.motivo_sin_entradas(estado, mt5.cuenta)


def test_cierre_colocado_no_se_duplica(mt5, bundle):
    estado = nuevo_estado(mt5)
    mt5.abrir_posicion()
    mt5.avanzar_vela(config.MAX_BARS_EN_TRADE)
    mt5.respuestas, mt5.modo_incierto = [PLACED], "coloca"
    tb.ciclo(bundle, estado)
    tb.ciclo(bundle, estado)                 # el cierre sigue colocado: no se envía otro
    assert len(cierres(mt5)) == 1
    mt5.ejecutar_colocadas()
    tb.ciclo(bundle, estado)
    assert mt5.posiciones == [] and len(cierres(mt5)) == 1


# ---------------- cuenta ----------------
def test_cambio_de_cuenta_tras_reconectar_detiene(mt5, bundle):
    mt5.conectado, mt5.login_tras_reconectar = False, 2002
    with pytest.raises(SystemExit, match="cambió de cuenta"):
        tb.ciclo(bundle, nuevo_estado(mt5))
    assert mt5.enviadas == []


def test_reconexion_con_la_misma_cuenta_sigue(mt5, bundle):
    mt5.conectado = False
    tb.ciclo(bundle, nuevo_estado(mt5))
    assert len(aperturas(mt5)) == 1


def test_cuenta_real_detiene_antes_de_operar(mt5, bundle):
    mt5.cuenta.trade_mode = tb.ACCOUNT_TRADE_MODE_REAL
    with pytest.raises(SystemExit, match="REAL"):
        tb.ciclo(bundle, nuevo_estado(mt5))
    assert mt5.enviadas == []


def test_instancia_unica(config_aislado):
    codigo = (f"import sys; sys.path.insert(0, {str(Path(tb.__file__).parent)!r}); from pathlib import Path; "
              f"import trading_bot as tb; tb.InstanciaUnica(Path({str(config.LOCK_PATH)!r})).__enter__()")
    with tb.InstanciaUnica(config.LOCK_PATH):
        segunda = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=120)
    assert segunda.returncode != 0 and "otra instancia" in segunda.stderr
    tercera = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=120)
    assert tercera.returncode == 0  # liberado al salir


# ---------------- límites ----------------
def test_pausa_bloquea_entradas_pero_no_el_cierre(mt5, bundle):
    config.PAUSA_PATH.touch()
    estado = nuevo_estado(mt5)
    mt5.abrir_posicion()
    mt5.avanzar_vela(config.MAX_BARS_EN_TRADE)
    tb.ciclo(bundle, estado)
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert len(cierres(mt5)) == 1 and aperturas(mt5) == []


def test_limite_de_perdida_diaria(mt5, bundle):
    from types import SimpleNamespace
    mt5.deals.append(SimpleNamespace(ticket=1, position_id=1, entry=tb.mt5.DEAL_ENTRY_OUT, magic=config.MAGIC_NUMBER,
                                     symbol=config.SYMBOL, time=mt5.tick.time - 1, volume=1.0, price=1.1,
                                     profit=-250.0, swap=0.0, commission=0.0, fee=0.0, comment=""))
    mt5.cuenta.balance = mt5.cuenta.equity = 9_750.0
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)
    assert aperturas(mt5) == []
    assert "pérdida del día" in tb.motivo_sin_entradas(estado, mt5.cuenta)


def test_drawdown_bloquea_y_persiste(mt5, bundle):
    estado = nuevo_estado(mt5)
    tb.actualizar_equity_max(estado, 12_000.0)
    assert json.loads(config.ESTADO_PATH.read_text())["equity_max"] == 12_000.0
    estado = {**nuevo_estado(mt5), **tb.cargar_estado()}       # reinicio
    mt5.cuenta.equity = 10_000.0                                # -16.7% desde el máximo
    tb.ciclo(bundle, estado)
    assert aperturas(mt5) == []
    assert "drawdown" in tb.motivo_sin_entradas(estado, mt5.cuenta)


def test_calcular_volumen(mt5, monkeypatch):
    for v in (0.01, 0.03, 0.07, 0.29):
        monkeypatch.setattr(config, "VOLUMEN", v)
        assert tb.calcular_volumen(mt5.info, 0.0015) == v
    monkeypatch.setattr(config, "RIESGO_PCT", 0.005)
    assert tb.calcular_volumen(mt5.info, 0.0015) == 0.33   # 50 / (0.0015 / 0.00001 * 1)
    monkeypatch.setattr(config, "MAX_LOTES", 0.2)
    assert tb.calcular_volumen(mt5.info, 0.0015) == 0.2
    mt5.cuenta.equity = 100.0
    assert tb.calcular_volumen(mt5.info, 0.0015) is None


def test_modo_llenado(mt5):
    from types import SimpleNamespace
    assert tb.modo_llenado(SimpleNamespace(filling_mode=1)) == tb.mt5.ORDER_FILLING_FOK
    assert tb.modo_llenado(SimpleNamespace(filling_mode=2)) == tb.mt5.ORDER_FILLING_IOC
    assert tb.modo_llenado(SimpleNamespace(filling_mode=0)) == tb.mt5.ORDER_FILLING_RETURN


# ---------------- registro y conteo ----------------
def test_registro_de_ordenes_y_cierres(mt5, bundle):
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)
    mt5.avanzar_vela(config.MAX_BARS_EN_TRADE)
    tb.ciclo(bundle, estado)   # cierre por time stop
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)   # registra el cierre con su resultado
    eventos = [f["evento"] for f in operaciones()]
    assert eventos[0] == "Apertura LONG" and eventos[1].startswith("Cierre por time stop") and "cierre" in eventos
    cierre = next(f for f in operaciones() if f["evento"] == "cierre")
    assert float(cierre["beneficio"]) == -1.5 and float(cierre["swap"]) == -0.1


def test_velas_en_trade(mt5):
    from types import SimpleNamespace
    apertura = mt5.bar_time - 12 * config.TIMEFRAME_SEGUNDOS  # vela en la que se entró
    assert tb.velas_en_trade(SimpleNamespace(time=apertura + 3)) == 12
    assert tb.velas_en_trade(SimpleNamespace(time=apertura)) == 12          # mismo segundo de la apertura
    assert tb.velas_en_trade(SimpleNamespace(time=apertura * 1000)) == 0    # time_msc sin dividir: la trampa


# ---------------- carga del modelo ----------------
def _wf(t, positivas=(True,) * 5, trades=300):
    return {"ventanas": [{"trades": 10, "esperanza_R": 0.1 if p else -0.1} for p in positivas],
            "agregado": {"trades": trades, "esperanza_R": 0.1, "t_stat": t}}


def _bundle_guardado(robustez, holdout):
    b = {"model_long": ModeloFijo(0.9), "model_short": ModeloFijo(0.1), "umbral_long": 0.5, "umbral_short": 0.5,
         "features": FEATURES, "sl_mult": config.SL_MULTIPLICADOR_ATR, "tp_mult": config.TP_MULTIPLICADOR_ATR,
         "n_barras": config.MAX_BARS_EN_TRADE, "symbol": config.SYMBOL, "timeframe": config.TIMEFRAME,
         "informe": {"robustez": [{"semilla": s, "walk_forward": wf} for s, wf in robustez.items()],
                     "holdout": {"desde": config.HOLDOUT_DESDE, **holdout}}}
    joblib.dump(b, config.MODEL_PATH)


TODAS = lambda wf: {s: wf for s in config.SEMILLAS_ROBUSTEZ}  # noqa: E731
HOLDOUT_OK = {"trades": 60, "t_stat": 2.4}


def test_carga_con_ventaja_robusta_y_holdout():
    _bundle_guardado(TODAS(_wf(2.5)), HOLDOUT_OK)
    assert tb.cargar_modelo()["umbral_long"] == 0.5


@pytest.mark.parametrize("robustez, holdout, motivo", [
    ({42: _wf(2.5), 1: _wf(-0.2), 2: _wf(2.5)}, HOLDOUT_OK, "falla una semilla"),
    ({42: _wf(2.5)}, HOLDOUT_OK, "faltan semillas"),
    (None, HOLDOUT_OK, "t no significativo"),
    (None, {"trades": 0}, "holdout vacío"),
    (None, {"trades": 20, "t_stat": 3.0}, "holdout con pocos trades"),
    (None, {"trades": 60, "t_stat": 1.0}, "holdout no significativo"),
])
def test_bloqueos_del_modelo(robustez, holdout, motivo):
    if robustez is None:
        robustez = TODAS(_wf(1.0 if motivo == "t no significativo" else 2.5))
    _bundle_guardado(robustez, holdout)
    with pytest.raises(SystemExit, match="ventaja significativa"):
        tb.cargar_modelo()


def test_bloqueo_por_estabilidad_entre_ventanas():
    _bundle_guardado(TODAS(_wf(2.5, (True, True, False, False, False))), HOLDOUT_OK)
    with pytest.raises(SystemExit):
        tb.cargar_modelo()
    _bundle_guardado(TODAS(_wf(2.5, (True, True, True, False, False))), HOLDOUT_OK)
    assert tb.cargar_modelo() is not None


def test_modelo_antiguo_pide_reentrenar():
    _bundle_guardado(TODAS(_wf(2.5)), HOLDOUT_OK)
    b = joblib.load(config.MODEL_PATH)
    del b["informe"]["holdout"]
    joblib.dump(b, config.MODEL_PATH)
    with pytest.raises(SystemExit, match="reentrénalo"):
        tb.cargar_modelo()


def test_modelo_con_parametros_distintos(monkeypatch):
    _bundle_guardado(TODAS(_wf(2.5)), HOLDOUT_OK)
    monkeypatch.setattr(config, "MAX_BARS_EN_TRADE", config.MAX_BARS_EN_TRADE + 1)
    with pytest.raises(SystemExit, match="no coincide"):
        tb.cargar_modelo()


def test_bloqueo_desactivable_para_probar_en_demo(monkeypatch):
    _bundle_guardado(TODAS(_wf(-1.0)), {"trades": 0})
    monkeypatch.setattr(config, "EXIGIR_VENTAJA_SIGNIFICATIVA", False)
    assert tb.cargar_modelo() is not None
