"""El panel refleja lo que hace el bot real y sus fallos nunca lo interrumpen."""
import logging
import math

import numpy as np
import pytest

import config
import demo_exness_bot as lanzador
import mt5_utils
import panel_mt5
import trading_bot as tb
from fake_mt5 import FakeMT5
from features import FEATURES


class ModeloFijo:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.array([[1 - self.p, self.p]] * len(x))


def bundle(p_long, p_short):
    return {"features": FEATURES, "model_long": ModeloFijo(p_long), "model_short": ModeloFijo(p_short),
            "umbral_long": 0.33, "umbral_short": 0.38}


def leer(panel):
    return dict(linea.split("=", 1) for linea in panel.ruta.read_text(encoding="utf-8").splitlines())


@pytest.fixture
def mt5(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setattr(tb, "mt5", fake)
    monkeypatch.setattr(mt5_utils, "descargar_velas", fake.descargar_velas)
    return fake


@pytest.fixture
def panel(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=tb.log.name)
    p = panel_mt5.PanelMT5("EURUSDm", 480569059, config.MAGIC_NUMBER, fin=1793994300, ruta=tmp_path / "panel.txt")
    p.fijar_umbrales(0.33, 0.38)
    tb.log.addHandler(p)
    yield p
    tb.log.removeHandler(p)


def ciclo(mt5, modelo):
    tb.ciclo(modelo, {"login": mt5.cuenta.login, "last_bar_time": None, "deals_vistos": set()})


def test_sin_senal_publica_probabilidades_y_motivo(mt5, panel):
    ciclo(mt5, bundle(0.311, 0.245))
    datos = leer(panel)
    assert (datos["p_long"], datos["p_short"]) == ("0.311", "0.245")
    assert (datos["umbral_long"], datos["umbral_short"]) == ("0.33", "0.38")
    assert datos["nota"] == "Sin señal: ninguna probabilidad llega a su umbral"
    assert datos["estado"] == "en_marcha"
    assert mt5.enviadas == []


def test_una_apertura_del_bot_aparece_como_ultima_decision(mt5, panel):
    ciclo(mt5, bundle(0.9, 0.1))
    assert len(mt5.enviadas) == 1
    assert leer(panel)["nota"].startswith("Apertura LONG ejecutada")


def test_bajar_el_umbral_abre_lo_que_antes_no_abria(mt5, panel):
    modelo = bundle(0.311, 0.245)
    ciclo(mt5, modelo)
    assert mt5.enviadas == []

    lanzador.aplicar_umbral(modelo, 0.30)
    panel.fijar_umbrales(modelo["umbral_long"], modelo["umbral_short"])
    ciclo(mt5, modelo)
    assert len(mt5.enviadas) == 1
    assert leer(panel)["nota"].startswith("Apertura LONG ejecutada")
    assert (leer(panel)["umbral_long"], leer(panel)["umbral_short"]) == ("0.30", "0.30")


def test_el_criterio_de_decision_coincide_con_el_del_bot(mt5, panel):
    ciclo(mt5, bundle(0.9, 0.9))
    assert leer(panel)["nota"] == "Señales opuestas: no se opera"
    assert mt5.enviadas == []


def test_lado_desactivado_nunca_da_senal(tmp_path):
    p = panel_mt5.PanelMT5("EURUSDm", 1, 1, fin=0, ruta=tmp_path / "panel.txt")
    p.fijar_umbrales(None, 0.38)
    assert leer(p)["umbral_long"] == ""
    assert p._decision(0.9, 0.1).startswith("Sin señal")


def test_avisos_y_errores_llegan_al_panel_en_una_sola_linea(panel):
    tb.log.warning("Spread de %s puntos\nno se opera", 40)
    datos = leer(panel)
    assert datos["nota"] == "Spread de 40 puntos no se opera"
    assert datos["nota_nivel"] == "WARNING"


def test_el_latido_limita_las_escrituras(panel):
    panel.ruta.unlink()
    panel.latido()
    assert not panel.ruta.exists()  # acaba de publicar al fijar los umbrales
    panel._ultimo_latido = -math.inf
    panel.latido()
    assert panel.ruta.exists()


def test_detener_lo_refleja_el_panel(panel):
    panel.detener()
    assert leer(panel)["estado"] == "detenido"


def test_el_archivo_usa_lo_que_espera_el_indicador(panel):
    tb.log.info("Señal con acentos")
    crudo = panel.ruta.read_bytes()
    assert b"\r\n" in crudo and "Señal".encode("utf-8") in crudo
    assert not crudo.startswith(b"\xef\xbb\xbf")


def test_un_fallo_al_escribir_nunca_interrumpe_al_bot(tmp_path, mt5, caplog):
    caplog.set_level(logging.INFO, logger=tb.log.name)
    (tmp_path / "ocupado").write_text("no es una carpeta")
    p = panel_mt5.PanelMT5("EURUSDm", 1, 1, fin=0, ruta=tmp_path / "ocupado" / "panel.txt")
    tb.log.addHandler(p)
    try:
        p.fijar_umbrales(0.33, 0.38)
        p.latido()
        p.detener()
        ciclo(mt5, bundle(0.9, 0.1))
    finally:
        tb.log.removeHandler(p)
    assert len(mt5.enviadas) == 1  # la orden se envió igual


def test_el_latido_publica_la_cuenta_solo_cuando_toca_y_aguanta_sus_fallos(panel):
    from unittest.mock import Mock

    cuenta = Mock(return_value={"balance": "9994.98", "equity": "9990.00", "moneda": "USD"})
    panel._ultimo_latido = -math.inf
    panel.latido(cuenta)
    texto = panel.ruta.read_text(encoding="utf-8")
    assert "balance=9994.98" in texto and "equity=9990.00" in texto and "cuenta_leida=" in texto
    panel.latido(cuenta)  # demasiado pronto: ni publica ni vuelve a leer la cuenta
    cuenta.assert_called_once()
    for fallo in (Mock(return_value=None), Mock(side_effect=RuntimeError("terminal caída"))):
        panel._ultimo_latido = -math.inf
        panel.latido(fallo)
        assert "balance=9994.98" in panel.ruta.read_text(encoding="utf-8")  # el latido sale con lo último leído
