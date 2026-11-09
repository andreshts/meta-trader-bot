"""Riesgo compartido de la cuenta: turno entre bots, máximo de equity común y topes de exposición."""
import json

import pytest

import config
import mt5_utils
import riesgo_cuenta as rc
import trading_bot as tb
from fake_mt5 import FakeMT5
from test_bot import ModeloFijo, aperturas, nuevo_estado
from features import FEATURES


@pytest.fixture
def rutas(tmp_path):
    return tmp_path / "riesgo_cuenta.json", tmp_path / "riesgo_cuenta.lock"


# ---------------- turno ----------------
def test_el_turno_excluye_a_un_segundo_bot_y_se_libera(rutas):
    _, bloqueo = rutas
    with rc.turno(bloqueo, espera_s=0):
        with pytest.raises(rc.CuentaOcupada):
            with rc.turno(bloqueo, espera_s=0):
                pass
    with rc.turno(bloqueo, espera_s=0):  # libre otra vez
        pass


def test_el_turno_se_libera_aunque_falle_lo_de_dentro(rutas):
    _, bloqueo = rutas
    with pytest.raises(RuntimeError):
        with rc.turno(bloqueo, espera_s=0):
            raise RuntimeError("fallo")
    with rc.turno(bloqueo, espera_s=0):
        pass


# ---------------- máximo de equity compartido ----------------
def test_el_maximo_solo_sube_y_lo_ven_todos(rutas):
    ruta, bloqueo = rutas
    assert rc.subir_equity_max(ruta, bloqueo, 10_000.0) == 10_000.0  # primer arranque: se crea
    assert rc.subir_equity_max(ruta, bloqueo, 10_500.0) == 10_500.0
    assert rc.subir_equity_max(ruta, bloqueo, 9_000.0) == 10_500.0   # otro bot, con la equity ya caída
    assert json.loads(ruta.read_text()) == {"equity_max": 10_500.0}
    assert list(ruta.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("contenido", ["", "{", "[]", '{"equity_max": -1}', '{"equity_max": "10000"}', '{"equity_max": true}'])
def test_un_maximo_corrupto_no_se_adopta_ni_se_pisa(rutas, contenido):
    ruta, bloqueo = rutas
    ruta.write_text(contenido)
    with pytest.raises(ValueError):
        rc.subir_equity_max(ruta, bloqueo, 10_000.0)
    assert ruta.read_text() == contenido


# ---------------- el bot con riesgo de cuenta ----------------
@pytest.fixture
def mt5(monkeypatch, rutas):
    fake = FakeMT5()
    monkeypatch.setattr(tb, "mt5", fake)
    monkeypatch.setattr(mt5_utils, "descargar_velas", fake.descargar_velas)
    monkeypatch.setattr(config, "ESTADO_CUENTA_PATH", rutas[0])
    monkeypatch.setattr(config, "BLOQUEO_CUENTA_PATH", rutas[1])
    monkeypatch.setattr(config, "ESPERA_TURNO_CUENTA_S", 0)
    monkeypatch.setattr(config, "MAGICS_CUENTA", {config.MAGIC_NUMBER, 202602})
    return fake


@pytest.fixture
def bundle():
    return {"features": FEATURES, "model_long": ModeloFijo(0.9), "model_short": ModeloFijo(0.1),
            "umbral_long": 0.5, "umbral_short": 0.5}


def test_el_drawdown_se_mide_contra_el_maximo_de_la_cuenta(mt5, bundle, rutas):
    """Un bot recién arrancado, con la equity ya caída, hereda el máximo que vieron los demás."""
    rutas[0].write_text(json.dumps({"equity_max": 12_000.0}))
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)  # equity 10.000: 16,7 % por debajo del máximo de la cuenta
    assert estado["equity_max"] == 12_000.0 and aperturas(mt5) == []
    assert "riesgo_cuenta.json" in tb.motivo_sin_entradas(estado, mt5.cuenta)


def test_borrar_el_maximo_de_la_cuenta_reinicia_el_drawdown(mt5, bundle, rutas):
    rutas[0].write_text(json.dumps({"equity_max": 12_000.0}))
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)
    rutas[0].unlink()
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert estado["equity_max"] == 10_000.0 and len(aperturas(mt5)) == 1


def test_un_maximo_de_cuenta_corrupto_bloquea_las_entradas_hasta_arreglarlo(mt5, bundle, rutas):
    rutas[0].write_text("{")
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)
    assert aperturas(mt5) == [] and "ilegible" in tb.motivo_sin_entradas(estado, mt5.cuenta)
    rutas[0].unlink()
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1 and "error_cuenta" not in estado


def test_si_el_turno_esta_ocupado_el_maximo_sigue_con_el_ultimo_conocido(mt5, rutas):
    estado = {**nuevo_estado(mt5), "equity_max": 10_000.0}
    rutas[0].write_text(json.dumps({"equity_max": 10_000.0}))
    with rc.turno(rutas[1], espera_s=0):
        tb.actualizar_equity_max(estado, 10_100.0)  # no puede escribir el compartido: sube el suyo y reintenta luego
    assert estado["equity_max"] == 10_100.0 and "error_cuenta" not in estado
    tb.actualizar_equity_max(estado, 10_100.0)
    assert json.loads(rutas[0].read_text()) == {"equity_max": 10_100.0}


def otra_posicion(fake, magic=202602, sl=1.09, simbolo="XAUUSDm"):
    pos = fake.abrir_posicion()
    pos.magic, pos.sl, pos.symbol = magic, sl, simbolo
    return pos


def test_el_tope_de_posiciones_cuenta_las_de_todos_los_activos(mt5, bundle, monkeypatch):
    monkeypatch.setattr(config, "MAX_POSICIONES_CUENTA", 2)
    otra_posicion(mt5)
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1  # una de otro activo: cabe otra
    otra_posicion(mt5, simbolo="US500m")
    mt5.posiciones = [p for p in mt5.posiciones if p.magic != config.MAGIC_NUMBER]  # la propia se cerró
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1  # dos de otros activos: no cabe


def test_las_posiciones_ajenas_al_bot_no_cuentan(mt5, bundle, monkeypatch):
    monkeypatch.setattr(config, "MAX_POSICIONES_CUENTA", 1)
    otra_posicion(mt5, magic=999)
    tb.ciclo(bundle, nuevo_estado(mt5))
    assert len(aperturas(mt5)) == 1


def test_el_riesgo_abierto_suma_lo_ya_abierto_y_la_entrada_nueva(mt5, bundle, monkeypatch):
    # La posición del otro activo arriesga (1.10002 - 1.09) / 0.00001 * 1 * 0.01 = 10,02.
    otra_posicion(mt5)
    assert tb.exposicion_de_la_cuenta() == (1, pytest.approx(10.02))
    assert tb.motivo_exposicion(tb.mt5.ORDER_TYPE_BUY, 1.1, 1.0995, 0.01) is None  # sin tope no se limita
    monkeypatch.setattr(config, "MAX_RIESGO_ABIERTO_PCT", 0.00105)  # 10,50: cabe lo abierto, pero no con 0,50 más
    assert "10.52" in tb.motivo_exposicion(tb.mt5.ORDER_TYPE_BUY, 1.1, 1.0995, 0.01)
    assert tb.motivo_exposicion(tb.mt5.ORDER_TYPE_SELL, 1.1, 1.1004, 0.01) is None  # 0,40 más: cabe
    estado = nuevo_estado(mt5)
    tb.ciclo(bundle, estado)
    assert aperturas(mt5) == [] and estado.get("pendiente") is None
    monkeypatch.setattr(config, "MAX_RIESGO_ABIERTO_PCT", 0.01)   # 100,00: cabe
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1


def test_una_posicion_sin_stop_agota_el_riesgo_abierto(mt5, bundle, monkeypatch):
    monkeypatch.setattr(config, "MAX_RIESGO_ABIERTO_PCT", 0.5)
    otra_posicion(mt5, sl=0.0)
    tb.ciclo(bundle, nuevo_estado(mt5))
    assert aperturas(mt5) == []


def test_si_otro_bot_tiene_el_turno_no_se_entra_en_esta_vela(mt5, bundle, rutas, monkeypatch):
    monkeypatch.setattr(config, "MAX_POSICIONES_CUENTA", 3)
    estado = nuevo_estado(mt5)
    tb.actualizar_equity_max(estado, mt5.cuenta.equity)
    with rc.turno(rutas[1], espera_s=0):
        tb.ciclo(bundle, estado)
    assert aperturas(mt5) == [] and estado.get("pendiente") is None
    mt5.avanzar_vela()
    tb.ciclo(bundle, estado)
    assert len(aperturas(mt5)) == 1


def test_sin_configurar_la_cuenta_el_bot_se_comporta_como_antes(mt5, bundle, monkeypatch):
    for nombre in ("ESTADO_CUENTA_PATH", "BLOQUEO_CUENTA_PATH", "MAGICS_CUENTA"):
        monkeypatch.setattr(config, nombre, None)
    otra_posicion(mt5)
    tb.ciclo(bundle, nuevo_estado(mt5))
    assert len(aperturas(mt5)) == 1 and not config.ESTADO_PATH.with_name("riesgo_cuenta.json").exists()


def test_el_lote_por_riesgo_usa_el_valor_del_tick_al_perder(mt5, monkeypatch):
    monkeypatch.setattr(config, "RIESGO_PCT", 0.005)
    assert tb.calcular_volumen(mt5.info, 0.0015) == 0.33
    mt5.info.trade_tick_value_loss = 1.1  # p. ej. un índice en euros en una cuenta en dólares
    assert tb.calcular_volumen(mt5.info, 0.0015) == 0.30
