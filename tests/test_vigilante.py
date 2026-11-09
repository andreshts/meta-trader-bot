"""El vigilante: relanza lo que se cae, no toca lo que funciona ni lo que se detuvo a propósito, y sabe rendirse."""
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

import riesgo_cuenta
import vigilante as vg
from activos import Activo

AHORA = 1_800_000_000.0
ORDEN = ["python.exe", "demo_exness_bot.py", "--simbolo", "XAUUSDm", "--volumen", "0.01"]


def crear_activo(tmp_path, simbolo="XAUUSDm") -> Activo:
    return Activo(simbolo, simbolo, 202602, 0.01, 750, simbolo, tmp_path / f"{simbolo}.pkl", tmp_path / simbolo,
                  tmp_path / f"{simbolo}.lock", tmp_path / f"panel_{simbolo}.txt")


def publicar(activo, momento, estado="en_marcha"):
    activo.panel.write_text(f"estado={estado}\nactualizado={int(momento)}\n", encoding="utf-8")


class Escena:
    def __init__(self, tmp_path):
        self.ahora, self.ruta, self.vivos = AHORA, tmp_path / "vigilante.json", set()
        self.oro = crear_activo(tmp_path)
        self.lanzar, self.avisos = Mock(), []
        self.vigilante = vg.Vigilante({self.oro.simbolo: self.oro}, self.ruta, self.lanzar, self.avisos.append,
                                      lambda: self.ahora, lambda bloqueo: bloqueo in self.vivos)

    def pasan(self, segundos):
        self.ahora += segundos
        self.vigilante.revisar()


@pytest.fixture
def escena(tmp_path):
    return Escena(tmp_path)


# ---------------- la lista de bots que deben estar en marcha ----------------
def test_apuntar_y_borrar(tmp_path):
    ruta = tmp_path / "v.json"
    assert vg.leer_deseados(ruta) == {}
    vg.desear(ruta, "XAUUSDm", ORDEN, None)
    vg.desear(ruta, "DE30m", ORDEN, AHORA)
    vg.dejar_de_desear(ruta, "XAUUSDm")
    vg.dejar_de_desear(ruta, "no_estaba")
    assert vg.leer_deseados(ruta) == {"DE30m": {"orden": ORDEN, "fin": AHORA}}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("contenido", ["", "{", "[]", '{"XAUUSDm": 5}', '{"XAUUSDm": {"orden": "x"}}'])
def test_una_lista_ilegible_se_trata_como_vacia(tmp_path, contenido):
    ruta = tmp_path / "v.json"
    ruta.write_text(contenido)
    assert vg.leer_deseados(ruta) == {}


def test_una_sesion_con_duracion_se_relanza_con_lo_que_le_queda():
    entrada = {"orden": [*ORDEN, "--horas", "5"], "fin": AHORA + 2 * 3600}
    assert vg.orden_para_relanzar(entrada, AHORA)[-2:] == ["--horas", "2.0000"]
    assert entrada["orden"][-1] == "5"  # no modifica lo apuntado
    assert vg.orden_para_relanzar({"orden": ORDEN, "fin": None}, AHORA) == ORDEN
    assert vg.horas_de(ORDEN) is None and vg.horas_de([*ORDEN, "--horas", "5"]) == 5.0


def test_el_bloqueo_tomado_delata_un_proceso_vivo(tmp_path):
    bloqueo = tmp_path / "bot.lock"
    assert not vg.tiene_el_bloqueo(bloqueo)
    with riesgo_cuenta.turno(bloqueo, espera_s=0):
        assert vg.tiene_el_bloqueo(bloqueo)
    assert not vg.tiene_el_bloqueo(bloqueo)


# ---------------- revisar ----------------
def test_un_bot_con_latido_no_se_toca(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    publicar(escena.oro, AHORA - 5)
    escena.pasan(0)
    escena.lanzar.assert_not_called()
    assert escena.avisos == []


def test_un_bot_caido_se_relanza_con_sus_ajustes_y_se_avisa(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    publicar(escena.oro, AHORA - 300)
    escena.oro.carpeta.mkdir()
    escena.oro.salida.write_text("Traceback...\nMemoryError: sin memoria\n", encoding="utf-8")
    escena.pasan(0)
    escena.lanzar.assert_called_once_with(escena.oro, ORDEN)
    assert "se relanzó (intento 1 de 5)" in escena.avisos[0] and "MemoryError: sin memoria" in escena.avisos[0]


def test_tras_reiniciar_el_equipo_se_lanza_lo_apuntado_aunque_dijera_detenido(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    publicar(escena.oro, AHORA - 5, estado="detenido")  # el bot salió por su cuenta con un error
    escena.pasan(0)
    escena.lanzar.assert_called_once()


def test_lo_que_no_esta_apuntado_no_se_lanza(escena):
    publicar(escena.oro, AHORA - 300)  # se detuvo desde la ventana: ya no está en la lista
    escena.pasan(0)
    escena.lanzar.assert_not_called()


def test_recien_relanzado_se_le_da_tiempo_a_arrancar(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    escena.pasan(0)
    escena.pasan(vg.ARRANQUE_S - 1)  # todavía sin latido, pero puede estar cargando el modelo
    assert escena.lanzar.call_count == 1
    escena.pasan(2)
    assert escena.lanzar.call_count == 2


def test_un_proceso_vivo_sin_latido_se_avisa_una_vez_y_no_se_relanza(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    publicar(escena.oro, AHORA - 300)
    escena.vivos.add(escena.oro.bloqueo)
    escena.pasan(0)
    escena.pasan(10)
    escena.lanzar.assert_not_called()
    assert len(escena.avisos) == 1 and "sigue vivo pero no da señales desde hace 300 s" in escena.avisos[0]
    publicar(escena.oro, escena.ahora + 10)
    escena.pasan(10)
    assert "vuelve a responder" in escena.avisos[1]


def test_si_cae_una_y_otra_vez_se_rinde_avisa_y_lo_borra_de_la_lista(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    for _ in range(vg.MAX_REINTENTOS + 1):
        escena.pasan(vg.ARRANQUE_S * 2 ** vg.MAX_REINTENTOS)
    assert escena.lanzar.call_count == vg.MAX_REINTENTOS
    assert "no se relanza más" in escena.avisos[-1] and vg.leer_deseados(escena.ruta) == {}
    escena.pasan(10_000)
    assert escena.lanzar.call_count == vg.MAX_REINTENTOS


def test_un_relanzamiento_que_aguanta_pone_los_reintentos_a_cero(escena):
    vg.desear(escena.ruta, "XAUUSDm", ORDEN, None)
    escena.pasan(0)                       # cae y se relanza: intento 1
    publicar(escena.oro, escena.ahora + vg.ESTABLE_S)
    escena.pasan(vg.ESTABLE_S)            # sigue en marcha: se olvida el intento
    escena.pasan(vg.MAX_LATIDO_S + 1)     # vuelve a caer mucho después
    assert "intento 1 de" in escena.avisos[-1] and escena.lanzar.call_count == 2


def test_una_sesion_terminada_deja_de_vigilarse_sin_relanzar(escena):
    vg.desear(escena.ruta, "XAUUSDm", [*ORDEN, "--horas", "5"], AHORA + 100)
    publicar(escena.oro, AHORA)
    escena.pasan(101)
    escena.lanzar.assert_not_called()
    assert vg.leer_deseados(escena.ruta) == {} and escena.avisos == []


def test_una_sesion_a_medias_se_relanza_solo_por_lo_que_queda(escena):
    vg.desear(escena.ruta, "XAUUSDm", [*ORDEN, "--horas", "5"], AHORA + 3600)
    escena.pasan(0)
    assert escena.lanzar.call_args.args[1][-2:] == ["--horas", "1.0000"]


def test_un_activo_desconocido_en_la_lista_se_ignora(escena):
    vg.desear(escena.ruta, "BTCUSDm", ORDEN, None)
    escena.pasan(0)
    escena.lanzar.assert_not_called()


def test_solo_puede_haber_un_vigilante(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, "RUTA", tmp_path / "vigilante.json")
    monkeypatch.setattr(vg, "LOGS", tmp_path)
    monkeypatch.setattr(vg.logging, "basicConfig", lambda **_k: None)
    monkeypatch.setattr(vg.logging, "FileHandler", Mock())
    with riesgo_cuenta.turno(tmp_path / "vigilante.lock", espera_s=0):
        with pytest.raises(SystemExit, match="Ya hay un vigilante"):
            vg.main()


def test_lanzar_arranca_el_proceso_y_guarda_su_salida(tmp_path):
    activo = crear_activo(tmp_path)
    proceso = vg.lanzar(activo, [sys.executable, "-c", "print('hola desde el bot')"])
    assert isinstance(proceso, subprocess.Popen) and proceso.wait(timeout=30) == 0
    assert vg.ultima_linea(activo.salida) == "hola desde el bot"
    assert json.loads(json.dumps(vg.leer_panel(activo.panel))) == {}
