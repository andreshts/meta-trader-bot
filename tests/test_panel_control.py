"""La ventana de configuración: qué arranca, con qué argumentos, y cómo describe a cada bot."""
import json
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import demo_exness_bot
import panel_control as pc
from activos import Activo

COMUNES = {"horas": "5", "perdida_diaria": "2", "drawdown": "10"}
AJUSTES = {"operar": True, "lote": "0.02", "max_spread": "700", "umbral_long": "0,31", "umbral_short": "0.29"}


def crear_activo(tmp_path, simbolo="XAUUSDm", con_modelo=True) -> Activo:
    modelo = tmp_path / f"{simbolo}.pkl"
    if con_modelo:
        modelo.touch()
        modelo.with_suffix(".json").write_text(json.dumps({"umbral_long": 0.31, "umbral_short": 0.29}), encoding="utf-8")
    return Activo(simbolo, simbolo, 202602, 0.01, 750, simbolo, modelo, tmp_path / simbolo, tmp_path / f"{simbolo}.lock",
                  tmp_path / f"panel_{simbolo}.txt")


def publicar(activo, ahora, **datos):
    activo.panel.write_text("".join(f"{c}={v}\n" for c, v in {"estado": "en_marcha", "actualizado": int(ahora), "fin": "18:04", **datos}.items()),
                            encoding="utf-8")


@pytest.fixture
def activo(tmp_path):
    return crear_activo(tmp_path)


def test_el_comando_normaliza_y_lleva_todos_los_ajustes(activo):
    orden = pc.comando(activo, AJUSTES, COMUNES)
    assert orden[1] == str(pc.LANZADOR)
    argumentos = dict(zip(orden[2::2], orden[3::2]))
    assert argumentos == {"--simbolo": "XAUUSDm", "--horas": "5", "--volumen": "0.02", "--max-spread": "700",
                          "--umbral-long": "0.31", "--umbral-short": "0.29", "--perdida-diaria": "2", "--drawdown": "10"}


def test_lo_que_arma_la_ventana_lo_acepta_el_lanzador(activo, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", *pc.comando(activo, AJUSTES, COMUNES)[2:]])
    args = demo_exness_bot.leer_argumentos()
    assert (args.simbolo, args.volumen, args.umbral_long, args.perdida_diaria) == ("XAUUSDm", 0.02, 0.31, 0.02)


def test_un_umbral_vacio_apaga_ese_lado(activo):
    orden = pc.comando(activo, {**AJUSTES, "umbral_short": "  "}, COMUNES)
    assert orden[orden.index("--umbral-short") + 1] == "off"


@pytest.mark.parametrize("cambios,comunes,mensaje", [
    ({"lote": "0.001"}, {}, "lote"), ({"lote": "2"}, {}, "lote"), ({"lote": "abc"}, {}, "lote"),
    ({"max_spread": "0"}, {}, "spread"), ({"umbral_long": "1.5"}, {}, "compra"), ({"umbral_short": "nan"}, {}, "venta"),
    ({"umbral_long": "", "umbral_short": ""}, {}, "al menos un umbral"),
    ({}, {"horas": "0"}, "duración"), ({}, {"perdida_diaria": "200"}, "pérdida"), ({}, {"drawdown": ""}, "drawdown"),
])
def test_los_ajustes_no_validos_se_explican(activo, cambios, comunes, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        pc.comando(activo, {**AJUSTES, **cambios}, {**COMUNES, **comunes})


def test_los_umbrales_iniciales_salen_del_entrenamiento(tmp_path):
    con, sin = crear_activo(tmp_path), crear_activo(tmp_path, "DE30m", con_modelo=False)
    assert (pc.ajustes_iniciales(con)["umbral_long"], pc.ajustes_iniciales(con)["umbral_short"]) == ("0.31", "0.29")
    assert pc.ajustes_iniciales(sin)["umbral_long"] == ""


AHORA = 1_800_000_000
VIVO, TERMINADO_MAL, TERMINADO_BIEN = (SimpleNamespace(poll=lambda r=r: r, returncode=r) for r in (None, 1, 0))


@pytest.mark.parametrize("panel,proceso,texto,en_marcha", [
    ({"estado": "en_marcha", "actualizado": AHORA - 3, "fin": "18:04"}, None, "En marcha hasta las 18:04", True),
    ({"estado": "en_marcha", "actualizado": AHORA - 300, "fin": "18:04"}, VIVO, "Arrancando…", True),
    ({}, VIVO, "Arrancando…", True),
    ({}, TERMINADO_MAL, "No pudo seguir: motivo", False),
    ({"estado": "en_marcha", "actualizado": AHORA - 300}, None, "Sin respuesta", False),
    ({"estado": "detenido", "actualizado": AHORA}, TERMINADO_BIEN, "Detenido", False),
    ({}, None, "Detenido", False),
])
def test_estado_del_bot(panel, proceso, texto, en_marcha):
    estado = pc.estado_del_bot(panel, AHORA, proceso, "motivo")
    assert (estado.texto, estado.en_marcha) == (texto, en_marcha)


def test_leer_panel_sin_archivo(tmp_path):
    assert pc.leer_panel(tmp_path / "no_existe.txt") == {}


@pytest.fixture
def raiz():
    try:
        ventana = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"sin pantalla: {e}")
    ventana.withdraw()
    yield ventana
    try:
        ventana.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def abrir(raiz, tmp_path, monkeypatch):
    errores = Mock()
    monkeypatch.setattr(pc.messagebox, "showerror", errores)
    monkeypatch.setattr(pc.messagebox, "showinfo", errores)
    lanzar = Mock(return_value=SimpleNamespace(poll=lambda: None, returncode=None))

    def crear(*activos_):
        ventana = pc.Ventana(raiz, {a.simbolo: a for a in activos_}, lanzar, tmp_path / "ajustes.json")
        return ventana, lanzar, errores
    return crear


def marcar(ventana, n=0, valor=True):
    ventana.filas[n].variables["operar"].set(valor)


def test_iniciar_lanza_solo_lo_marcado(abrir, tmp_path):
    oro, dax = crear_activo(tmp_path), crear_activo(tmp_path, "DE30m")
    ventana, lanzar, errores = abrir(oro, dax)
    marcar(ventana, 0)
    marcar(ventana, 1, False)
    ventana.iniciar()
    lanzar.assert_called_once()
    assert lanzar.call_args.args[0] is oro
    errores.assert_not_called()
    assert ventana.estados["XAUUSDm"].texto == "Arrancando…"
    assert str(ventana.filas[0].entradas[0].cget("state")) == "disabled"


def test_no_se_lanza_lo_que_ya_esta_en_marcha(abrir, activo):
    publicar(activo, pc.time.time())
    ventana, lanzar, _ = abrir(activo)
    marcar(ventana)
    ventana.iniciar()
    lanzar.assert_not_called()
    assert ventana.estados["XAUUSDm"].en_marcha


def test_un_ajuste_no_valido_no_lanza_ninguno(abrir, tmp_path):
    bien, mal = crear_activo(tmp_path), crear_activo(tmp_path, "DE30m")
    ventana, lanzar, errores = abrir(bien, mal)
    marcar(ventana, 0)
    marcar(ventana, 1)
    ventana.filas[1].variables["lote"].set("0")
    ventana.iniciar()
    lanzar.assert_not_called()
    assert "lote" in errores.call_args.args[1].lower()


def test_sin_modelo_no_se_puede_iniciar(abrir, tmp_path):
    ventana, lanzar, _ = abrir(crear_activo(tmp_path, "DE30m", con_modelo=False))
    assert ventana.estados["DE30m"].texto == "Falta entrenar su modelo"
    assert str(ventana.filas[0].casilla.cget("state")) == "disabled"
    marcar(ventana)
    ventana.iniciar()
    lanzar.assert_not_called()


def test_detener_deja_la_orden_para_ese_bot(abrir, activo):
    publicar(activo, pc.time.time())
    ventana, _, _ = abrir(activo)
    ventana.detener_todos()
    assert activo.detener.exists()
    assert ventana.estados["XAUUSDm"].texto == "Deteniendo…"
    assert str(ventana.filas[0].detener.cget("state")) == "disabled"


def test_publica_probabilidades_y_novedad(abrir, activo):
    publicar(activo, pc.time.time(), p_long="0.268", p_short="0.271", nota_hora="14:30:01", nota="Sin señal")
    ventana, _, _ = abrir(activo)
    assert ventana.filas[0].probabilidades.cget("text") == "0.268 / 0.271"
    assert ventana.filas[0].nota.cget("text") == "14:30:01  Sin señal"


def test_los_ajustes_se_recuerdan(abrir, activo, raiz, tmp_path):
    ventana, _, _ = abrir(activo)
    ventana.filas[0].variables["lote"].set("0.05")
    ventana.comunes["horas"].set("8")
    ventana.guardar()
    otra = pc.Ventana(raiz, {activo.simbolo: activo}, Mock(), tmp_path / "ajustes.json")
    assert otra.filas[0].variables["lote"].get() == "0.05"
    assert otra.comunes["horas"].get() == "8"


def test_un_archivo_de_ajustes_roto_no_impide_abrir(abrir, activo, tmp_path):
    (tmp_path / "ajustes.json").write_text("{roto", encoding="utf-8")
    ventana, _, _ = abrir(activo)
    assert ventana.comunes["horas"].get() == "5"
