"""La ventana de configuración: qué arranca, con qué argumentos, y cómo describe a cada bot."""
import json
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import demo_exness_bot
import panel_control as pc
from activos import Activo
from operaciones import Instantanea, Operacion

COMUNES = {"horas": "5", "perdida_diaria": "2", "drawdown": "10", "riesgo_abierto": "2,5", "max_posiciones": "3"}
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
                          "--umbral-long": "0.31", "--umbral-short": "0.29", "--perdida-diaria": "2", "--drawdown": "10",
                          "--riesgo-abierto": "2.5", "--max-posiciones": "3", "--sin-entradas": "23:30-01:15"}


def test_lo_que_arma_la_ventana_lo_acepta_el_lanzador(activo, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", *pc.comando(activo, AJUSTES, COMUNES)[2:]])
    args = demo_exness_bot.leer_argumentos()
    assert (args.simbolo, args.volumen, args.umbral_long, args.perdida_diaria) == ("XAUUSDm", 0.02, 0.31, 0.02)
    assert (args.riesgo_abierto, args.max_posiciones) == (0.025, 3)


def test_un_tope_de_cuenta_vacio_no_se_pasa_al_lanzador(activo):
    orden = pc.comando(activo, AJUSTES, {**COMUNES, "riesgo_abierto": " ", "max_posiciones": ""})
    assert "--riesgo-abierto" not in orden and "--max-posiciones" not in orden
    assert "--max-posiciones" not in pc.comando(activo, AJUSTES, {"horas": "5", "perdida_diaria": "2", "drawdown": "10"})


def test_con_riesgo_por_operacion_el_lote_lo_calcula_el_bot(activo, monkeypatch):
    orden = pc.comando(activo, {**AJUSTES, "lote": "no se usa"}, {**COMUNES, "riesgo": "0,5"})
    assert "--volumen" not in orden and orden[orden.index("--riesgo") + 1] == "0.5"
    monkeypatch.setattr("sys.argv", ["x", *orden[2:]])
    args = demo_exness_bot.leer_argumentos()
    assert (args.riesgo, args.volumen) == (0.005, None)


def test_la_franja_sin_entradas_llega_al_lanzador(activo, monkeypatch):
    orden = pc.comando(activo, AJUSTES, {**COMUNES, "sin_entradas": " 22:45 - 02:00 "})
    assert orden[orden.index("--sin-entradas") + 1] == "22:45-02:00"
    monkeypatch.setattr("sys.argv", ["x", *orden[2:]])
    assert demo_exness_bot.leer_argumentos().sin_entradas == ("22:45", "02:00")
    vacia = pc.comando(activo, AJUSTES, {**COMUNES, "sin_entradas": "  "})
    monkeypatch.setattr("sys.argv", ["x", *vacia[2:]])
    assert demo_exness_bot.leer_argumentos().sin_entradas is None


def test_un_umbral_vacio_apaga_ese_lado(activo):
    orden = pc.comando(activo, {**AJUSTES, "umbral_short": "  "}, COMUNES)
    assert orden[orden.index("--umbral-short") + 1] == "off"


@pytest.mark.parametrize("cambios,comunes,mensaje", [
    ({"lote": "0.001"}, {}, "lote"), ({"lote": "2"}, {}, "lote"), ({"lote": "abc"}, {}, "lote"),
    ({"max_spread": "0"}, {}, "spread"), ({"umbral_long": "1.5"}, {}, "compra"), ({"umbral_short": "nan"}, {}, "venta"),
    ({"umbral_long": "", "umbral_short": ""}, {}, "al menos un umbral"),
    ({}, {"horas": "0"}, "duración"), ({}, {"perdida_diaria": "200"}, "pérdida"), ({}, {"drawdown": ""}, "drawdown"),
    ({}, {"max_posiciones": "2.5"}, "entero"), ({}, {"max_posiciones": "0"}, "posiciones"), ({}, {"riesgo_abierto": "x"}, "riesgo abierto"),
    ({}, {"riesgo": "0"}, "riesgo por operación"), ({}, {"sin_entradas": "23:30"}, "franja"), ({}, {"sin_entradas": "25:00-01:00"}, "franja"),
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
    ({"estado": "en_marcha", "actualizado": AHORA - 3, "fin": "--:--"}, None, "En marcha, sin límite", True),
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
        ventana = pc.crear_raiz()
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


def operacion(ticket=1, abierta=False, resultado=5.0, apertura=1_794_000_000):
    return Operacion(ticket, True, 0.01, apertura, 4000.123, 3990.0, 4020.0, None if abierta else apertura + 900, 4010.5,
                     "Abierta" if abierta else "Take profit", resultado)


def foto(*ops, digitos=3):
    return Instantanea(list(ops), digitos, "USD")


def test_fila_de_operacion_da_formato_y_color():
    valores, etiqueta = pc.fila_de_operacion(operacion(resultado=-1.5), 3)
    assert (valores[1], valores[2], valores[3], valores[8], valores[9], etiqueta) == ("Compra", "0.01", "4000.123", "Take profit", "-1.50", "perdida")
    assert pc.fila_de_operacion(operacion(resultado=2.0), 3)[1] == "ganada"
    abierta, etiqueta = pc.fila_de_operacion(operacion(abierta=True), 3)
    assert (abierta[6], etiqueta) == ("", "abierta")


def test_el_resumen_cuenta_cerradas_y_abierta():
    texto = pc.texto_resumen(foto(operacion(1, resultado=5.0), operacion(2, resultado=-2.0), operacion(3, abierta=True, resultado=1.25)))
    assert texto == "2 cerradas · 1 ganadas · 1 perdidas · resultado +3.00 USD · abierta ahora +1.25 USD"
    assert pc.texto_resumen(foto()) == "Todavía no hay operaciones de este activo"


@pytest.fixture
def abrir_con_lector(raiz, tmp_path, monkeypatch):
    def crear(activo_, datos):
        lecturas = []

        def leer(activos_):
            lecturas.append([a.simbolo for a in activos_])
            return {a.simbolo: datos["valor"] for a in activos_}
        return pc.Ventana(raiz, {activo_.simbolo: activo_}, Mock(), tmp_path / "ajustes.json", leer), lecturas
    monkeypatch.setattr(pc, "INTERVALO_VISTAS_S", 0)
    return crear


def esperar(raiz, condicion, segundos=5):
    limite = pc.time.time() + segundos
    while not condicion() and pc.time.time() < limite:
        raiz.update()
        pc.time.sleep(0.01)
    assert condicion()


def test_ver_muestra_las_operaciones_y_las_actualiza(abrir_con_lector, activo, raiz):
    datos = {"valor": foto(operacion(1, abierta=True, resultado=1.0))}
    ventana, _ = abrir_con_lector(activo, datos)
    ventana.filas[0].ver.invoke()
    vista = ventana.vistas["XAUUSDm"]
    esperar(raiz, lambda: vista.tabla.exists("1"))
    assert vista.tabla.item("1", "tags") == ("abierta",)

    datos["valor"] = foto(operacion(2, abierta=True, apertura=1_794_001_000), operacion(1, resultado=5.0))
    esperar(raiz, lambda: vista.tabla.exists("2"))
    esperar(raiz, lambda: vista.tabla.item("1", "tags") == ("ganada",))
    assert vista.tabla.get_children() == ("2", "1")  # la nueva arriba, y la anterior ya cerrada sin duplicarse


def test_ver_dos_veces_no_abre_otra_ventana(abrir_con_lector, activo, raiz):
    ventana, _ = abrir_con_lector(activo, {"valor": foto()})
    ventana.filas[0].ver.invoke()
    primera = ventana.vistas["XAUUSDm"]
    ventana.filas[0].ver.invoke()
    assert ventana.vistas["XAUUSDm"] is primera


def test_si_no_se_puede_leer_lo_dice(abrir_con_lector, activo, raiz):
    ventana, _ = abrir_con_lector(activo, {"valor": "la terminal no responde"})
    ventana.filas[0].ver.invoke()
    vista = ventana.vistas["XAUUSDm"]
    esperar(raiz, lambda: "la terminal no responde" in vista.resumen.cget("text"))


def test_sin_vistas_abiertas_no_se_consulta_mt5(abrir_con_lector, activo, raiz):
    ventana, lecturas = abrir_con_lector(activo, {"valor": foto()})
    ventana.filas[0].ver.invoke()
    esperar(raiz, lambda: lecturas)
    esperar(raiz, lambda: ventana.lectura is None or ventana.lectura.done())
    ventana.cerrar_vista("XAUUSDm")
    ventana._atender_vistas()
    cuantas = len(lecturas)
    for _ in range(30):
        raiz.update()
        pc.time.sleep(0.01)
    assert len(lecturas) == cuantas and not ventana.vistas


# ---------------- sesión sin límite y vigilante ----------------
def test_la_duracion_vacia_lanza_una_sesion_sin_limite(activo, monkeypatch):
    orden = pc.comando(activo, AJUSTES, {**COMUNES, "horas": ""})
    assert "--horas" not in orden
    monkeypatch.setattr("sys.argv", ["x", *orden[2:]])
    assert demo_exness_bot.leer_argumentos().horas is None


def test_iniciar_apunta_el_bot_para_el_vigilante_y_detener_lo_borra(abrir, activo, tmp_path):
    ventana, lanzar, _ = abrir(activo)
    marcar(ventana)
    antes = pc.time.time()
    ventana.iniciar()
    apuntado = pc.leer_deseados(tmp_path / "vigilante.json")["XAUUSDm"]
    assert apuntado["orden"] == lanzar.call_args.args[1]
    assert antes + 5 * 3600 <= apuntado["fin"] <= pc.time.time() + 5 * 3600  # la duración por defecto: 5 horas
    ventana.detener(ventana.filas[0])
    assert pc.leer_deseados(tmp_path / "vigilante.json") == {} and activo.detener.exists()


def test_una_sesion_sin_limite_se_apunta_sin_hora_de_fin(abrir, activo, tmp_path):
    ventana, _, _ = abrir(activo)
    marcar(ventana)
    ventana.comunes["horas"].set("")
    ventana.iniciar()
    assert pc.leer_deseados(tmp_path / "vigilante.json")["XAUUSDm"]["fin"] is None


def test_la_ventana_dice_si_el_vigilante_esta_en_marcha(tmp_path):
    latido = tmp_path / "vigilante.txt"
    assert "parado" in pc.texto_vigilante(latido, AHORA)[0]
    latido.write_text(str(AHORA - 15))
    assert "en marcha" in pc.texto_vigilante(latido, AHORA)[0]
    assert "parado" in pc.texto_vigilante(latido, AHORA + 300)[0]


def test_el_cierre_al_limite_solo_se_pide_si_esta_marcado(activo, monkeypatch):
    assert "--cerrar-al-limite" not in pc.comando(activo, AJUSTES, {**COMUNES, "cerrar_al_limite": ""})
    orden = pc.comando(activo, AJUSTES, {**COMUNES, "cerrar_al_limite": "1"})
    assert orden[-1] == "--cerrar-al-limite"
    monkeypatch.setattr("sys.argv", ["x", *orden[2:]])
    args = demo_exness_bot.leer_argumentos()
    assert args.cerrar_al_limite and args.sin_entradas == ("23:30", "01:15")
    monkeypatch.setattr("sys.argv", ["x", *pc.comando(activo, AJUSTES, COMUNES)[2:]])
    assert not demo_exness_bot.leer_argumentos().cerrar_al_limite
