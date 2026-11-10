"""La medición de la ejecución real frente a lo que supone el backtest."""
import csv
import json
from types import SimpleNamespace

import medicion
import trading_bot

APERTURA = {"fecha": "2026-11-09T17:14:59", "evento": "Apertura LONG", "ticket": "7", "tipo": "BUY", "precio_pedido": "100.00",
            "precio": "100.25", "slippage_puntos": "25.0", "spread_puntos": "112", "sl": "90", "tp": "120", "retcode": "10009"}


def cierre(ticket="7", precio="89.90", comentario="[sl 90.25]", beneficio="-10.35", **mas) -> dict:
    return {"fecha": "2026-11-10T03:00:00", "evento": "cierre", "ticket": ticket, "precio": precio, "comentario": comentario,
            "beneficio": beneficio, "swap": "-0.10", "comision": "0.0", "tasas": "0.0", **mas}


def test_mide_entradas_y_salidas_en_contra():
    venta = {**APERTURA, "evento": "Apertura SHORT", "ticket": "8", "tipo": "SELL", "slippage_puntos": "-3.0", "spread_puntos": "100"}
    m = medicion.medir([APERTURA, cierre(), venta, cierre("8", "80.02", "[tp 80.00]", "20.0")], point=0.01)
    assert (m.entradas, m.spreads) == ([25.0, -3.0], [112.0, 100.0])
    assert m.stops == [35.0]  # la compra salió 35 puntos por debajo de su stop
    assert m.objetivos == [2.0]  # la venta se recompró 2 puntos por encima de su objetivo
    assert (m.cerradas, round(m.resultado, 2)) == ({"7", "8"}, 9.45)


def test_las_ordenes_no_ejecutadas_y_los_cierres_por_tiempo_van_aparte():
    rechazada = {**APERTURA, "ticket": "", "precio": "", "slippage_puntos": "", "retcode": "10016"}
    tiempo = {**APERTURA, "evento": "Cierre por time stop (ticket 7)", "tipo": "SELL", "slippage_puntos": "4.0"}
    limite = {**tiempo, "evento": "Cierre por limite (ticket 7)", "slippage_puntos": "6.0"}
    m = medicion.medir([rechazada, APERTURA, tiempo, limite, cierre(comentario="time stop")], point=0.01)
    assert (m.no_ejecutadas, m.entradas, m.a_mercado, m.stops) == (1, [25.0], [4.0, 6.0], [])


def test_sin_el_punto_no_se_miden_stops_y_desde_filtra_por_fecha():
    assert medicion.medir([APERTURA, cierre()]).stops == []
    m = medicion.medir([APERTURA, cierre()], point=0.01, desde="2026-11-10")
    assert (m.entradas, m.stops, m.cerradas) == ([], [], {"7"})  # sin su apertura no se sabe el lado


def test_lee_las_columnas_que_escribe_el_bot(tmp_path):
    ruta = tmp_path / "operaciones.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=trading_bot.CAMPOS_OPERACIONES)
        escritor.writeheader()
        escritor.writerows([APERTURA, cierre()])
    assert medicion.medir(medicion.leer_filas(ruta), point=0.01).stops == [35.0]
    assert medicion.leer_filas(tmp_path / "no_existe.csv") == []


def test_el_informe_compara_con_el_supuesto(tmp_path):
    activo = SimpleNamespace(nombre="Nasdaq 100", simbolo="USTECm", modelo=tmp_path / "USTECm.pkl")
    assert medicion.leer_supuestos(activo) == {}
    supuestos = {"point": 0.01, "spread_mediano": 112.0, "slippage_puntos": 14.0}
    activo.modelo.with_suffix(".json").write_text(json.dumps({"supuestos": supuestos}), encoding="utf-8")
    assert medicion.leer_supuestos(activo) == supuestos

    texto = medicion.informe(activo, medicion.medir([APERTURA, cierre()], point=0.01), supuestos)
    entrada = next(linea for linea in texto.splitlines() if linea.strip().startswith("Entrada"))
    assert entrada.split()[1:] == ["1", "25.0", "25.0", "25.0", "25.0", "+11.0"]
    assert "14.0 puntos" in texto and "orientativo" in texto and "-10.45" in texto
    assert "no está guardado" in medicion.informe(activo, medicion.medir([APERTURA]), {})
    assert "Sin operaciones" in medicion.informe(activo, medicion.medir([]), supuestos)
