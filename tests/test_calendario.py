"""El calendario económico: lectura del archivo del servicio de MT5, filtro de noticias del bot y variantes del banco."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import calendario
import demo_exness_bot as lanzador
import experimentos as ex

NFP = 1_794_058_200  # 2026-11-06 13:30 UTC
MIN = 60


def escribir(ruta, exportado=NFP, eventos=None):
    eventos = eventos if eventos is not None else [
        (NFP, "USD", 3, "Nonfarm Payrolls"), (NFP, "USD", 3, "Unemployment Rate"), (NFP + 3600, "USD", 2, "Factory Orders"),
        (NFP - 7200, "EUR", 3, "ECB President Speech"), (NFP + 86400, "JPY", 3, "BoJ Rate Decision")]
    lineas = [f"# exportado={exportado}", "time_utc\tmoneda\timportancia\tevento_id\tnombre",
              *(f"{t}\t{moneda}\t{imp}\t{n}\t{nombre}" for n, (t, moneda, imp, nombre) in enumerate(eventos))]
    ruta.write_bytes(("\r\n".join(lineas) + "\r\n").encode("utf-8"))  # como lo escribe MT5
    return ruta


def test_lee_el_archivo_del_servicio_ordenado_por_hora(tmp_path):
    exportado, eventos = calendario.leer(escribir(tmp_path / "c.csv"))
    assert exportado == NFP and len(eventos) == 5
    assert eventos[0] == (NFP - 7200, "EUR", 3, "ECB President Speech")
    assert list(calendario.horas(eventos, ("USD",))) == [NFP]  # sin repetir, solo alto impacto
    assert list(calendario.horas(eventos, ("EUR", "USD"))) == [NFP - 7200, NFP]
    assert list(calendario.horas(eventos, ("USD",), importancia=2)) == [NFP, NFP + 3600]
    with pytest.raises(OSError):
        calendario.leer(tmp_path / "no_existe.csv")
    (tmp_path / "otro.csv").write_text("cualquier cosa\n", encoding="utf-8")
    with pytest.raises(ValueError):
        calendario.leer(tmp_path / "otro.csv")


def test_la_ventana_va_de_15_minutos_antes_a_15_despues():
    horas = np.array([NFP], dtype=np.int64)
    instantes = [NFP - 30 * MIN, NFP - 15 * MIN, NFP, NFP + 15 * MIN, NFP + 30 * MIN]
    assert list(calendario.cerca_de_noticia(horas, instantes, 15 * MIN, 15 * MIN)) == [False, True, True, True, False]
    assert not calendario.cerca_de_noticia(np.array([], dtype=np.int64), instantes, 15 * MIN, 15 * MIN).any()


def test_minutos_a_la_proxima_noticia_y_desde_la_ultima():
    horas = np.array([NFP, NFP + 120 * MIN], dtype=np.int64)
    hasta, desde = calendario.minutos_a_la_noticia(horas, [NFP - 30 * MIN, NFP, NFP + 45 * MIN, NFP + 3 * 86400])
    assert list(hasta) == [30, 0, 75, 1440] and list(desde) == [1440, 0, 45, 1440]
    hasta, desde = calendario.minutos_a_la_noticia(np.array([], dtype=np.int64), [NFP])
    assert (hasta[0], desde[0]) == (1440, 1440)


def test_el_calendario_del_activo_avisa_de_la_noticia_y_se_relee_al_cambiar(tmp_path):
    ruta = escribir(tmp_path / "c.csv")
    oro = calendario.Calendario(("USD",), ruta)
    assert oro.noticia_cercana(NFP - 15 * MIN) == (NFP, "USD", "Nonfarm Payrolls")
    assert oro.noticia_cercana(NFP - 30 * MIN) is None
    assert oro.noticia_cercana(NFP - 7200) is None  # el discurso del BCE no es de sus monedas
    assert oro.noticia_cercana(NFP + 3600) is None  # importancia media
    assert calendario.Calendario(("EUR", "USD"), ruta).noticia_cercana(NFP - 7200)[1] == "EUR"

    import os
    escribir(ruta, eventos=[(NFP - 30 * MIN, "USD", 3, "CPI")])
    os.utime(ruta, ns=(1, ruta.stat().st_mtime_ns + 1_000_000_000))
    assert oro.noticia_cercana(NFP - 30 * MIN) == (NFP - 30 * MIN, "USD", "CPI")


def test_un_calendario_caducado_o_ausente_no_se_da_por_bueno(tmp_path):
    ruta = escribir(tmp_path / "c.csv", exportado=NFP - calendario.CADUCA_S - 3600)
    with pytest.raises(LookupError, match="hace 49 h"):
        calendario.Calendario(("USD",), ruta).noticia_cercana(NFP)
    with pytest.raises(OSError):
        calendario.Calendario(("USD",), tmp_path / "no_existe.csv").noticia_cercana(NFP)


def test_el_bot_no_evalua_la_vela_cercana_a_una_noticia(tmp_path, monkeypatch):
    registro, evaluar = Mock(), Mock()
    monkeypatch.setattr(lanzador.trading_bot, "log", registro)
    envuelta = lanzador.evaluar_sin_noticias(evaluar, calendario.Calendario(("USD",), escribir(tmp_path / "c.csv")))
    envuelta({}, None, NFP, {})
    evaluar.assert_not_called()
    assert registro.info.call_args.args[1:] == ("13:30", "USD", "Nonfarm Payrolls")
    envuelta({}, None, NFP + 30 * MIN, {})
    evaluar.assert_called_once_with({}, None, NFP + 30 * MIN, {})


def test_sin_calendario_fiable_el_bot_avisa_y_evalua_igual(tmp_path, monkeypatch):
    registro, evaluar = Mock(), Mock()
    monkeypatch.setattr(lanzador.trading_bot, "log", registro)
    caducado = escribir(tmp_path / "c.csv", exportado=NFP - 5 * 86400)
    for ruta in (caducado, tmp_path / "no_existe.csv"):
        lanzador.evaluar_sin_noticias(evaluar, calendario.Calendario(("USD",), ruta))({}, None, NFP, {})
    assert evaluar.call_count == 2 and registro.warning.call_count == 2
    assert "Calendario económico no disponible" in registro.warning.call_args.args[0]


def test_las_variantes_de_noticias_miden_desde_el_instante_de_la_entrada():
    df = pd.DataFrame({"time_utc": [NFP - 45 * MIN, NFP - 30 * MIN, NFP - 15 * MIN, NFP, NFP + 15 * MIN]})
    resultado = ex.noticias(df, {"noticias": np.array([NFP], dtype=np.int64)})
    assert list(resultado["noticia_cerca"]) == [0, 1, 1, 1, 0]  # la vela que cierra a las 13:15 entraría a 15 min de la noticia
    assert resultado["noticia_hasta"].iloc[0] == pytest.approx(30 / 1440)
    assert resultado["noticia_desde"].iloc[4] == pytest.approx(30 / 1440)
    assert list(ex.VARIANTES["noticias_filtro"].filtro(resultado)) == [True, False, False, False, True]
    assert ex.VARIANTES["noticias_feature"].columnas({}) == ["noticia_hasta", "noticia_desde"]
