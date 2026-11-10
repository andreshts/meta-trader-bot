"""El informe de rendimiento: estadísticas, desgloses y comparación con el backtest."""
import math
from types import SimpleNamespace

import pytest

import informe
from informe import Trade
from operaciones import Instantanea, Operacion

LUNES = 1_794_182_400  # 2026-11-09 00:00 UTC
H = 3600


def trade(resultado, r, simbolo="XAUUSDm", compra=True, hora=14, motivo="Stop loss", n=0) -> Trade:
    apertura = LUNES + n * 86400 + hora * H
    return Trade(simbolo, compra, apertura, apertura + H, motivo, resultado, r)


def test_las_estadisticas_resumen_una_serie_de_trades_en_orden_de_cierre():
    trades = [trade(20.0, 2.0, n=2), trade(-10.0, -1.0, n=0), trade(-10.0, -1.0, n=1), trade(-5.0, -0.5, n=3)]
    e = informe.estadisticas(trades)
    assert (e["trades"], e["acierto"], e["neto"]) == (4, 0.25, -5.0)
    assert e["drawdown"] == 20.0  # -10 y -10 antes de la ganancia
    assert e["profit_factor"] == pytest.approx(20 / 25) and e["esperanza_R"] == pytest.approx(-0.125)
    assert e["t"] == pytest.approx(-0.125 / (1.4361406616345072 / 2))


def test_las_estadisticas_aguantan_casos_limite():
    vacio = informe.estadisticas([])
    assert (vacio["trades"], vacio["acierto"], vacio["esperanza_R"], vacio["t"], vacio["profit_factor"]) == (0, 0.0, None, None, 0.0)
    solo_ganancias = informe.estadisticas([trade(5.0, 0.5), trade(5.0, 0.5, n=1)])
    assert math.isinf(solo_ganancias["profit_factor"]) and solo_ganancias["t"] is None and solo_ganancias["drawdown"] == 0.0
    sin_r = informe.estadisticas([trade(5.0, None), trade(-2.0, -0.2, n=1)])
    assert sin_r["esperanza_R"] == -0.2 and sin_r["neto"] == 3.0  # el trade sin stop no entra en la esperanza


def test_los_desgloses_agrupan_por_activo_lado_salida_y_sesion():
    trades = [trade(10.0, 1.0, hora=2), trade(-5.0, -0.5, simbolo="DE30m", compra=False, hora=8, motivo="Take profit"),
              trade(-5.0, -0.5, hora=14), trade(1.0, 0.1, hora=22, motivo="Por tiempo")]
    assert {s: e["trades"] for s, e in informe.desglose(trades, lambda t: t.simbolo).items()} == {"XAUUSDm": 3, "DE30m": 1}
    sesiones = informe.desglose(trades, lambda t: informe.sesion_de(t.apertura))
    assert list(sesiones) == ["Asia (00-07)", "Europa (07-13)", "América (13-21)", "Cierre (21-24)"]
    assert informe.sesion_de(LUNES + 7 * H) == "Europa (07-13)" and informe.sesion_de(LUNES + 21 * H - 1) == "América (13-21)"


def test_la_referencia_es_la_media_de_las_semillas_de_la_base():
    filas = [{"variante": "base", "simbolo": "XAUUSDm", "semilla": "42", "esperanza_R": "0.03"},
             {"variante": "base", "simbolo": "XAUUSDm", "semilla": "1", "esperanza_R": "0.01"},
             {"variante": "base", "simbolo": "XAUUSDm", "semilla": "1", "esperanza_R": "0.05"},  # una repetición: cuenta la última
             {"variante": "squeeze", "simbolo": "XAUUSDm", "semilla": "42", "esperanza_R": "9"}]
    assert informe.referencia(filas) == {"XAUUSDm": pytest.approx(0.04)}
    assert informe.referencia([]) == {}


def test_el_texto_compara_con_el_backtest_y_avisa_si_hay_pocos_trades():
    trades = [trade(20.0, 2.0, motivo="Take profit"), trade(-10.0, -1.0, simbolo="DE30m", compra=False, n=1)]
    texto = informe.texto(trades, LUNES, LUNES + 7 * 86400, "USD", {"XAUUSDm": 0.024}, {"XAUUSDm": "Oro", "DE30m": "DAX"})
    lineas = texto.splitlines()
    assert lineas[0] == "📈 Informe de rendimiento del 09/11/2026 al 16/11/2026"
    assert lineas[1].startswith("Total: 2 trades · acierto 50% · +10.00 USD · +0.50 R · PF 2.00 · drawdown 10.00 USD · t = ")
    assert "  Oro: 1 trade · acierto 100% · +20.00 USD · +2.00 R / +0.02 R · PF ∞" in lineas
    assert "  DAX: 1 trade · acierto 0% · -10.00 USD · -1.00 R / s/d · PF 0.00" in lineas
    assert "  Venta: 1 trade · acierto 0% · -10.00 USD · -1.00 R · PF 0.00" in lineas and "  Take profit: 1 trade" in texto
    assert "Con 2 trades el resultado es orientativo" in lineas[-1]
    muchos = [trade(1.0, 0.1, n=n % 5, hora=n % 24) for n in range(30)]
    assert "orientativo" not in informe.texto(muchos, None, LUNES, "USD", {}, {}) and "hasta el 09/11/2026" in informe.texto(muchos, None, LUNES, "USD", {}, {})
    assert "Sin operaciones cerradas" in informe.texto([], LUNES, LUNES + 86400, "USD", {}, {})


def test_lee_solo_las_operaciones_cerradas_del_periodo_y_las_pasa_a_r():
    def op(ticket, cierre, resultado, sl=1.09):
        return Operacion(ticket, True, 0.01, LUNES, 1.10, sl, 1.12, cierre, 1.09, "Stop loss", resultado)

    operaciones = [op(1, LUNES + H, -5.0), op(2, None, 3.0), op(3, LUNES - 86400, -5.0), op(4, LUNES + 2 * H, 4.0, sl=0.0)]
    leer = lambda activos: {"XAUUSDm": Instantanea(operaciones, 2, "USD")}  # noqa: E731
    trades, moneda = informe.leer_trades({"XAUUSDm": SimpleNamespace()}, LUNES, lambda simbolo, o: 5.0 if o.sl else None, leer)
    assert moneda == "USD" and [(t.resultado, t.r) for t in trades] == [(-5.0, -1.0), (4.0, None)]
    assert len(informe.leer_trades({"XAUUSDm": SimpleNamespace()}, None, lambda s, o: 5.0, leer)[0]) == 3  # --todo
    with pytest.raises(RuntimeError, match="terminal"):
        informe.leer_trades({"XAUUSDm": SimpleNamespace()}, None, leer=lambda activos: {"XAUUSDm": "no se pudo conectar con la terminal"})
