"""El inventario de ticks: qué días se piden y cómo se resume un día."""
from datetime import datetime, timezone

import numpy as np

import inventario_ticks as it


def test_los_dias_de_muestra_son_miercoles_ya_cerrados():
    hoy = datetime(2026, 11, 10, 17, 0, tzinfo=timezone.utc)  # martes
    dias = it.dias_de_muestra(2025, hoy)
    assert [f"{d:%Y-%m-%d}" for d in dias] == ["2025-01-08", "2025-07-09", "2026-01-14", "2026-07-08", "2026-10-21", "2026-10-28", "2026-11-04"]
    assert all(d.weekday() == 2 for d in dias)
    assert it.dias_de_muestra(2027, hoy)[-1] == dias[-1]  # un año futuro no añade días


def test_el_resumen_cuenta_campos_ritmo_y_spread():
    tipo = [("time_msc", "i8"), ("bid", "f8"), ("ask", "f8"), ("last", "f8"), ("volume", "u8"), ("flags", "u4")]
    ticks = np.array([(0, 1.1000, 1.1002, 0, 0, 6), (30_000, 1.1001, 1.1003, 0, 0, 6), (60_000, 1.1001, 1.1005, 0, 0, 4),
                      (120_000, 1.1002, 1.1004, 0, 0, 2)], dtype=tipo)
    r = it.resumir(ticks, 0.0001)
    assert (r["ticks"], r["por_minuto"], r["ms_mediana"], r["con_last"], r["con_volumen"]) == (4, 2.0, 30_000, 0, 0)
    assert r["banderas"] == {"bid": 3, "ask": 3} and round(r["spread_mediano"], 6) == 2.0
    assert it.resumir(None, 0.0001) == {"ticks": 0} == it.resumir(ticks[:0], 0.0001)
    assert "sin ticks" in it.fila(datetime(2025, 1, 8), {"ticks": 0}) and "ask 75%" in it.fila(datetime(2026, 1, 14), r)
