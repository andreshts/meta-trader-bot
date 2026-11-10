"""Inventario de ticks (Plan_de_mejoras.md, 5.1): cuánto histórico de ticks entrega el broker por activo y con qué campos.

Uso: python inventario_ticks.py [SIMBOLO ...] [--desde 2019]

Pide solo un día de muestra por semestre (el segundo miércoles de enero y de julio) y los últimos días, porque la
terminal descarga y guarda en disco el mes entero de cada día que se le pide. Solo lee: no envía órdenes.
"""
import argparse
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from activos import ACTIVOS
from cuenta import CUENTA

BANDERAS = {2: "bid", 4: "ask", 8: "last", 16: "volumen", 32: "compra", 64: "venta"}  # TICK_FLAG_*


def segundo_miercoles(anio: int, mes: int) -> datetime:
    primero = datetime(anio, mes, 1, tzinfo=timezone.utc)
    return primero + timedelta(days=(2 - primero.weekday()) % 7 + 7)


def dias_de_muestra(desde: int, hoy: datetime) -> list:
    """Un día por semestre desde `desde` y los tres últimos miércoles ya cerrados."""
    semestrales = [segundo_miercoles(anio, mes) for anio in range(desde, hoy.year + 1) for mes in (1, 7)]
    ultimo = hoy - timedelta(days=(hoy.weekday() - 2) % 7 or 7)
    recientes = [ultimo.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(weeks=n) for n in (2, 1, 0)]
    return sorted({d for d in semestrales + recientes if d + timedelta(days=1) <= hoy})


def resumir(ticks, point: float) -> dict:
    """Qué trae un día de ticks: cuántos, cada cuánto, con qué campos y con qué spread."""
    if ticks is None or len(ticks) == 0:
        return {"ticks": 0}
    tiempos = ticks["time_msc"].astype(np.int64)
    banderas = Counter()
    for valor, veces in zip(*np.unique(ticks["flags"], return_counts=True)):
        for bit, nombre in BANDERAS.items():
            if valor & bit:
                banderas[nombre] += int(veces)
    con_los_dos = (ticks["bid"] > 0) & (ticks["ask"] > 0)
    spread = (ticks["ask"][con_los_dos] - ticks["bid"][con_los_dos]) / point
    return {"ticks": len(ticks), "horas": round((tiempos[-1] - tiempos[0]) / 3_600_000, 1),
            "por_minuto": round(len(ticks) / max(1, (tiempos[-1] - tiempos[0]) / 60_000), 1),
            "ms_mediana": float(np.median(np.diff(tiempos))) if len(ticks) > 1 else None,
            "spread_mediano": float(np.median(spread)) if len(spread) else None,
            "con_last": int((ticks["last"] > 0).sum()), "con_volumen": int((ticks["volume"] > 0).sum()),
            "banderas": dict(banderas)}


def fila(dia: datetime, r: dict) -> str:
    if not r["ticks"]:
        return f"  {dia:%Y-%m-%d}  sin ticks"
    campos = ", ".join(f"{nombre} {veces / r['ticks']:.0%}" for nombre, veces in sorted(r["banderas"].items()))
    return (f"  {dia:%Y-%m-%d}  {r['ticks']:>8} ticks en {r['horas']:>4} h  {r['por_minuto']:>6}/min  cada {r['ms_mediana']:>6.0f} ms  "
            f"spread {r['spread_mediano']:>6.1f}  last {r['con_last']}  volumen {r['con_volumen']}  [{campos}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("simbolos", nargs="*", default=list(ACTIVOS))
    parser.add_argument("--desde", type=int, default=2019, help="primer año del que se piden muestras")
    args = parser.parse_args()
    if not mt5.initialize(CUENTA.terminal):
        raise SystemExit(f"No se pudo conectar con MT5: {mt5.last_error()}")
    try:
        datos = Path(mt5.terminal_info().data_path)
        libre_antes = shutil.disk_usage(datos).free
        for simbolo in args.simbolos:
            info = mt5.symbol_info(simbolo)
            if info is None:
                print(f"\n=== {simbolo}: no existe en esta cuenta")
                continue
            print(f"\n=== {simbolo}", flush=True)
            for dia in dias_de_muestra(args.desde, datetime.now(timezone.utc)):
                ticks = mt5.copy_ticks_range(simbolo, dia, dia + timedelta(days=1), mt5.COPY_TICKS_ALL)
                print(fila(dia, resumir(ticks, info.point)), flush=True)
        usado = (libre_antes - shutil.disk_usage(datos).free) / 2**20
        print(f"\nEspacio en disco que ocupó la descarga: {usado:.0f} MB (libre: {shutil.disk_usage(datos).free / 2**30:.0f} GB)")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
