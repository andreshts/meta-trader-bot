"""Compara la ejecución real, registrada en operaciones.csv, con lo que supone el backtest de cada activo.

Uso: python medicion.py [SIMBOLO ...] [--desde AAAA-MM-DD]   (sin símbolos, todos)
El deslizamiento va en puntos y en contra: positivo es peor precio que el pedido (o que el nivel del stop o del
objetivo); negativo, mejor. Solo lee archivos: no se conecta a MT5 y puede ejecutarse con los bots en marcha.
"""
import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from activos import ACTIVOS, Activo

EJECUTADA = {"10009", "10010"}  # TRADE_RETCODE_DONE y DONE_PARTIAL
NIVEL = re.compile(r"\[(sl|tp) ([\d.]+)\]")  # comentario que MT5 pone al deal de un cierre por stop u objetivo
POCAS = 30  # con menos ejecuciones, la media todavía se mueve mucho
FILAS = (("entradas", "Entrada"), ("stops", "Salida por stop"), ("objetivos", "Salida por objetivo"),
         ("por_tiempo", "Cierre por tiempo"))


@dataclass
class Medicion:
    entradas: list = field(default_factory=list)  # deslizamiento de cada ejecución, en puntos
    stops: list = field(default_factory=list)
    objetivos: list = field(default_factory=list)
    por_tiempo: list = field(default_factory=list)
    spreads: list = field(default_factory=list)  # spread al enviar cada entrada ejecutada
    no_ejecutadas: int = 0
    cerradas: set = field(default_factory=set)
    resultado: float = 0.0  # neto de swap, comisión y tasas


def _numero(texto) -> float | None:
    try:
        valor = float(texto)
    except (TypeError, ValueError):
        return None
    return valor if math.isfinite(valor) else None


def medir(filas, point: float | None = None, desde: str = "") -> Medicion:
    """Resume las filas de un operaciones.csv. Sin `point`, no se mide el deslizamiento de stops ni objetivos."""
    m, compras = Medicion(), {}
    for fila in filas:
        if (fila.get("fecha") or "") < desde:
            continue
        evento, ticket, precio = fila.get("evento") or "", fila.get("ticket"), _numero(fila.get("precio"))
        desliz = _numero(fila.get("slippage_puntos"))
        if evento.startswith(("Apertura", "Cierre por time stop")):
            if fila.get("retcode") not in EJECUTADA or desliz is None:
                m.no_ejecutadas += 1
            elif evento.startswith("Apertura"):
                compras[ticket] = fila.get("tipo") == "BUY"
                m.entradas.append(desliz)
                if (spread := _numero(fila.get("spread_puntos"))) is not None:
                    m.spreads.append(spread)
            else:
                m.por_tiempo.append(desliz)
        elif evento in ("entrada", "cierre"):
            m.resultado += sum(_numero(fila.get(c)) or 0.0 for c in ("beneficio", "swap", "comision", "tasas"))
            if evento == "entrada":
                continue
            m.cerradas.add(ticket)
            nivel = NIVEL.search(fila.get("comentario") or "")
            if nivel and point and precio is not None and ticket in compras:
                contra = (float(nivel[2]) - precio) if compras[ticket] else (precio - float(nivel[2]))
                (m.stops if nivel[1] == "sl" else m.objetivos).append(round(contra / point, 1))
    return m


def leer_filas(ruta: Path) -> list[dict]:
    try:
        with open(ruta, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def leer_supuestos(activo: Activo) -> dict:
    """Lo que supuso el backtest del modelo (train_exness.py lo guarda junto a él); vacío si no está."""
    try:
        datos = json.loads(activo.modelo.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return datos.get("supuestos") or {} if isinstance(datos, dict) else {}


def _p90(valores: list) -> float:
    return sorted(valores)[max(0, math.ceil(0.9 * len(valores)) - 1)]


def _linea(nombre: str, valores: list, supuesto: float | None = None) -> str:
    if not valores:
        return f"  {nombre:<21}{0:>4}"
    media = statistics.fmean(valores)
    texto = (f"  {nombre:<21}{len(valores):>4}{media:>9.1f}{statistics.median(valores):>9.1f}{_p90(valores):>9.1f}"
             f"{max(valores):>9.1f}")
    if supuesto is not None:
        texto += f"{media - supuesto:>+12.1f}"
    return texto


def informe(activo: Activo, m: Medicion, supuestos: dict) -> str:
    supuesto = _numero(supuestos.get("slippage_puntos"))
    lineas = [f"=== {activo.nombre} ({activo.simbolo}): {len(m.entradas)} entradas, {len(m.cerradas)} operaciones cerradas"]
    if not (m.entradas or m.cerradas or m.no_ejecutadas):
        return lineas[0] + "\n  Sin operaciones registradas"
    if supuesto is None:
        lineas.append("  Supuesto del backtest: no está guardado (python train_exness.py --supuestos)")
    else:
        lineas.append(f"  Supuesto del backtest: {supuesto:.1f} puntos de deslizamiento por ejecución; "
                      f"spread mediano {supuestos.get('spread_mediano', '?')}")
    lineas.append(f"  {'Deslizamiento':<21}{'n':>4}{'media':>9}{'mediana':>9}{'p90':>9}{'máx':>9}{'vs supuesto':>12}")
    lineas += [_linea(nombre, getattr(m, clave), supuesto) for clave, nombre in FILAS]
    lineas.append(_linea("Spread al entrar", m.spreads))
    if "point" not in supuestos:
        lineas.append("  Sin el tamaño del punto guardado, no se mide el deslizamiento de stops ni objetivos")
    ejecuciones = len(m.entradas) + len(m.stops) + len(m.objetivos) + len(m.por_tiempo)
    if 0 < ejecuciones < POCAS:
        lineas.append(f"  {ejecuciones} ejecuciones medidas: orientativo hasta llegar a {POCAS}")
    lineas.append(f"  Órdenes no ejecutadas: {m.no_ejecutadas} · resultado neto registrado: {m.resultado:+.2f}")
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simbolos", nargs="*", help=f"entre: {', '.join(ACTIVOS)}")
    parser.add_argument("--desde", type=date.fromisoformat, help="solo las filas desde esta fecha, AAAA-MM-DD")
    args = parser.parse_args()
    if desconocidos := set(args.simbolos) - set(ACTIVOS):
        parser.error(f"símbolos no válidos: {sorted(desconocidos)}")
    for simbolo in args.simbolos or ACTIVOS:
        activo, supuestos = ACTIVOS[simbolo], leer_supuestos(ACTIVOS[simbolo])
        medicion = medir(leer_filas(activo.carpeta / "operaciones.csv"), _numero(supuestos.get("point")),
                         args.desde.isoformat() if args.desde else "")
        print(informe(activo, medicion, supuestos) + "\n")


if __name__ == "__main__":
    main()
