"""Calendario económico que exporta el servicio mql5/RF_Calendario.mq5 (la API de Python de MT5 no lo ofrece).

El archivo está en Common\\Files, con las horas ya en UTC. Lo usan el filtro de noticias del bot
(demo_exness_bot.py) y las variantes de noticias del banco de experimentos. Solo lee.
"""
import csv
from pathlib import Path

import numpy as np

import panel_mt5

RUTA = panel_mt5.RUTA.with_name("rf_calendario.csv")
ALTA = 3  # importancia de las noticias de alto impacto
CADUCA_S = 2 * 86400  # el servicio exporta cada hora: más viejo que esto, el calendario ya no es de fiar
MINUTOS_ANTES, MINUTOS_DESPUES = 15, 15  # ventana sin entradas alrededor de cada noticia
# Monedas cuyas noticias mueven cada activo.
MONEDAS = {"EURUSDm": ("EUR", "USD"), "XAUUSDm": ("USD",), "US500m": ("USD",), "USTECm": ("USD",), "DE30m": ("EUR", "USD")}


def leer(ruta: Path = RUTA) -> tuple:
    """(hora UTC de la exportación, eventos). Cada evento es (hora UTC, moneda, importancia, nombre), por hora.

    OSError si el archivo no existe; ValueError si no tiene la forma esperada.
    """
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        cabecera = f.readline().strip()
        if not cabecera.startswith("# exportado="):
            raise ValueError(f"{ruta} no empieza con la hora de exportación")
        exportado = int(cabecera.partition("=")[2])
        eventos = [(int(fila["time_utc"]), fila["moneda"], int(fila["importancia"]), fila["nombre"])
                   for fila in csv.DictReader(f, delimiter="\t")]
    return exportado, sorted(eventos)


def horas(eventos: list, monedas, importancia: int = ALTA) -> np.ndarray:
    """Las horas UTC, ordenadas y sin repetir, de los eventos de esas monedas con al menos esa importancia."""
    return np.unique(np.array([t for t, moneda, imp, _ in eventos if moneda in monedas and imp >= importancia], dtype=np.int64))


def cerca_de_noticia(horas_eventos: np.ndarray, instantes, antes_s: int, despues_s: int) -> np.ndarray:
    """True en cada instante que tiene una noticia en los próximos `antes_s` o en los últimos `despues_s` segundos."""
    instantes = np.asarray(instantes, dtype=np.int64)
    return np.searchsorted(horas_eventos, instantes + antes_s, side="right") > np.searchsorted(horas_eventos, instantes - despues_s, side="left")


def minutos_a_la_noticia(horas_eventos: np.ndarray, instantes, tope_min: int = 1440) -> tuple:
    """(minutos hasta la próxima noticia, minutos desde la última), cada uno limitado a `tope_min`."""
    instantes = np.asarray(instantes, dtype=np.int64)
    if len(horas_eventos) == 0:
        return np.full(len(instantes), float(tope_min)), np.full(len(instantes), float(tope_min))
    siguiente = np.searchsorted(horas_eventos, instantes, side="left")
    hasta = np.where(siguiente < len(horas_eventos), horas_eventos[np.minimum(siguiente, len(horas_eventos) - 1)] - instantes, np.inf)
    anterior = np.searchsorted(horas_eventos, instantes, side="right") - 1
    desde = np.where(anterior >= 0, instantes - horas_eventos[np.maximum(anterior, 0)], np.inf)
    return np.minimum(hasta / 60, tope_min), np.minimum(desde / 60, tope_min)


class Calendario:
    """El calendario de las monedas de un activo, releído cuando el servicio lo vuelve a exportar."""

    def __init__(self, monedas, ruta: Path = RUTA, importancia: int = ALTA):
        self.monedas, self.ruta, self.importancia = tuple(monedas), ruta, importancia
        self._leido, self._exportado, self._eventos = None, 0, []

    def _actualizar(self) -> None:
        modificado = self.ruta.stat().st_mtime_ns  # OSError si no existe
        if modificado != self._leido:
            exportado, eventos = leer(self.ruta)
            self._exportado = exportado
            self._eventos = [e for e in eventos if e[1] in self.monedas and e[2] >= self.importancia]
            self._leido = modificado

    def noticia_cercana(self, instante: int, antes_s: int = MINUTOS_ANTES * 60, despues_s: int = MINUTOS_DESPUES * 60):
        """(hora UTC, moneda, nombre) de la noticia que impide entrar en ese instante (UTC), o None.

        OSError o ValueError si el calendario no se puede leer; LookupError si está caducado.
        """
        self._actualizar()
        if instante - self._exportado > CADUCA_S:
            raise LookupError(f"el calendario se exportó hace {(instante - self._exportado) / 3600:.0f} h")
        for hora, moneda, _, nombre in self._eventos:
            if instante - despues_s <= hora <= instante + antes_s:
                return hora, moneda, nombre
        return None
