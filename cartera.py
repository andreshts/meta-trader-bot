"""Mide reglas de cartera sobre los trades fuera de muestra del walk-forward de los cinco activos a la vez.

Uso: python cartera.py [--repetir]   (--repetir vuelve a generar los trades en vez de usar los guardados)

Cada trade arriesga 1 R (el mismo porcentaje de la cuenta), y las reglas, fijadas antes de ver su resultado, cambian
qué trades se toman o cuánto pesa cada uno:
  correlación  no se entra si otro activo del mismo grupo (activos.GRUPOS_CORRELACION) ya tiene una posición del mismo lado
  reducción    con la cartera a UMBRAL_DD_R o más por debajo de su máximo, cada trade nuevo pesa FACTOR_DD
  confianza    un trade cuya probabilidad supera su umbral en MARGEN_CONFIANZA o más pesa FACTOR_CONFIANZA
Usa las velas de experimentos.py (--descargar) y, como él, nunca toca el holdout. No envía órdenes.
"""
import argparse
import csv
import heapq
import pickle
import statistics
from datetime import datetime

import numpy as np
import pandas as pd

import config
import experimentos as ex
import train_model as tm
from activos import ACTIVOS, grupo_de
from features import FEATURES

TRADES = ex.DATOS / "trades_cartera.pkl"
RESULTADOS = ex.CARPETA / "cartera.csv"
UMBRAL_DD_R, FACTOR_DD = 10.0, 0.5  # con 0,5 % de riesgo por trade, 10 R son un 5 % de drawdown
MARGEN_CONFIANZA, FACTOR_CONFIANZA = 0.05, 1.5
REGLAS = {"base": (), "correlación": ("correlacion",), "reducción": ("reducir",), "confianza": ("confianza",),
          "las tres": ("correlacion", "reducir", "confianza")}
CAMPOS = ["fecha", "regla", "semilla", "trades", "total_R", "esperanza_R", "t", "max_drawdown_R", "total_sobre_drawdown"]


def simular_con_detalle(filas: pd.DataFrame, p_long, p_short, umbral_long, umbral_short, tiempos) -> pd.DataFrame:
    """Los mismos trades que train_model.simular, con su entrada y salida (UTC), su lado y cuánto superó la
    probabilidad a su umbral. `tiempos` es la apertura de cada vela por posición."""
    pos = filas.index.to_numpy()
    r = {lado: filas[f"r_{lado}"].to_numpy() for lado in tm.LADOS}
    k = {lado: filas[f"k_{lado}"].to_numpy() for lado in tm.LADOS}
    to = {lado: filas[f"to_{lado}"].to_numpy() for lado in tm.LADOS}
    proba, umbral = {"long": np.asarray(p_long), "short": np.asarray(p_short)}, {"long": umbral_long, "short": umbral_short}
    senal = {lado: proba[lado] >= umbral[lado] if umbral[lado] is not None else np.zeros(len(pos), dtype=bool) for lado in tm.LADOS}
    trades, libre = [], -1
    for i in np.flatnonzero(senal["long"] != senal["short"]):
        if pos[i] < libre:
            continue
        lado = "long" if senal["long"][i] else "short"
        ultima = int(k[lado][i])
        trades.append({"entrada": int(tiempos[pos[i] + 1]), "salida": int(tiempos[ultima]) + config.TIMEFRAME_SEGUNDOS, "lado": lado,
                       "r": float(r[lado][i]), "margen": float(proba[lado][i] - umbral[lado])})
        libre = ultima + (1 if to[lado][i] else 0)
    return pd.DataFrame(trades, columns=["entrada", "salida", "lado", "r", "margen"])


def trades_de(simbolo: str, paquete: dict, semilla: int) -> pd.DataFrame:
    """Los trades de test de todas las ventanas del walk-forward de la base, con una semilla."""
    base = ex.VARIANTES[ex.BASE]
    tiempos = paquete["velas"]["time"].to_numpy(dtype=np.int64)
    with ex.con_ajustes(simbolo, paquete["velas"]), ex.con_variante(base, {}):
        datos = ex.preparar(base, paquete["velas"], paquete["point"], paquete["costes"], {})
        ventanas = tm.walk_forward(datos, semilla)
        partes = []
        for ventana, (_, _, i_test) in zip(ventanas, tm.ventanas_walk_forward(datos, config.WF_VENTANAS)):
            test = datos.iloc[i_test]
            p = {lado: ventana["modelos"][lado].predict_proba(test[FEATURES])[:, 1] for lado in tm.LADOS}
            detalle = simular_con_detalle(test, p["long"], p["short"], ventana["umbrales"]["long"], ventana["umbrales"]["short"], tiempos)
            if not np.allclose(detalle["r"].to_numpy(), ventana["trades"]):
                raise RuntimeError(f"{simbolo}: los trades con detalle no coinciden con los del walk-forward")
            partes.append(detalle)
    return pd.concat(partes, ignore_index=True).assign(simbolo=simbolo, semilla=semilla)


def simular_cartera(trades: pd.DataFrame, correlacion: bool = False, reducir: bool = False, confianza: bool = False) -> pd.DataFrame:
    """Los trades que se toman, en orden de entrada, con su `peso` y su resultado `r_cartera` = peso × r.

    El drawdown que decide la reducción es el de lo ya cerrado al entrar: lo que el bot sabe en ese momento.
    """
    abiertos, tomados = [], []  # abiertos: (salida, orden, símbolo, lado, r_cartera)
    curva = pico = 0.0
    for orden, t in enumerate(trades.sort_values(["entrada", "simbolo"]).itertuples(index=False)):
        while abiertos and abiertos[0][0] <= t.entrada:
            curva += heapq.heappop(abiertos)[4]
            pico = max(pico, curva)
        if correlacion and any(lado == t.lado and s != t.simbolo and s in grupo_de(t.simbolo) for _, _, s, lado, _ in abiertos):
            continue
        peso = (FACTOR_CONFIANZA if confianza and t.margen >= MARGEN_CONFIANZA else 1.0) * (
            FACTOR_DD if reducir and pico - curva >= UMBRAL_DD_R else 1.0)
        heapq.heappush(abiertos, (t.salida, orden, t.simbolo, t.lado, peso * t.r))
        tomados.append({**t._asdict(), "peso": peso, "r_cartera": peso * t.r})
    return pd.DataFrame(tomados, columns=[*trades.columns, "peso", "r_cartera"])


def metricas(tomados: pd.DataFrame) -> dict:
    """Resultado de la cartera en R; el drawdown, sobre la curva ordenada por la salida de cada trade."""
    r = tomados.sort_values("salida")["r_cartera"].to_numpy()
    resumen = tm.resumen(r)
    if not resumen["trades"]:
        return {"trades": 0, "total_R": 0.0, "esperanza_R": 0.0, "t": 0.0, "max_drawdown_R": 0.0, "total_sobre_drawdown": 0.0}
    drawdown = resumen["max_drawdown_R"]
    return {"trades": resumen["trades"], "total_R": round(resumen["total_R"], 2), "esperanza_R": round(resumen["esperanza_R"], 4),
            "t": round(resumen["t_stat"], 2), "max_drawdown_R": round(drawdown, 2),
            "total_sobre_drawdown": round(resumen["total_R"] / drawdown, 2) if drawdown > 0 else 0.0}


def comparar(trades: pd.DataFrame) -> list[dict]:
    """Una fila por regla y semilla."""
    filas = []
    for semilla, de_la_semilla in trades.groupby("semilla"):
        for regla, opciones in REGLAS.items():
            tomados = simular_cartera(de_la_semilla, **{opcion: True for opcion in opciones})
            filas.append({"fecha": datetime.now().isoformat(timespec="seconds"), "regla": regla, "semilla": int(semilla), **metricas(tomados)})
    return filas


def tabla(filas: list[dict]) -> str:
    lineas = [f"{'regla':<13}{'trades':>8}{'total R':>10}{'t por semilla':>24}{'drawdown máx. R':>18}{'total / drawdown':>18}"]
    for regla in REGLAS:
        propias = [f for f in filas if f["regla"] == regla]
        media = lambda campo: statistics.fmean(f[campo] for f in propias)  # noqa: E731
        ts = " / ".join(f"{f['t']:+.2f}" for f in propias)
        lineas.append(f"{regla:<13}{media('trades'):>8.0f}{media('total_R'):>+10.1f}{ts:>24}{media('max_drawdown_R'):>18.1f}"
                      f"{media('total_sobre_drawdown'):>+18.2f}")
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repetir", action="store_true", help="vuelve a generar los trades del walk-forward")
    args = parser.parse_args()
    if TRADES.exists() and not args.repetir:
        trades = pd.read_pickle(TRADES)  # archivo propio, escrito aquí abajo
    else:
        paquetes = ex.cargar(ACTIVOS)
        partes = []
        for simbolo, paquete in paquetes.items():
            for semilla in ex.SEMILLAS:
                partes.append(trades_de(simbolo, paquete, semilla))
                print(f"{simbolo} semilla {semilla}: {len(partes[-1])} trades", flush=True)
        trades = pd.concat(partes, ignore_index=True)
        with open(TRADES, "wb") as f:
            pickle.dump(trades, f)
    filas = comparar(trades)
    with open(RESULTADOS, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)
    print(tabla(filas))


if __name__ == "__main__":
    main()
