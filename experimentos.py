"""Banco de experimentos: evalúa variantes del modelo con el walk-forward de train_model.py, sobre velas de Exness, y
apunta cada resultado en experimentos/registro.csv.

Uso: python experimentos.py --descargar                    guarda las velas de cada activo (solo lee de MT5)
     python experimentos.py [VARIANTE ...] [--simbolos X Y]  evalúa (sin variantes, todas); lo ya registrado no se repite
     python experimentos.py --tabla                         compara cada variante con la base

Cada variante se fija en VARIANTES antes de ver su resultado (protocolo de Plan_de_mejoras.md). El walk-forward usa
solo las velas anteriores a HOLDOUT_DESDE: el banco nunca evalúa el holdout, que queda para decidir un despliegue.
No envía órdenes ni toca los modelos de models/.
"""
import argparse
import csv
import ctypes
import ctypes.wintypes
import pickle
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
import train_model as tm
from activos import ACTIVOS, hora_modelo
from features import FEATURES, adx_wilder, atr_wilder, build_features

CARPETA = config.BASE_DIR / "experimentos"
DATOS = CARPETA / "datos"  # velas descargadas; no van al repositorio
REGISTRO = CARPETA / "registro.csv"
HOLDOUT_DESDE = "2026-01-01"
SEMILLAS = config.SEMILLAS_ROBUSTEZ
CAMPOS = ["fecha", "variante", "simbolo", "semilla", "features", "filas", "trades", "esperanza_R", "t", "profit_factor",
          "ventanas_positivas", "auc_long", "auc_short"]
BASE = "base"
SESIONES = {"asia": ("Asia/Tokyo", "09:00", "15:00"), "londres": ("Europe/London", "08:00", "16:30"),
            "ny": ("America/New_York", "09:30", "16:00")}
ADX_TENDENCIA = 0.25  # ADX > 25, fijado de antemano
VENTANA_DESLIZANTE_SEGMENTOS = 2  # con 100.000 velas y 8 segmentos, unos 12 meses de entrenamiento
ANTIGUEDAD_MAX_S = 86400  # un dato de otro activo más viejo que esto no se usa


# ---------------- features de cada variante (la fila t solo usa lo conocido al cierre de la vela t) ----------------
def _minuto(hora: str) -> int:
    return int(hora[:2]) * 60 + int(hora[3:])


def sesiones(df: pd.DataFrame, contexto: dict) -> pd.DataFrame:
    """Banderas de sesión por la hora local de cada plaza en la apertura de la vela, con sus cambios de horario."""
    utc = pd.to_datetime(df["time_utc"].to_numpy(dtype=np.int64), unit="s", utc=True)
    for nombre, (zona, desde, hasta) in SESIONES.items():
        local = utc.tz_convert(zona)
        minuto = np.asarray(local.hour * 60 + local.minute)
        df[f"ses_{nombre}"] = ((minuto >= _minuto(desde)) & (minuto < _minuto(hasta)) & (np.asarray(local.day_of_week) < 5)).astype(float)
    df["ses_solape"] = df["ses_londres"] * df["ses_ny"]
    return df


def squeeze(df: pd.DataFrame, contexto: dict) -> pd.DataFrame:
    """Anchura de las Bandas de Bollinger(20, 2) entre la del Canal de Keltner(20, 1,5 × ATR), y si hay squeeze."""
    bollinger = 4 * df["close"].rolling(20).std()
    keltner = 3 * atr_wilder(df, 20)
    df["squeeze_ratio"] = bollinger / keltner
    df["squeeze_activo"] = (df["squeeze_ratio"] < 1).astype(float).where(df["squeeze_ratio"].notna())
    return df


def regimen(df: pd.DataFrame, contexto: dict) -> pd.DataFrame:
    """Régimen por reglas: +1 tendencia alcista, -1 bajista (ADX > 25 y el DI dominante), 0 rango."""
    adx, plus_di, minus_di = adx_wilder(df)
    df["adx"] = adx
    df["regimen"] = np.where(adx > ADX_TENDENCIA, np.sign(plus_di - minus_di), 0.0)
    return df


def otros_activos(df: pd.DataFrame, contexto: dict) -> pd.DataFrame:
    """Rentabilidad a 1 h y 4 h de los demás activos, tomada de su última vela que abrió a la vez o antes que la vela t
    (misma rejilla de M15: cierra a la vez o antes). Sin dato reciente del otro activo, 0."""
    izquierda = pd.DataFrame({"time_utc": df["time_utc"].to_numpy(dtype=np.int64)})
    for simbolo, otro in contexto["otros"].items():
        derecha = pd.DataFrame({"time_utc": otro["time"].to_numpy(dtype=np.int64),
                                f"rel_{simbolo}_1h": otro["close"].pct_change(4).to_numpy(),
                                f"rel_{simbolo}_4h": otro["close"].pct_change(16).to_numpy()})
        unido = pd.merge_asof(izquierda, derecha, on="time_utc", direction="backward", tolerance=ANTIGUEDAD_MAX_S)
        for columna in derecha.columns[1:]:
            df[columna] = unido[columna].fillna(0.0).to_numpy()
    return df


def _sin_cambios(df: pd.DataFrame, contexto: dict) -> pd.DataFrame:
    return df


@dataclass(frozen=True)
class Variante:
    descripcion: str
    extra: object = _sin_cambios  # (features base con `time_utc`, contexto) -> el mismo DataFrame con columnas nuevas
    features: object = ()  # columnas que se añaden a las 11 de la base; una función las da según el contexto
    filtro: object = None  # filas en las que se permite entrar (como el filtro de spread: ni entrenan ni cuentan)
    segmentos_train: int | None = None  # None = anclado; n = solo los últimos n segmentos (ventana deslizante)

    def columnas(self, contexto: dict) -> list:
        return list(self.features(contexto) if callable(self.features) else self.features)


VARIANTES = {
    BASE: Variante("Las 11 features actuales, walk-forward anclado"),
    "sesiones": Variante("2.1 Banderas de sesión: Asia, Londres, Nueva York y solape", sesiones,
                         ("ses_asia", "ses_londres", "ses_ny", "ses_solape")),
    "squeeze": Variante("2.2 Squeeze: Bollinger(20, 2) / Keltner(20, 1,5 ATR) y bandera", squeeze,
                        ("squeeze_ratio", "squeeze_activo")),
    "regimen_feature": Variante("3.3a Régimen por reglas (ADX > 25 y DI) como feature", regimen, ("regimen", "adx")),
    "regimen_filtro": Variante("3.3b Sin entradas en rango (ADX <= 25)", regimen, (), lambda datos: datos["adx"] > ADX_TENDENCIA),
    "otros_activos": Variante("5.4 Rentabilidad a 1 h y 4 h de los otros cuatro activos", otros_activos,
                              lambda contexto: [f"rel_{s}_{h}" for s in contexto["otros"] for h in ("1h", "4h")]),
    "deslizante": Variante("4.3 Entrenamiento con ventana deslizante de unos 12 meses", segmentos_train=VENTANA_DESLIZANTE_SEGMENTOS),
}


# ---------------- evaluación ----------------
@contextmanager
def con_variante(variante: Variante, contexto: dict):
    """Aplica la variante a train_model mientras dura el bloque y lo deja todo como estaba al salir."""
    originales = list(FEATURES), tm.build_features, tm.ventanas_walk_forward

    def features_con_extra(velas: pd.DataFrame, point: float) -> pd.DataFrame:
        return variante.extra(build_features(velas, point), contexto)

    def ventanas_deslizantes(datos: pd.DataFrame, n_ventanas: int):
        maximo = variante.segmentos_train * (len(datos) // (n_ventanas + 3))
        for train, val, test in originales[2](datos, n_ventanas):
            yield train[-maximo:], val, test

    FEATURES[:] = originales[0] + variante.columnas(contexto)  # la misma lista que usa train_model
    tm.build_features = features_con_extra
    if variante.segmentos_train is not None:
        tm.ventanas_walk_forward = ventanas_deslizantes
    try:
        yield
    finally:
        FEATURES[:] = originales[0]
        tm.build_features, tm.ventanas_walk_forward = originales[1], originales[2]


def preparar(variante: Variante, velas: pd.DataFrame, point: float, costes: dict, contexto: dict) -> pd.DataFrame:
    """Las filas del walk-forward (anteriores al holdout) con las features de la variante. Dentro de con_variante."""
    velas = velas.assign(time_utc=velas["time"], time=hora_modelo(velas["time"]))
    datos, _ = tm.separar_holdout(tm.preparar_datos(velas, point, costes))
    return datos if variante.filtro is None else datos[variante.filtro(datos)]


def _auc_medio(ventanas: list, lado: str) -> float | None:
    valores = [v["lados"][lado]["auc_test"] for v in ventanas if v["lados"][lado]["auc_test"] is not None]
    return round(statistics.fmean(valores), 4) if valores else None


def evaluar(nombre: str, simbolo: str, paquete: dict, contexto: dict, semillas=SEMILLAS):
    """Una fila del registro por semilla. `paquete` es lo que guarda --descargar para ese símbolo."""
    variante, velas = VARIANTES[nombre], paquete["velas"]
    mediana = float(velas["spread"].median())
    ajustes = {"SYMBOL": simbolo, "MAX_SPREAD_PUNTOS": mediana * 25 / 8, "SLIPPAGE_PUNTOS": mediana / 8, "COMISION_PUNTOS": 0,
               "HOLDOUT_DESDE": HOLDOUT_DESDE}  # como train_exness.py, salvo el holdout
    originales = {clave: getattr(config, clave) for clave in ajustes}
    for clave, valor in ajustes.items():
        setattr(config, clave, valor)
    try:
        with con_variante(variante, contexto):
            datos = preparar(variante, velas, paquete["point"], paquete["costes"], contexto)
            for semilla in semillas:
                ventanas = tm.walk_forward(datos, semilla)
                wf = tm._resumen_walk_forward(ventanas)
                agregado = wf["agregado"]
                yield {"fecha": datetime.now().isoformat(timespec="seconds"), "variante": nombre, "simbolo": simbolo,
                       "semilla": semilla, "features": len(FEATURES), "filas": len(datos), "trades": agregado["trades"],
                       "esperanza_R": round(agregado.get("esperanza_R", 0.0), 4), "t": round(agregado.get("t_stat", 0.0), 2),
                       "profit_factor": round(min(agregado.get("profit_factor", 0.0), 99.0), 3),
                       "ventanas_positivas": round(tm._proporcion_positivas(wf), 2),
                       "auc_long": _auc_medio(ventanas, "long"), "auc_short": _auc_medio(ventanas, "short")}
    finally:
        for clave, valor in originales.items():
            setattr(config, clave, valor)


# ---------------- registro ----------------
def leer_registro(ruta: Path = REGISTRO) -> list[dict]:
    try:
        with open(ruta, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def apuntar(fila: dict, ruta: Path = REGISTRO) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            escritor.writeheader()
        escritor.writerow(fila)


def tabla(filas: list[dict]) -> str:
    """Por variante y activo, el t de cada semilla y su media frente a la base; la última fila de cada (variante,
    activo, semilla) es la que cuenta."""
    ultimo = {(f["variante"], f["simbolo"], f["semilla"]): f for f in filas}
    variantes = list(dict.fromkeys(f["variante"] for f in filas))
    simbolos = list(dict.fromkeys(f["simbolo"] for f in filas))

    def de(variante: str, simbolo: str) -> list:
        return [f for (v, s, _), f in ultimo.items() if v == variante and s == simbolo]

    lineas = [f"{'variante':<17}{'activo':<9}{'t por semilla':<22}{'t medio':>8}{'vs base':>9}{'trades':>8}{'AUC L/S':>14}{'t>=2':>6}"]
    for variante in variantes:
        mejoran = 0
        for simbolo in simbolos:
            propias, base = de(variante, simbolo), de(BASE, simbolo)
            if not propias:
                continue
            ts = [float(f["t"]) for f in propias]
            medio = statistics.fmean(ts)
            diferencia = medio - statistics.fmean(float(f["t"]) for f in base) if base else None
            mejoran += bool(diferencia and diferencia > 0)
            aucs = "/".join(f"{statistics.fmean(float(f[c]) for f in propias if f[c]):.3f}" if any(f[c] for f in propias) else "-"
                            for c in ("auc_long", "auc_short"))
            lineas.append(f"{variante:<17}{simbolo:<9}{' / '.join(f'{t:+.2f}' for t in ts):<22}{medio:>+8.2f}"
                          f"{'' if diferencia is None or variante == BASE else f'{diferencia:+.2f}':>9}"
                          f"{round(statistics.fmean(int(f['trades']) for f in propias)):>8}{aucs:>14}"
                          f"{'sí' if len(ts) >= len(SEMILLAS) and min(ts) >= config.T_MINIMO_TEST else 'no':>6}")
        if variante != BASE:
            lineas.append(f"{'':<17}mejora el t medio de la base en {mejoran} de {sum(bool(de(variante, s)) for s in simbolos)} activos")
    return "\n".join(lineas)


# ---------------- datos ----------------
def descargar(simbolos) -> None:
    """Guarda las velas M15, el punto y los costes de cada símbolo. Solo lee de la terminal."""
    import MetaTrader5 as mt5

    import train_exness
    from cuenta import CUENTA

    if not mt5.initialize(CUENTA.terminal):
        raise SystemExit(f"No se pudo conectar con MT5: {mt5.last_error()}")
    try:
        DATOS.mkdir(parents=True, exist_ok=True)
        for simbolo in simbolos:
            velas, info = train_exness.descargar(simbolo), mt5.symbol_info(simbolo)
            with open(DATOS / f"{simbolo}.pkl", "wb") as f:
                pickle.dump({"velas": velas, "point": info.point, "costes": tm.costes_de_simbolo(info)}, f)
            print(f"{simbolo}: {len(velas)} velas hasta {tm.fecha(velas['time'].iloc[-1])}", flush=True)
    finally:
        mt5.shutdown()


def cargar(simbolos) -> dict:
    paquetes = {}
    for simbolo in simbolos:
        try:
            with open(DATOS / f"{simbolo}.pkl", "rb") as f:
                paquetes[simbolo] = pickle.load(f)  # archivo propio, escrito por --descargar
        except OSError:
            raise SystemExit(f"Faltan las velas de {simbolo}: python experimentos.py --descargar") from None
    return paquetes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variantes", nargs="*", help=f"entre: {', '.join(VARIANTES)}")
    parser.add_argument("--simbolos", nargs="+", default=list(ACTIVOS), choices=list(ACTIVOS))
    parser.add_argument("--descargar", action="store_true", help="guarda las velas de cada activo y termina")
    parser.add_argument("--tabla", action="store_true", help="muestra la comparación del registro y termina")
    parser.add_argument("--repetir", action="store_true", help="evalúa también lo que ya está en el registro")
    args = parser.parse_args()
    if desconocidas := set(args.variantes) - set(VARIANTES):
        parser.error(f"variantes no válidas: {sorted(desconocidas)}")
    if args.descargar:
        return descargar(ACTIVOS)
    if args.tabla:
        return print(tabla(leer_registro()))

    ctypes.windll.kernel32.SetPriorityClass(ctypes.wintypes.HANDLE(-1), 0x4000)  # por debajo de lo normal: los bots van primero
    paquetes = cargar(ACTIVOS)  # todos: una variante puede mirar los demás activos
    hechos = set() if args.repetir else {(f["variante"], f["simbolo"], int(f["semilla"])) for f in leer_registro()}
    for nombre in args.variantes or VARIANTES:
        for simbolo in args.simbolos:
            semillas = [s for s in SEMILLAS if (nombre, simbolo, s) not in hechos]
            contexto = {"otros": {s: p["velas"] for s, p in paquetes.items() if s != simbolo}}
            for fila in evaluar(nombre, simbolo, paquetes[simbolo], contexto, semillas):
                apuntar(fila)
                print(f"{nombre:<17}{simbolo:<9}semilla {fila['semilla']:>2}: {fila['trades']} trades, "
                      f"{fila['esperanza_R']:+.3f} R, t = {fila['t']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
