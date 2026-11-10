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

import calendario
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


def noticias(df: pd.DataFrame, contexto: dict) -> pd.DataFrame:
    """Cercanía a las noticias de alto impacto de las monedas del activo, en el instante de la entrada (el cierre de
    la vela t). La hora de una noticia programada se conoce de antemano, así que mirar la próxima no es una fuga."""
    entrada = df["time_utc"].to_numpy(dtype=np.int64) + config.TIMEFRAME_SEGUNDOS
    hasta, desde = calendario.minutos_a_la_noticia(contexto["noticias"], entrada)
    df["noticia_hasta"], df["noticia_desde"] = hasta / 1440, desde / 1440
    df["noticia_cerca"] = calendario.cerca_de_noticia(contexto["noticias"], entrada, calendario.MINUTOS_ANTES * 60,
                                                      calendario.MINUTOS_DESPUES * 60).astype(float)
    return df


# ---------------- gestión de la operación (cambia cómo sale cada trade, no la señal) ----------------
FIJA, BREAKEVEN, TRAILING, PARCIAL = "fija", "breakeven", "trailing", "parcial"
ACTIVACION_R = 1.0  # a +1 R a favor se mueve el stop a la entrada (breakeven) o se cierra la mitad (parcial)


def _salida_gestionada(apertura, minimo, maximo, e: int, cierre: int, entrada: float, sl_dist: float, tp_dist: float,
                       gestion: str) -> tuple:
    """(resultado bruto en precio por unidad, última vela con el trade abierto, si salió por tiempo, ejecuciones extra).

    Una compra vista con los precios a los que saldría (una venta se pasa cambiada de signo). El stop solo se mueve
    al cerrar una vela, con lo visto hasta ella, y vale desde la siguiente: es lo que puede hacer el bot, que actúa
    en cada vela nueva. Si la apertura siguiente ya está por debajo del stop nuevo, el broker lo rechazaría y se
    conserva el anterior.
    """
    sl, tp, mejor = entrada - sl_dist, entrada + tp_dist, -np.inf
    abierto, realizado, extra = 1.0, 0.0, 0.0
    for j in range(e, cierre):
        if minimo[j] <= sl:
            return realizado + abierto * (min(apertura[j], sl) - entrada), j, False, extra
        if maximo[j] >= tp:
            return realizado + abierto * (tp - entrada), j, False, extra
        if gestion == FIJA or j + 1 >= cierre:
            continue
        mejor = max(mejor, maximo[j])
        activada = mejor >= entrada + ACTIVACION_R * sl_dist
        if gestion == PARCIAL and activada and abierto == 1.0:
            realizado, abierto, extra = 0.5 * (apertura[j + 1] - entrada), 0.5, 0.5  # media posición, a mercado
        nuevo = mejor - sl_dist if gestion == TRAILING else entrada if activada else sl
        if nuevo > sl and apertura[j + 1] > nuevo:
            sl = nuevo
    return realizado + abierto * (apertura[cierre] - entrada), cierre - 1, True, extra


def etiquetar_con_gestion(gestion: str, etiquetar=tm.etiquetar_triple_barrera):
    """`etiquetar_triple_barrera` con r, k y to recalculados para esa gestión. `y` (lo que aprende el modelo) y la
    purga no cambian: el stop solo se acerca, así que ningún trade dura más que con el stop y el objetivo fijos."""
    def etiquetar_gestionado(df, point, sl_mult, tp_mult, n, comision_puntos=0.0, slippage_puntos=0.0,
                             swap_long_puntos=0.0, swap_short_puntos=0.0, swap_3dias=3):
        etiquetas = etiquetar(df, point, sl_mult, tp_mult, n, comision_puntos, slippage_puntos, swap_long_puntos,
                              swap_short_puntos, swap_3dias)
        o, h, l = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low"))
        spr = df["spread"].to_numpy(dtype=float) * point
        atr, tiempos = df["atr"].to_numpy(dtype=float), df["time"].to_numpy(dtype=np.int64)
        # La compra sale al bid; la venta, al ask, y cambiada de signo se trata como una compra.
        vistas = {"long": (o, l, h, o + spr, swap_long_puntos), "short": (-(o + spr), -(h + spr), -(l + spr), -o, swap_short_puntos)}
        for lado, (apertura, minimo, maximo, entradas, swap) in vistas.items():
            r, k, to = (etiquetas[f"{c}_{lado}"].to_numpy(copy=True) for c in ("r", "k", "to"))
            for t in np.flatnonzero(np.isfinite(r)):
                e, sl_dist = t + 1, sl_mult * atr[t]
                bruto, ultima, por_tiempo, extra = _salida_gestionada(apertura, minimo, maximo, e, e + n, entradas[e], sl_dist,
                                                                      tp_mult * atr[t], gestion)
                noches = tm.noches_de_swap(tiempos[e], tiempos[e + n if por_tiempo else ultima], swap_3dias)
                costes = (comision_puntos + (2 + extra) * slippage_puntos) * point
                r[t], k[t], to[t] = (bruto + noches * swap * point - costes) / sl_dist, ultima, por_tiempo
            etiquetas[f"r_{lado}"], etiquetas[f"k_{lado}"], etiquetas[f"to_{lado}"] = r, k, to
        return etiquetas
    return etiquetar_gestionado


# ---------------- otros modelos sobre las mismas features (hiperparámetros fijados de antemano) ----------------
def boosting(semilla: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=50,
                                          l2_regularization=1.0, random_state=semilla)


def logistica(semilla: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=semilla))


def bosque(arboles: int, profundidad: int, hoja: int):
    def nuevo(semilla: int):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=arboles, max_depth=profundidad, min_samples_split=2 * hoja,
                                      min_samples_leaf=hoja, max_features="sqrt", random_state=semilla, n_jobs=-1)
    return nuevo


def _sin_cambios(df: pd.DataFrame, contexto: dict) -> pd.DataFrame:
    return df


@dataclass(frozen=True)
class Variante:
    descripcion: str
    extra: object = _sin_cambios  # (features base con `time_utc`, contexto) -> el mismo DataFrame con columnas nuevas
    features: object = ()  # columnas que se añaden a las 11 de la base; una función las da según el contexto
    filtro: object = None  # filas en las que se permite entrar (como el filtro de spread: ni entrenan ni cuentan)
    segmentos_train: int | None = None  # None = anclado; n = solo los últimos n segmentos (ventana deslizante)
    gestion: str | None = None  # cómo sale cada trade: None = stop y objetivo fijos
    modelo: object = None  # semilla -> clasificador con predict_proba; None = el Random Forest de train_model

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
    "noticias_filtro": Variante("1.4 Sin entradas de 15 min antes a 15 min después de noticias de alto impacto", noticias, (),
                                lambda datos: datos["noticia_cerca"] == 0),
    "noticias_feature": Variante("5.4 Tiempo hasta la próxima noticia de alto impacto y desde la última", noticias,
                                 ("noticia_hasta", "noticia_desde")),
    "breakeven": Variante("M2 Stop a la entrada cuando el precio llega a +1 R", gestion=BREAKEVEN),
    "trailing": Variante("M2 Stop dinámico a 1,5 ATR del mejor precio alcanzado", gestion=TRAILING),
    "parcial": Variante("M2 Media posición cerrada a +1 R y stop a la entrada para el resto", gestion=PARCIAL),
    "boosting": Variante("M Gradient boosting por histogramas (200 árboles de profundidad 3, tasa 0,05)", modelo=boosting),
    "logistica": Variante("M Regresión logística con las features normalizadas (C = 0,1)", modelo=logistica),
    "bosque_500": Variante("M El mismo Random Forest con 500 árboles en vez de 150", modelo=bosque(500, 5, 25)),
    "bosque_profundo": Variante("M Random Forest de 500 árboles, profundidad 10 y hojas de 50", modelo=bosque(500, 10, 50)),
    "deslizante": Variante("4.3 Entrenamiento con ventana deslizante de unos 12 meses", segmentos_train=VENTANA_DESLIZANTE_SEGMENTOS),
}


# ---------------- evaluación ----------------
@contextmanager
def con_variante(variante: Variante, contexto: dict):
    """Aplica la variante a train_model mientras dura el bloque y lo deja todo como estaba al salir."""
    originales = list(FEATURES), tm.build_features, tm.ventanas_walk_forward, tm.etiquetar_triple_barrera, tm.nuevo_modelo

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
    if variante.gestion is not None:
        tm.etiquetar_triple_barrera = etiquetar_con_gestion(variante.gestion, originales[3])
    if variante.modelo is not None:
        tm.nuevo_modelo = variante.modelo
    try:
        yield
    finally:
        FEATURES[:] = originales[0]
        tm.build_features, tm.ventanas_walk_forward, tm.etiquetar_triple_barrera, tm.nuevo_modelo = originales[1:]


def preparar(variante: Variante, velas: pd.DataFrame, point: float, costes: dict, contexto: dict) -> pd.DataFrame:
    """Las filas del walk-forward (anteriores al holdout) con las features de la variante. Dentro de con_variante."""
    velas = velas.assign(time_utc=velas["time"], time=hora_modelo(velas["time"]))
    datos, _ = tm.separar_holdout(tm.preparar_datos(velas, point, costes))
    return datos if variante.filtro is None else datos[variante.filtro(datos)]


def _auc_medio(ventanas: list, lado: str) -> float | None:
    valores = [v["lados"][lado]["auc_test"] for v in ventanas if v["lados"][lado]["auc_test"] is not None]
    return round(statistics.fmean(valores), 4) if valores else None


@contextmanager
def con_ajustes(simbolo: str, velas: pd.DataFrame):
    """Los costes y filtros de ese activo en config, como en train_exness.py salvo el holdout, mientras dura el bloque."""
    mediana = float(velas["spread"].median())
    ajustes = {"SYMBOL": simbolo, "MAX_SPREAD_PUNTOS": mediana * 25 / 8, "SLIPPAGE_PUNTOS": mediana / 8, "COMISION_PUNTOS": 0,
               "HOLDOUT_DESDE": HOLDOUT_DESDE}
    originales = {clave: getattr(config, clave) for clave in ajustes}
    for clave, valor in ajustes.items():
        setattr(config, clave, valor)
    try:
        yield
    finally:
        for clave, valor in originales.items():
            setattr(config, clave, valor)


def evaluar(nombre: str, simbolo: str, paquete: dict, contexto: dict, semillas=SEMILLAS):
    """Una fila del registro por semilla. `paquete` es lo que guarda --descargar para ese símbolo."""
    variante, velas = VARIANTES[nombre], paquete["velas"]
    with con_ajustes(simbolo, velas), con_variante(variante, contexto):
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
    try:
        eventos = calendario.leer(calendario.RUTA)[1]
    except (OSError, ValueError):
        eventos = None  # sin el servicio RF_Calendario, las variantes de noticias no se pueden evaluar
    hechos = set() if args.repetir else {(f["variante"], f["simbolo"], int(f["semilla"])) for f in leer_registro()}
    for nombre in args.variantes or VARIANTES:
        for simbolo in args.simbolos:
            semillas = [s for s in SEMILLAS if (nombre, simbolo, s) not in hechos]
            contexto = {"otros": {s: p["velas"] for s, p in paquetes.items() if s != simbolo}}
            if VARIANTES[nombre].extra is noticias:
                if eventos is None:
                    print(f"{nombre}: falta el calendario ({calendario.RUTA}); inicia el servicio RF_Calendario en MT5", flush=True)
                    break
                contexto["noticias"] = calendario.horas(eventos, calendario.MONEDAS[simbolo])
                inicio = int(paquetes[simbolo]["velas"]["time"].iloc[0])
                if len(contexto["noticias"]) == 0 or contexto["noticias"][0] > inicio + 30 * 86400:
                    print(f"{nombre}: el calendario no cubre el histórico de {simbolo}; sube DiasDeHistoria en el servicio", flush=True)
                    continue
            for fila in evaluar(nombre, simbolo, paquetes[simbolo], contexto, semillas):
                apuntar(fila)
                print(f"{nombre:<17}{simbolo:<9}semilla {fila['semilla']:>2}: {fila['trades']} trades, "
                      f"{fila['esperanza_R']:+.3f} R, t = {fila['t']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
