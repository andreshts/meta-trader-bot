"""Seguimiento de tendencia por reglas (canal de Donchian) en varios mercados: backtest y evaluación.

Las reglas, los periodos y el criterio están fijados de antemano en Plan_de_mejoras.md (Fase 6). No se entrena ni se
ajusta nada, así que todo el histórico es fuera de muestra; el holdout solo confirma.

Uso: python tendencia.py                  (6.1: 6 mercados)
     python tendencia.py --diversificada  (6.2: cartera diversificada por sectores)
"""
import argparse
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import config
from features import atr_wilder
from train_model import costes_de_simbolo, fecha, noches_de_swap, resumen

MERCADOS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "XAGUSD")
SEGUNDOS = {"H4": 14_400, "D1": 86_400}
INICIO = 100                  # primera vela operable: el canal más largo (100) y el ATR de Wilder ya están formados
CORTE_HOLDOUT = "2024-01-01"  # desarrollo antes, holdout desde aquí (fecha del servidor)
N_SEGMENTOS = 8
SWAP_PESIMISTA_ANUAL = 0.02   # swap en contra en los dos lados, como fracción anual del precio de entrada
COLUMNAS_TRADES = ["lado", "vela_senal", "vela_entrada", "vela_salida", "t_entrada", "t_fin", "r", "motivo"]

# 6.2: candidatos por sector, según la regla de Plan_de_mejoras.md. Entran solo los que pasan `problemas_de_datos`.
CANDIDATOS_DIVERSIFICADA = {
    "divisas": ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"),
    "indices": ("AUS200", "UK100", "ESP35", "EUSTX50", "FRA40", "HK50", "JPN225", "US500", "US30", "USTECH100M",
                "DE40", "CA60", "CHINA50", "CHINAH", "IT40", "NETH25", "NOR25", "SA40", "SE30", "SWI20", "US2000",
                "MIDDE50", "TECHDE30",   # sin US500M, US30M ni USTEC, versiones del mismo índice con menos histórico
                "SPY", "EFA", "EWJ", "EEM"),  # ETFs de la enmienda: ningún CFD de índice pasa el filtro de calidad
    "metales": ("XAUUSD", "XAGUSD", "XPDUSD", "XPTUSD", "ALUMINIUM", "COPPER", "LEAD", "NICKEL"),
    "bonos": ("TLT", "IEF"),
    "materias_primas": ("USO", "UNG", "DBA"),
}
INICIO_MAXIMO = "2010-02-01"  # el histórico D1 tiene que empezar a más tardar este día
MAX_HUECO_DIAS = 15           # entre velas consecutivas
MAX_SALTO = 0.40              # entre cierres consecutivos: un split sin ajustar o datos erróneos


def simular_donchian(velas: pd.DataFrame, point: float, n_entrada: int = 55, n_salida: int = 20,
                     stop_atr: float = 2.0, periodo_atr: int = 20, *, segundos_vela: int = 86_400,
                     slippage_puntos: float = 0.0, comision_puntos: float = 0.0,
                     swap_long_puntos: float = 0.0, swap_short_puntos: float = 0.0, swap_3dias: int = 3,
                     swap_anual_en_contra: float | None = None, inicio: int = INICIO):
    """Simula las reglas sobre velas bid de MT5 y devuelve `(trades, mtm)`.

    En cada vela `j`, en este orden:
      1. en la apertura se ejecutan las órdenes de las señales del cierre de `j-1` (primero la salida y luego la
         entrada, que puede ser al lado contrario);
      2. el stop se dispara si el bid (long) o el ask (short) lo toca dentro de la vela, al precio de la apertura si
         la vela abre más allá;
      3. la posición se marca a mercado al cierre;
      4. con el cierre de `j` se calculan las señales para la apertura de `j+1`.
    Los canales usan las velas anteriores a `j`, sin incluirla, así que ninguna decisión ve precios futuros.

    trades: una fila por operación cerrada, con `r` en múltiplos del riesgo inicial y neto de costes, `t_entrada`
            (apertura de la vela de entrada) y `t_fin` (instante del último precio del que depende).
    mtm:    cambio de valor en R de la posición en cada vela, indexado por `time`. Sumado sobre una operación da su
            `r`. Las posiciones abiertas al final no están en `trades`, pero sí en `mtm` hasta la última vela.
    Swap: cada rollover según `noches_de_swap`, con el del símbolo o, si se da `swap_anual_en_contra`, con esa
    fracción anual del precio de entrada en contra en los dos lados.
    """
    o, h, l, c = (velas[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close"))
    spr = velas["spread"].to_numpy(dtype=float) * point
    tiempos = velas["time"].to_numpy(dtype=np.int64)
    atr = atr_wilder(velas, periodo_atr).to_numpy()
    techo_entrada = velas["high"].rolling(n_entrada).max().shift(1).to_numpy()
    suelo_entrada = velas["low"].rolling(n_entrada).min().shift(1).to_numpy()
    techo_salida = velas["high"].rolling(n_salida).max().shift(1).to_numpy()
    suelo_salida = velas["low"].rolling(n_salida).min().shift(1).to_numpy()
    slip, com = slippage_puntos * point, comision_puntos * point

    mtm = np.zeros(len(velas))
    trades = []
    pos, salir, entrar = None, False, None

    def valor(precio, t_hasta):
        """R de la posición si se liquidara a `precio` (bid el long, ask el short) en el instante `t_hasta`."""
        signo = 1 if pos["lado"] == "long" else -1
        noches = noches_de_swap(tiempos[pos["vela_entrada"]], t_hasta, swap_3dias)
        return (signo * (precio - pos["entrada"]) + noches * pos["swap_noche"] - com) / pos["riesgo"]

    def cerrar(j, precio, t_swap, t_fin, motivo):
        nonlocal pos
        r = valor(precio, t_swap)
        mtm[j] += r - pos["valor"]
        trades.append([pos["lado"], pos["vela_senal"], pos["vela_entrada"], j,
                       tiempos[pos["vela_entrada"]], t_fin, r, motivo])
        pos = None

    for j in range(inicio, len(velas)):
        if pos is not None and salir:
            precio = o[j] - slip if pos["lado"] == "long" else o[j] + spr[j] + slip
            cerrar(j, precio, tiempos[j], tiempos[j], "canal")
        if entrar is not None and pos is None:
            lado, s = entrar
            entrada = o[j] + spr[j] + slip if lado == "long" else o[j] - slip
            riesgo = stop_atr * atr[s]
            if swap_anual_en_contra is not None:
                swap_noche = -swap_anual_en_contra * entrada / 365
            else:
                swap_noche = (swap_long_puntos if lado == "long" else swap_short_puntos) * point
            pos = {"lado": lado, "vela_senal": s, "vela_entrada": j, "entrada": entrada, "riesgo": riesgo,
                   "stop": entrada - riesgo if lado == "long" else entrada + riesgo,
                   "swap_noche": swap_noche, "valor": 0.0}
        salir, entrar = False, None

        if pos is not None:
            fin_vela = tiempos[j] + segundos_vela - 1
            if pos["lado"] == "long" and l[j] <= pos["stop"]:
                cerrar(j, min(o[j], pos["stop"]) - slip, tiempos[j], fin_vela, "stop")
            elif pos["lado"] == "short" and h[j] + spr[j] >= pos["stop"]:
                cerrar(j, max(o[j] + spr[j], pos["stop"]) + slip, tiempos[j], fin_vela, "stop")

        if pos is not None:
            v = valor(c[j] if pos["lado"] == "long" else c[j] + spr[j], tiempos[j] + segundos_vela)
            mtm[j] += v - pos["valor"]
            pos["valor"] = v
            salir = bool(c[j] < suelo_salida[j]) if pos["lado"] == "long" else bool(c[j] > techo_salida[j])

        lado = "long" if c[j] > techo_entrada[j] else "short" if c[j] < suelo_entrada[j] else None
        if lado is not None and atr[j] > 0 and (pos is None or (salir and lado != pos["lado"])):
            entrar = (lado, j)

    return pd.DataFrame(trades, columns=COLUMNAS_TRADES), pd.Series(mtm, index=tiempos)


# ---------------- evaluación ----------------
def segundos_de(dia: str) -> int:
    return int(datetime.strptime(dia, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def separar_por_corte(trades: pd.DataFrame, corte: int):
    """(desarrollo, holdout). El desarrollo solo tiene trades cuyos precios son todos anteriores al corte, y el holdout
    los que entran desde el corte. Los que lo cruzan no cuentan en ninguno de los dos."""
    return trades[trades["t_fin"] < corte], trades[trades["t_entrada"] >= corte]


def cartera_mensual(mtms: dict, desde: int) -> pd.Series:
    """Suma por mes (fecha del servidor) del marcado a mercado de todos los mercados, desde el mes de `desde`.
    Los meses sin posiciones valen 0."""
    meses = [pd.Series(m.to_numpy(), index=pd.to_datetime(m.index, unit="s")).resample("MS").sum()
             for m in mtms.values()]
    total = pd.concat(meses, axis=1).fillna(0.0).sum(axis=1)
    return total[total.index >= pd.Timestamp(desde, unit="s").to_period("M").to_timestamp()]


def resumen_mensual(meses: pd.Series) -> dict:
    x = meses.to_numpy(dtype=float)
    if len(x) < 2:
        return {"meses": len(x)}
    sd = x.std(ddof=1)
    curva = np.cumsum(x)
    return {
        "meses": len(x),
        "desde": str(meses.index[0].date()),
        "hasta": str(meses.index[-1].date()),
        "media_R": float(x.mean()),
        "t_mensual": float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0,
        "sharpe_anual": float(x.mean() / sd * np.sqrt(12)) if sd > 0 else 0.0,
        "meses_positivos": float((x > 0).mean()),
        "max_drawdown_R": float((np.maximum.accumulate(np.maximum(curva, 0)) - curva).max()),
        "total_R": float(curva[-1]),
    }


def segmentos(meses: pd.Series, n: int = N_SEGMENTOS) -> list[dict]:
    """Resultado de cartera en `n` tramos consecutivos de meses del mismo tamaño."""
    tramos = [meses.iloc[idx] for idx in np.array_split(np.arange(len(meses)), n)]
    return [{"desde": str(t.index[0].date()), "hasta": str(t.index[-1].date()), "total_R": float(t.sum())}
            for t in tramos if len(t)]


def problemas_de_datos(velas: pd.DataFrame) -> list[str]:
    """Motivos para excluir un mercado de la cartera diversificada (6.2), solo por la calidad de su histórico D1."""
    tiempos = velas["time"].to_numpy(dtype=np.int64)
    problemas = []
    if tiempos[0] > segundos_de(INICIO_MAXIMO):
        problemas.append(f"empieza el {fecha(tiempos[0])}")
    hueco = int(np.diff(tiempos).max() // 86400) if len(tiempos) > 1 else 0
    if hueco > MAX_HUECO_DIAS:
        problemas.append(f"hueco de {hueco} días")
    salto = float(velas["close"].pct_change().abs().max())
    if salto > MAX_SALTO:
        problemas.append(f"salto de {salto:.0%} entre cierres")
    return problemas


def velas_con_spread_anomalo(velas: pd.DataFrame, point: float, max_fraccion: float = 0.01) -> int:
    """Velas cuyo spread supera `max_fraccion` del precio: casi seguro un error del servidor (p. ej. IEF en 2024, con
    spreads de 58 dólares). No excluye el mercado, que no estaba en la regla de 6.2; el informe lo avisa."""
    return int((velas["spread"] * point > max_fraccion * velas["close"]).sum())


def pesos_por_sector(sectores: dict) -> dict:
    """Mismo riesgo total por sector, repartido a partes iguales entre sus mercados, con media 1 por mercado."""
    por_sector = Counter(sectores.values())
    return {m: len(sectores) / (len(por_sector) * por_sector[s]) for m, s in sectores.items()}


def evaluar(velas: dict, costes: dict, puntos: dict, timeframe: str, n_entrada: int = 55, n_salida: int = 20, *,
            swap_pesimista: bool = False, multiplicador_costes: float = 1.0, pesos: dict | None = None) -> dict:
    """Una variante sobre todos los mercados: trades y cartera mensual en desarrollo y holdout, y el detalle por
    mercado (sin ponderar). Cada mercado arriesga 1 R por operación, multiplicado por su peso si se da `pesos`."""
    corte = segundos_de(CORTE_HOLDOUT)
    mtms, desarrollo, holdout, por_mercado = {}, [], [], {}
    for simbolo, df in velas.items():
        cs = costes[simbolo]
        trades, mtm = simular_donchian(
            df.assign(spread=df["spread"] * multiplicador_costes), puntos[simbolo], n_entrada, n_salida,
            segundos_vela=SEGUNDOS[timeframe],
            slippage_puntos=config.SLIPPAGE_PUNTOS * multiplicador_costes,
            comision_puntos=config.COMISION_PUNTOS * multiplicador_costes,
            swap_long_puntos=cs["swap_long_puntos"], swap_short_puntos=cs["swap_short_puntos"],
            swap_3dias=cs["swap_3dias"], swap_anual_en_contra=SWAP_PESIMISTA_ANUAL if swap_pesimista else None)
        dev, hold = separar_por_corte(trades, corte)
        por_mercado[simbolo] = {"desarrollo": resumen(dev["r"].to_numpy()), "holdout": resumen(hold["r"].to_numpy())}
        peso = 1.0 if pesos is None else pesos[simbolo]
        mtms[simbolo] = mtm * peso
        desarrollo.append(dev.assign(r=dev["r"] * peso))
        holdout.append(hold.assign(r=hold["r"] * peso))

    desde = max(int(df["time"].iloc[INICIO]) for df in velas.values())
    meses = cartera_mensual(mtms, desde)
    corte_mes = pd.Timestamp(CORTE_HOLDOUT)
    meses_dev, meses_hold = meses[meses.index < corte_mes], meses[meses.index >= corte_mes]
    r_dev = pd.concat(desarrollo).sort_values("t_fin")["r"].to_numpy()
    r_hold = pd.concat(holdout).sort_values("t_fin")["r"].to_numpy()
    return {
        "variante": f"{timeframe} {n_entrada}/{n_salida}",
        "desarrollo": {"trades": resumen(r_dev), "mensual": resumen_mensual(meses_dev),
                       "segmentos": segmentos(meses_dev)},
        "holdout": {"trades": resumen(r_hold), "mensual": resumen_mensual(meses_hold)},
        "por_mercado": por_mercado,
    }


def proporcion_segmentos_positivos(ev: dict) -> float:
    return float(np.mean([s["total_R"] > 0 for s in ev["desarrollo"]["segmentos"]]))


def pasa_desarrollo(ev: dict) -> bool:
    t, m = ev["desarrollo"]["trades"], ev["desarrollo"]["mensual"]
    return (t.get("trades", 0) >= config.MIN_TRADES_TEST
            and t.get("t_stat", 0.0) >= config.T_MINIMO_TEST
            and m.get("t_mensual", 0.0) >= config.T_MINIMO_TEST
            and proporcion_segmentos_positivos(ev) >= config.WF_MIN_PROP_VENTANAS_POSITIVAS)


def veredicto(principal: dict, robustez: list[dict]) -> dict:
    """Criterio de la Fase 6: el principal pasa el desarrollo, todas las comprobaciones de robustez tienen esperanza
    > 0 en el desarrollo y el principal tiene esperanza > 0 en el holdout."""
    v = {
        "desarrollo": pasa_desarrollo(principal),
        "robustez": all(ev["desarrollo"]["trades"].get("esperanza_R", 0.0) > 0 for ev in robustez),
        "holdout": principal["holdout"]["trades"].get("esperanza_R", 0.0) > 0,
    }
    v["pasa"] = all(v.values())
    return v


# ---------------- datos ----------------
def descargar_mercado(simbolo: str, timeframe: str, desde: str, intentos: int = 20, espera_s: float = 3,
                      exigir_inicio: bool = True) -> pd.DataFrame:
    """Velas cerradas de `simbolo` desde `desde` (AAAA-MM-DD, hora del servidor), al día.

    La última vela que devuelve la terminal tiene que ser la que contiene el último tick (la vela en formación); si
    no, la caché está desfasada y se reintenta. Esa vela se descarta. Con `exigir_inicio`, se detiene si el histórico
    empieza más de una semana después de `desde` (la terminal lo recorta); sin él, quien llama decide qué hacer.
    """
    tf, segundos = getattr(mt5, f"TIMEFRAME_{timeframe}"), SEGUNDOS[timeframe]
    inicio = datetime.strptime(desde, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    velas, ultima, tick = None, None, None
    for _ in range(intentos):
        rates = mt5.copy_rates_range(simbolo, tf, inicio, datetime.now(timezone.utc) + timedelta(days=2))
        tick = mt5.symbol_info_tick(simbolo)
        if rates is not None and len(rates) and tick is not None:
            ultima = int(rates["time"][-1])
            if ultima <= tick.time < ultima + segundos:
                velas = pd.DataFrame(rates).iloc[:-1].reset_index(drop=True)
                break
        time.sleep(espera_s)
    else:
        raise RuntimeError(f"Las velas {timeframe} de {simbolo} no se sincronizaron: la última es {ultima} y el "
                           f"último tick {getattr(tick, 'time', None)}")
    if exigir_inicio and int(velas["time"].iloc[0]) > inicio.timestamp() + 7 * 86400:
        raise RuntimeError(f"El histórico {timeframe} de {simbolo} empieza el {fecha(velas['time'].iloc[0])}, no el "
                           f"{desde}: la terminal lo recorta. Pon 'Máx. barras en la ventana' en Ilimitado.")
    return velas


# ---------------- informe ----------------
def _texto_trades(bt: dict) -> str:
    if bt["trades"] == 0:
        return "sin operaciones"
    return (f"{bt['trades']} trades  acierto {bt['acierto']:.0%}  esperanza {bt['esperanza_R']:+.3f} R "
            f"± {bt['error_estandar_R']:.3f} (t = {bt['t_stat']:.2f})  PF {bt['profit_factor']:.2f}")


def _texto_mensual(m: dict) -> str:
    if m["meses"] < 2:
        return f"{m['meses']} meses"
    return (f"{m['meses']} meses ({m['desde'][:7]} -> {m['hasta'][:7]})  media {m['media_R']:+.2f} R/mes "
            f"(t = {m['t_mensual']:.2f})  Sharpe {m['sharpe_anual']:.2f}  meses+ {m['meses_positivos']:.0%}  "
            f"DD máx {m['max_drawdown_R']:.1f} R  total {m['total_R']:+.1f} R")


def imprimir_variante(nombre: str, ev: dict, detalle: bool = False) -> None:
    print(f"\n[{nombre}]")
    dev, hold = ev["desarrollo"], ev["holdout"]
    print(f"  Desarrollo, trades:   {_texto_trades(dev['trades'])}")
    print(f"  Desarrollo, cartera:  {_texto_mensual(dev['mensual'])}")
    signos = " ".join("+" if s["total_R"] > 0 else "-" for s in dev["segmentos"])
    print(f"  Segmentos ({N_SEGMENTOS}):        {signos}  ({proporcion_segmentos_positivos(ev):.0%} positivos)  "
          + "  ".join(f"{s['total_R']:+.1f}" for s in dev["segmentos"]))
    print(f"  Holdout, trades:      {_texto_trades(hold['trades'])}")
    print(f"  Holdout, cartera:     {_texto_mensual(hold['mensual'])}")
    if detalle:
        print("  Por mercado (desarrollo | holdout):")
        for simbolo, r in ev["por_mercado"].items():
            d, h = r["desarrollo"], r["holdout"]
            texto_h = f"{h['trades']} trades {h['esperanza_R']:+.3f} R" if h["trades"] else "sin operaciones"
            print(f"    {simbolo:<10} {_texto_trades(d)}  |  {texto_h}")


def descargar_universo(simbolos, timeframes, tolerante: bool = False):
    """(velas por timeframe y símbolo, costes, puntos, fallos). Con `tolerante`, un símbolo que no existe o no se
    sincroniza queda en `fallos` en vez de detener todo, y no se exige que el histórico empiece en HISTORICO_DESDE.
    Los símbolos que no estaban en la Observación del mercado se vuelven a ocultar al terminar."""
    import mt5_utils

    if not mt5_utils.conectar():
        raise SystemExit(f"No se pudo conectar con MT5: {mt5.last_error()}")
    velas, costes, puntos, fallos, ocultos = {tf: {} for tf in timeframes}, {}, {}, {}, []
    try:
        for simbolo in simbolos:
            try:
                info = mt5.symbol_info(simbolo)
                if info is None:
                    raise RuntimeError("no existe en el servidor")
                if not info.visible:
                    ocultos.append(simbolo)
                if not mt5.symbol_select(simbolo, True):
                    raise RuntimeError(f"no se pudo seleccionar: {mt5.last_error()}")
                info = mt5.symbol_info(simbolo)
                descargas = {tf: descargar_mercado(simbolo, tf, config.HISTORICO_DESDE, exigir_inicio=not tolerante)
                             for tf in timeframes}
            except RuntimeError as e:
                if not tolerante:
                    raise SystemExit(f"{simbolo}: {e}")
                fallos[simbolo] = str(e)
                continue
            costes[simbolo], puntos[simbolo] = costes_de_simbolo(info), info.point
            for tf, df in descargas.items():
                velas[tf][simbolo] = df
    finally:
        for simbolo in ocultos:
            mt5.symbol_select(simbolo, False)
        mt5.shutdown()
    return velas, costes, puntos, fallos


def imprimir_datos(velas_d1: dict, costes: dict) -> None:
    for simbolo, d1 in velas_d1.items():
        cs = costes[simbolo]
        print(f"{simbolo:<10} D1 {len(d1)} velas ({fecha(d1['time'].iloc[0])} -> {fecha(d1['time'].iloc[-1])}) | "
              f"swap long/short {cs['swap_long_puntos']}/{cs['swap_short_puntos']} pts, "
              f"spread mediano {d1['spread'].median():.0f} pts")
    print(f"Desarrollo hasta {CORTE_HOLDOUT}; holdout desde {CORTE_HOLDOUT}. Costes: deslizamiento "
          f"{config.SLIPPAGE_PUNTOS} pts por ejecución, comisión {config.COMISION_PUNTOS} pts.")


def imprimir_veredicto(principal: dict, robustez: dict) -> None:
    v = veredicto(principal, list(robustez.values()))
    print(f"\nCriterio de la Fase 6: desarrollo {'pasa' if v['desarrollo'] else 'no pasa'} | robustez "
          f"{'pasa' if v['robustez'] else 'no pasa'} | holdout {'pasa' if v['holdout'] else 'no pasa'}")
    print(f"VEREDICTO: {'ventaja significativa y robusta' if v['pasa'] else 'SIN ventaja demostrada'}")


def informe_seis_mercados() -> None:
    """Fase 6.1."""
    velas, costes, puntos, _ = descargar_universo(MERCADOS, ("D1", "H4"))
    imprimir_datos(velas["D1"], costes)

    principal = evaluar(velas["D1"], costes, puntos, "D1", 55, 20)
    robustez = {
        "D1 20/10 (vecino rápido)": evaluar(velas["D1"], costes, puntos, "D1", 20, 10),
        "D1 100/50 (vecino lento)": evaluar(velas["D1"], costes, puntos, "D1", 100, 50),
        "D1 55/20 con swap pesimista": evaluar(velas["D1"], costes, puntos, "D1", 55, 20, swap_pesimista=True),
    }
    informativas = {
        "D1 55/20 con costes x3": evaluar(velas["D1"], costes, puntos, "D1", 55, 20, multiplicador_costes=3),
        "H4 55/20": evaluar(velas["H4"], costes, puntos, "H4", 55, 20),
    }

    imprimir_variante("PRINCIPAL: D1 55/20", principal, detalle=True)
    for nombre, ev in robustez.items():
        imprimir_variante(f"ROBUSTEZ: {nombre}", ev)
    for nombre, ev in informativas.items():
        imprimir_variante(f"INFORMATIVA: {nombre}", ev, detalle=nombre.startswith("H4"))
    imprimir_veredicto(principal, robustez)


def informe_diversificado() -> None:
    """Fase 6.2: los candidatos que pasan el filtro de calidad, con el mismo riesgo por sector."""
    candidatos = {m: sector for sector, mercados in CANDIDATOS_DIVERSIFICADA.items() for m in mercados}
    velas, costes, puntos, fallos = descargar_universo(list(candidatos), ("D1",), tolerante=True)
    excluidos = {**{m: [motivo] for m, motivo in fallos.items()},
                 **{m: p for m, df in velas["D1"].items() if (p := problemas_de_datos(df))}}
    print("Excluidos por la calidad de los datos:")
    for m, motivos in excluidos.items():
        print(f"  {m:<10} {candidatos[m]:<16} {'; '.join(motivos)}")

    sectores = {m: s for m, s in candidatos.items() if m not in excluidos}
    pesos = pesos_por_sector(sectores)
    print("\nCartera:")
    for sector in CANDIDATOS_DIVERSIFICADA:
        mercados = [m for m, s in sectores.items() if s == sector]
        texto = ", ".join(mercados) if mercados else "ninguno"
        peso = f" (peso {pesos[mercados[0]]:.2f} R por mercado)" if mercados else ""
        print(f"  {sector:<16} {texto}{peso}")
    d1 = {m: velas["D1"][m] for m in sectores}
    imprimir_datos(d1, costes)
    for m, df in d1.items():
        if n := velas_con_spread_anomalo(df, puntos[m]):
            print(f"AVISO: {m} tiene {n} velas con un spread mayor que el 1% del precio (error de datos); "
                  "distorsionan sus stops y costes")

    principal = evaluar(d1, costes, puntos, "D1", 55, 20, pesos=pesos)
    robustez = {
        "D1 20/10 (vecino rápido)": evaluar(d1, costes, puntos, "D1", 20, 10, pesos=pesos),
        "D1 100/50 (vecino lento)": evaluar(d1, costes, puntos, "D1", 100, 50, pesos=pesos),
        "D1 55/20 con swap pesimista": evaluar(d1, costes, puntos, "D1", 55, 20, swap_pesimista=True, pesos=pesos),
    }
    informativas = {
        "D1 55/20 con el mismo peso por mercado": evaluar(d1, costes, puntos, "D1", 55, 20),
        "D1 55/20 con costes x3": evaluar(d1, costes, puntos, "D1", 55, 20, multiplicador_costes=3, pesos=pesos),
    }

    imprimir_variante("PRINCIPAL: D1 55/20, mismo riesgo por sector", principal, detalle=True)
    for nombre, ev in robustez.items():
        imprimir_variante(f"ROBUSTEZ: {nombre}", ev)
    for nombre, ev in informativas.items():
        imprimir_variante(f"INFORMATIVA: {nombre}", ev)
    imprimir_veredicto(principal, robustez)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Seguimiento de tendencia por reglas (Fase 6 de Plan_de_mejoras.md)")
    parser.add_argument("--diversificada", action="store_true", help="6.2: cartera diversificada por sectores")
    if parser.parse_args(argv).diversificada:
        informe_diversificado()
    else:
        informe_seis_mercados()


if __name__ == "__main__":
    main()
