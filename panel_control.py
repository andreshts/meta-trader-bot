"""Ventana para configurar, iniciar y detener el bot en cada activo de la demo Exness.

Uso: abrir_panel.bat (o python panel_control.py). Cada activo se lanza como un proceso de demo_exness_bot.py, con su
propio modelo, registro y estado de riesgo; cerrar la ventana no los detiene.
"""
import ctypes
import json
import math
import os
import subprocess
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import config
from activos import ACTIVOS, EURUSD, LOGS, Activo
from demo_exness_probe import LOGIN, SERVER

TITULO = "RF Bot: configuración"
LANZADOR = config.BASE_DIR / "demo_exness_bot.py"
RUTA_AJUSTES = LOGS / "panel_control.json"
CREATE_NO_WINDOW = 0x08000000
INTERVALO_MS = 2000
MAX_LATIDO_S = 20  # como el indicador de MT5: el bot publica al menos cada 5 s
MAX_HORAS = 24 * 30
VERDE, AMBAR, ROJO, GRIS = "#1a7f37", "#9a6700", "#cf222e", "#656d76"
COMUNES_INICIALES = {"horas": "5", "perdida_diaria": f"{config.MAX_PERDIDA_DIARIA_PCT * 100:g}",
                     "drawdown": f"{config.MAX_DRAWDOWN_PCT * 100:g}"}


def numero(texto: str) -> float:
    """Admite la coma decimal. ValueError si no es un número finito."""
    valor = float(texto.strip().replace(",", "."))
    if not math.isfinite(valor):
        raise ValueError(texto)
    return valor


def _en_rango(texto: str, nombre: str, minimo: float, maximo: float) -> str:
    """El número normalizado como texto. ValueError con un mensaje para la persona si no vale."""
    try:
        valor = numero(texto)
    except ValueError:
        raise ValueError(f"{nombre}: «{texto}» no es un número") from None
    if not minimo <= valor <= maximo:
        raise ValueError(f"{nombre} debe estar entre {minimo:g} y {maximo:g}")
    return f"{valor:g}"


def _umbral(texto: str, nombre: str) -> str:
    """Vacío significa que ese lado no opera."""
    if not texto.strip():
        return "off"
    return _en_rango(texto, nombre, 0.01, 0.99)


def comando(activo: Activo, ajustes: dict, comunes: dict) -> list[str]:
    """Línea de órdenes de demo_exness_bot.py para estos ajustes; ValueError con un mensaje si alguno no vale."""
    umbral_long, umbral_short = _umbral(ajustes["umbral_long"], "El umbral de compra"), _umbral(ajustes["umbral_short"], "El umbral de venta")
    if umbral_long == umbral_short == "off":
        raise ValueError("indica al menos un umbral, de compra o de venta")
    return [
        str(Path(sys.executable).with_name("python.exe")), str(LANZADOR),
        "--simbolo", activo.simbolo,
        "--horas", _en_rango(comunes["horas"], "La duración", 0.01, MAX_HORAS),
        "--volumen", _en_rango(ajustes["lote"], "El lote", activo.lote_min, config.MAX_LOTES),
        "--max-spread", _en_rango(ajustes["max_spread"], "El spread máximo", 1, 1_000_000),
        "--umbral-long", umbral_long,
        "--umbral-short", umbral_short,
        "--perdida-diaria", _en_rango(comunes["perdida_diaria"], "La pérdida diaria", 0.01, 100),
        "--drawdown", _en_rango(comunes["drawdown"], "El drawdown", 0.01, 100),
    ]


def lanzar(activo: Activo, orden: list[str]) -> subprocess.Popen:
    """Arranca el bot sin consola. Lo que escriba al salir con error queda en `activo.salida`."""
    activo.carpeta.mkdir(parents=True, exist_ok=True)
    with open(activo.salida, "w", encoding="utf-8") as salida:
        return subprocess.Popen(orden, cwd=config.BASE_DIR, stdin=subprocess.DEVNULL, stdout=salida, stderr=subprocess.STDOUT,
                                creationflags=CREATE_NO_WINDOW, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def ajustes_iniciales(activo: Activo) -> dict:
    umbrales = activo.umbrales
    if umbrales is None:
        try:
            datos = json.loads(activo.modelo.with_suffix(".json").read_text(encoding="utf-8"))
            umbrales = datos["umbral_long"], datos["umbral_short"]
        except (OSError, ValueError, KeyError):
            umbrales = "", ""
    return {"operar": activo.simbolo == EURUSD.simbolo, "lote": f"{activo.lote_min:g}", "max_spread": str(activo.max_spread),
            "umbral_long": f"{umbrales[0]}", "umbral_short": f"{umbrales[1]}"}


def leer_json(ruta: Path) -> dict:
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return datos if isinstance(datos, dict) else {}


def leer_panel(ruta: Path) -> dict:
    """Lo último que publicó el bot de ese activo; vacío si todavía no hay nada."""
    try:
        texto = ruta.read_text(encoding="utf-8")
    except (OSError, ValueError):  # el archivo se reemplaza justo en este momento
        return {}
    return dict(linea.split("=", 1) for linea in texto.splitlines() if "=" in linea)


def ultima_linea(ruta: Path) -> str:
    try:
        lineas = [l.strip() for l in ruta.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    except OSError:
        return ""
    return lineas[-1] if lineas else ""


@dataclass(frozen=True)
class Estado:
    texto: str
    color: str
    en_marcha: bool  # bloquea la edición y evita volver a iniciarlo


def estado_del_bot(panel: dict, ahora: float, proceso=None, salida: str = "") -> Estado:
    if panel.get("estado") == "en_marcha" and ahora - float(panel.get("actualizado") or 0) <= MAX_LATIDO_S:
        return Estado(f"En marcha hasta las {panel.get('fin', '?')}", VERDE, True)
    if proceso is not None and proceso.poll() is None:
        return Estado("Arrancando…", AMBAR, True)
    if proceso is not None and proceso.returncode != 0:
        return Estado(f"No pudo seguir: {salida}" if salida else "No pudo seguir", ROJO, False)
    if panel.get("estado") == "en_marcha":
        return Estado("Sin respuesta", ROJO, False)
    return Estado("Detenido", GRIS, False)


@dataclass
class Fila:
    activo: Activo
    variables: dict
    casilla: ttk.Checkbutton
    entradas: list
    estado: ttk.Label
    probabilidades: ttk.Label
    nota: ttk.Label
    detener: ttk.Button


class Ventana:
    def __init__(self, raiz: tk.Tk, activos: dict = ACTIVOS, lanzar=lanzar, ruta_ajustes: Path = RUTA_AJUSTES):
        self.raiz, self.lanzar, self.ruta_ajustes = raiz, lanzar, ruta_ajustes
        self.procesos, self.deteniendo, self.estados = {}, set(), {}
        guardados = leer_json(ruta_ajustes)
        raiz.title(TITULO)
        raiz.protocol("WM_DELETE_WINDOW", self.cerrar)

        marco = ttk.Frame(raiz, padding=14)
        marco.grid(sticky="nsew")
        raiz.columnconfigure(0, weight=1)
        ttk.Label(marco, text=f"Cuenta demo {LOGIN} · {SERVER}", foreground=GRIS).grid(sticky="w", pady=(0, 8))

        tabla = ttk.Frame(marco)
        tabla.grid(sticky="ew")
        tabla.columnconfigure(6, weight=1)
        negrita = ("Segoe UI", 9, "bold")
        for columna, titulo in enumerate(("Operar", "Activo", "Lote", "Spread máx.\n(puntos)", "Umbral\ncompra", "Umbral\nventa",
                                          "Estado", "P compra / venta")):
            ttk.Label(tabla, text=titulo, font=negrita).grid(row=0, column=columna, sticky="sw", padx=6, pady=(0, 6))
        self.filas = [self._crear_fila(tabla, 2 * n + 1, activo, guardados.get("activos", {}).get(activo.simbolo, {}))
                      for n, activo in enumerate(activos.values())]

        comunes = {**COMUNES_INICIALES, **guardados.get("comunes", {})}
        self.comunes = {clave: tk.StringVar(value=comunes[clave]) for clave in COMUNES_INICIALES}
        abajo = ttk.Frame(marco)
        abajo.grid(sticky="ew", pady=(12, 8))
        abajo.columnconfigure(6, weight=1)
        for columna, (texto, clave) in enumerate((("Duración (horas)", "horas"), ("Pérdida diaria máx. (%)", "perdida_diaria"),
                                                  ("Drawdown máx. (%)", "drawdown"))):
            ttk.Label(abajo, text=texto).grid(row=0, column=2 * columna, padx=(0 if columna == 0 else 14, 4))
            ttk.Entry(abajo, textvariable=self.comunes[clave], width=6, justify="right").grid(row=0, column=2 * columna + 1)
        ttk.Button(abajo, text="Iniciar marcados", command=self.iniciar).grid(row=0, column=7, padx=(14, 6))
        ttk.Button(abajo, text="Detener todos", command=self.detener_todos).grid(row=0, column=8)

        ttk.Label(marco, foreground=GRIS, text="Deja un umbral vacío para no operar ese lado. Cerrar esta ventana no detiene los "
                                               "bots.\nAl detener un bot, sus posiciones abiertas conservan el stop y el objetivo.",
                  justify="left").grid(sticky="w")
        self.refrescar()

    def _crear_fila(self, tabla: ttk.Frame, n: int, activo: Activo, guardado: dict) -> Fila:
        valores = {**ajustes_iniciales(activo), **guardado}
        variables = {"operar": tk.BooleanVar(value=bool(valores["operar"])),
                     **{clave: tk.StringVar(value=str(valores[clave]))
                        for clave in ("lote", "max_spread", "umbral_long", "umbral_short")}}
        casilla = ttk.Checkbutton(tabla, variable=variables["operar"])
        casilla.grid(row=n, column=0, padx=6)
        ttk.Label(tabla, text=f"{activo.nombre}  ({activo.simbolo})").grid(row=n, column=1, sticky="w", padx=6)
        entradas = []
        for columna, clave in enumerate(("lote", "max_spread", "umbral_long", "umbral_short"), start=2):
            entrada = ttk.Entry(tabla, textvariable=variables[clave], width=8, justify="right")
            entrada.grid(row=n, column=columna, sticky="w", padx=6, pady=(4, 0))
            entradas.append(entrada)
        estado, probabilidades = ttk.Label(tabla, width=30), ttk.Label(tabla, width=20)
        estado.grid(row=n, column=6, sticky="w", padx=6)
        probabilidades.grid(row=n, column=7, sticky="w", padx=6)
        fila = Fila(activo, variables, casilla, entradas, estado, probabilidades,
                    ttk.Label(tabla, foreground=GRIS), ttk.Button(tabla, text="Detener", width=9))
        fila.nota.grid(row=n + 1, column=1, columnspan=8, sticky="w", padx=6, pady=(0, 6))
        fila.detener.configure(command=lambda: self.detener(fila))
        fila.detener.grid(row=n, column=8, padx=(6, 0))
        return fila

    def _ajustes(self, fila: Fila) -> dict:
        return {clave: variable.get() for clave, variable in fila.variables.items()}

    def _comunes(self) -> dict:
        return {clave: variable.get() for clave, variable in self.comunes.items()}

    def guardar(self) -> None:
        datos = {"comunes": self._comunes(), "activos": {f.activo.simbolo: self._ajustes(f) for f in self.filas}}
        try:
            self.ruta_ajustes.parent.mkdir(parents=True, exist_ok=True)
            self.ruta_ajustes.write_text(json.dumps(datos, indent=1), encoding="utf-8")
        except OSError as e:
            messagebox.showwarning(TITULO, f"No se pudieron guardar los ajustes: {e}")

    def iniciar(self) -> None:
        self.refrescar(repetir=False)
        por_iniciar, problemas = [], []
        for fila in self.filas:
            if not fila.variables["operar"].get() or self.estados[fila.activo.simbolo].en_marcha:
                continue
            try:
                if not fila.activo.modelo.exists():
                    raise ValueError("falta entrenar su modelo (python train_exness.py)")
                por_iniciar.append((fila, comando(fila.activo, self._ajustes(fila), self._comunes())))
            except ValueError as e:
                problemas.append(f"{fila.activo.nombre}: {e}")
        if problemas:
            messagebox.showerror(TITULO, "\n".join(problemas))
            return
        if not por_iniciar:
            messagebox.showinfo(TITULO, "No hay nada que iniciar: marca un activo que no esté en marcha.")
            return
        self.guardar()
        for fila, orden in por_iniciar:
            self.procesos[fila.activo.simbolo] = self.lanzar(fila.activo, orden)
        self.refrescar(repetir=False)

    def detener(self, fila: Fila) -> None:
        fila.activo.carpeta.mkdir(parents=True, exist_ok=True)
        fila.activo.detener.touch()
        self.deteniendo.add(fila.activo.simbolo)
        self.refrescar(repetir=False)

    def detener_todos(self) -> None:
        for fila in self.filas:
            if self.estados[fila.activo.simbolo].en_marcha:
                self.detener(fila)

    def refrescar(self, repetir: bool = True) -> None:
        ahora = time.time()
        for fila in self.filas:
            simbolo, activo = fila.activo.simbolo, fila.activo
            panel = leer_panel(activo.panel)
            proceso = self.procesos.get(simbolo)
            salida = ultima_linea(activo.salida) if proceso is not None and proceso.poll() not in (None, 0) else ""
            estado = estado_del_bot(panel, ahora, proceso, salida)
            if not estado.en_marcha:
                self.deteniendo.discard(simbolo)
            elif simbolo in self.deteniendo:
                estado = Estado("Deteniendo…", AMBAR, True)
            if not activo.modelo.exists():
                estado = Estado("Falta entrenar su modelo", ROJO, False)
            self.estados[simbolo] = estado
            fila.estado.configure(text=estado.texto, foreground=estado.color)
            probabilidades = f"{panel['p_long']} / {panel['p_short']}" if estado.en_marcha and "p_long" in panel else ""
            fila.probabilidades.configure(text=probabilidades)
            fila.nota.configure(text=f"{panel.get('nota_hora', '')}  {panel.get('nota', '')}".strip() if estado.en_marcha else "")
            editable = "disabled" if estado.en_marcha else "normal"
            fila.casilla.configure(state=editable if activo.modelo.exists() else "disabled")
            for entrada in fila.entradas:
                entrada.configure(state=editable)
            fila.detener.configure(state="normal" if estado.en_marcha and simbolo not in self.deteniendo else "disabled")
        if repetir:
            self.raiz.after(INTERVALO_MS, self.refrescar)

    def cerrar(self) -> None:
        self.guardar()
        self.raiz.destroy()


def main() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sin esto Windows la dibuja borrosa en pantallas con escala
    except (AttributeError, OSError):
        pass
    raiz = tk.Tk()
    Ventana(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
