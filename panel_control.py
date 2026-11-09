"""Ventana para configurar, iniciar, detener y ver las operaciones del bot en cada activo de la demo Exness.

Uso: abrir_panel.bat (o python panel_control.py). Cada activo se lanza como un proceso de demo_exness_bot.py, con su
propio modelo, registro y estado de riesgo; cerrar la ventana no los detiene.
"""
import ctypes
import json
import math
import sys
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import config
from activos import ACTIVOS, EURUSD, LOGS, Activo
from cuenta import CUENTA
from operaciones import Instantanea, Operacion, leer_activos
from panel_mt5 import SIN_LIMITE
from vigilante import dejar_de_desear, desear, horas_de, lanzar, leer_deseados, leer_panel, ultima_linea  # noqa: F401

TITULO = "RF Bot: configuración"
LANZADOR = config.BASE_DIR / "demo_exness_bot.py"
RUTA_AJUSTES = LOGS / "panel_control.json"
INTERVALO_MS = 2000
MAX_LATIDO_S = 20  # como el indicador de MT5: el bot publica al menos cada 5 s
MAX_LATIDO_VIGILANTE_S = 45  # el vigilante publica cada 10 s
MAX_HORAS = 24 * 30
VERDE, AMBAR, ROJO, GRIS, AZUL = "#1a7f37", "#9a6700", "#cf222e", "#656d76", "#0969da"
INTERVALO_VISTAS_S = 2
TICK_VISTAS_MS = 200
COLUMNAS_VISTA = (("apertura", "Apertura", 115, "w"), ("tipo", "Tipo", 60, "w"), ("lote", "Lote", 50, "e"),
                  ("entrada", "Entrada", 90, "e"), ("stop", "Stop", 90, "e"), ("objetivo", "Objetivo", 90, "e"),
                  ("cierre", "Cierre", 115, "w"), ("salida", "Salida / actual", 95, "e"), ("estado", "Estado", 90, "w"),
                  ("resultado", "Resultado", 80, "e"))
COMUNES_INICIALES = {"horas": "5", "perdida_diaria": f"{config.MAX_PERDIDA_DIARIA_PCT * 100:g}",
                     "drawdown": f"{config.MAX_DRAWDOWN_PCT * 100:g}", "riesgo_abierto": "2", "max_posiciones": "3"}


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


def _tope(comunes: dict, clave: str, opcion: str, nombre: str, minimo: float, maximo: float, entero: bool = False) -> list[str]:
    """La opción y su valor, o nada si el campo está vacío: ese tope de la cuenta no se aplica."""
    texto = comunes.get(clave, "")
    if not texto.strip():
        return []
    valor = _en_rango(texto, nombre, minimo, maximo)
    if entero and not numero(valor).is_integer():
        raise ValueError(f"{nombre} debe ser un número entero")
    return [opcion, valor]


def comando(activo: Activo, ajustes: dict, comunes: dict) -> list[str]:
    """Línea de órdenes de demo_exness_bot.py para estos ajustes; ValueError con un mensaje si alguno no vale."""
    umbral_long, umbral_short = _umbral(ajustes["umbral_long"], "El umbral de compra"), _umbral(ajustes["umbral_short"], "El umbral de venta")
    if umbral_long == umbral_short == "off":
        raise ValueError("indica al menos un umbral, de compra o de venta")
    return [
        str(Path(sys.executable).with_name("python.exe")), str(LANZADOR),
        "--simbolo", activo.simbolo,
        *_tope(comunes, "horas", "--horas", "La duración", 0.01, MAX_HORAS),  # vacía: hasta que se detenga
        "--volumen", _en_rango(ajustes["lote"], "El lote", activo.lote_min, config.MAX_LOTES),
        "--max-spread", _en_rango(ajustes["max_spread"], "El spread máximo", 1, 1_000_000),
        "--umbral-long", umbral_long,
        "--umbral-short", umbral_short,
        "--perdida-diaria", _en_rango(comunes["perdida_diaria"], "La pérdida diaria", 0.01, 100),
        "--drawdown", _en_rango(comunes["drawdown"], "El drawdown", 0.01, 100),
        *_tope(comunes, "riesgo_abierto", "--riesgo-abierto", "El riesgo abierto", 0.01, 100),
        *_tope(comunes, "max_posiciones", "--max-posiciones", "El máximo de posiciones", 1, 100, entero=True),
    ]


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


def texto_vigilante(ruta_latido: Path, ahora: float) -> tuple:
    """(texto, color) sobre si vigilante.py está en marcha, según el latido que publica."""
    try:
        activo = ahora - float(ruta_latido.read_text(encoding="utf-8")) <= MAX_LATIDO_VIGILANTE_S
    except (OSError, ValueError):
        activo = False
    if activo:
        return "Vigilante en marcha: relanza los bots que se caigan", VERDE
    return "Vigilante parado: si un bot se cae, no se relanza (iniciar_vigilante.bat)", GRIS


@dataclass(frozen=True)
class Estado:
    texto: str
    color: str
    en_marcha: bool  # bloquea la edición y evita volver a iniciarlo


def estado_del_bot(panel: dict, ahora: float, proceso=None, salida: str = "") -> Estado:
    if panel.get("estado") == "en_marcha" and ahora - float(panel.get("actualizado") or 0) <= MAX_LATIDO_S:
        fin = panel.get("fin", "?")
        return Estado("En marcha, sin límite" if fin == SIN_LIMITE else f"En marcha hasta las {fin}", VERDE, True)
    if proceso is not None and proceso.poll() is None:
        return Estado("Arrancando…", AMBAR, True)
    if proceso is not None and proceso.returncode != 0:
        return Estado(f"No pudo seguir: {salida}" if salida else "No pudo seguir", ROJO, False)
    if panel.get("estado") == "en_marcha":
        return Estado("Sin respuesta", ROJO, False)
    return Estado("Detenido", GRIS, False)


def _fecha(tiempo: int) -> str:
    return datetime.fromtimestamp(tiempo).strftime("%d/%m %H:%M:%S")


def fila_de_operacion(op: Operacion, digitos: int) -> tuple:
    """(valores de las columnas, etiqueta de color)."""
    precio = lambda valor: f"{valor:.{digitos}f}" if valor else ""  # noqa: E731
    etiqueta = "abierta" if op.cierre is None else "ganada" if op.resultado > 0 else "perdida"
    return (_fecha(op.apertura), "Compra" if op.compra else "Venta", f"{op.lote:g}", precio(op.entrada), precio(op.sl),
            precio(op.tp), _fecha(op.cierre) if op.cierre else "", precio(op.salida), op.motivo, f"{op.resultado:+.2f}"), etiqueta


def texto_resumen(datos: Instantanea) -> str:
    if not datos.operaciones:
        return "Todavía no hay operaciones de este activo"
    cerradas = [o for o in datos.operaciones if o.cierre is not None]
    abiertas = [o for o in datos.operaciones if o.cierre is None]
    ganadas = sum(o.resultado > 0 for o in cerradas)
    partes = [f"{len(cerradas)} cerradas", f"{ganadas} ganadas", f"{len(cerradas) - ganadas} perdidas",
              f"resultado {sum(o.resultado for o in cerradas):+.2f} {datos.moneda}"]
    if abiertas:
        partes.append(f"abierta ahora {sum(o.resultado for o in abiertas):+.2f} {datos.moneda}")
    return " · ".join(partes)


class VistaOperaciones:
    """Ventana con todas las operaciones de un activo; Ventana le pasa cada lectura nueva."""

    def __init__(self, raiz: tk.Tk, activo: Activo, al_cerrar):
        self.ventana = tk.Toplevel(raiz)
        self.ventana.title(f"{activo.nombre} ({activo.simbolo}): operaciones")
        self.ventana.protocol("WM_DELETE_WINDOW", al_cerrar)
        marco = ttk.Frame(self.ventana, padding=12)
        marco.pack(fill="both", expand=True)
        self.resumen = ttk.Label(marco, text="Leyendo MT5…", font=("Segoe UI", 9, "bold"))
        self.resumen.pack(anchor="w", pady=(0, 8))
        cuerpo = ttk.Frame(marco)
        cuerpo.pack(fill="both", expand=True)
        self.tabla = ttk.Treeview(cuerpo, columns=[c[0] for c in COLUMNAS_VISTA], show="headings", height=14, selectmode="browse")
        for clave, titulo, ancho, alineacion in COLUMNAS_VISTA:
            self.tabla.heading(clave, text=titulo, anchor=alineacion)
            self.tabla.column(clave, width=ancho, anchor=alineacion, stretch=clave in ("apertura", "cierre"))
        barra = ttk.Scrollbar(cuerpo, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=barra.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")
        for etiqueta, color in (("abierta", AZUL), ("ganada", VERDE), ("perdida", ROJO)):
            self.tabla.tag_configure(etiqueta, foreground=color)

    def mostrar(self, datos) -> None:
        """`datos` es una Instantanea, o el motivo por el que no se pudo leer. Actualiza sin perder el scroll ni la selección."""
        if isinstance(datos, str):
            self.resumen.configure(text=f"No se pudo leer MT5: {datos}", foreground=ROJO)
            return
        self.resumen.configure(text=texto_resumen(datos), foreground="")
        for indice, op in enumerate(datos.operaciones):
            iid, (valores, etiqueta) = str(op.ticket), fila_de_operacion(op, datos.digitos)
            if self.tabla.exists(iid):
                self.tabla.item(iid, values=valores, tags=(etiqueta,))
                self.tabla.move(iid, "", indice)
            else:
                self.tabla.insert("", indice, iid=iid, values=valores, tags=(etiqueta,))


@dataclass
class Fila:
    activo: Activo
    variables: dict
    casilla: ttk.Checkbutton
    entradas: list
    estado: ttk.Label
    probabilidades: ttk.Label
    nota: ttk.Label
    ver: ttk.Button
    detener: ttk.Button


class Ventana:
    def __init__(self, raiz: tk.Tk, activos: dict = ACTIVOS, lanzar=lanzar, ruta_ajustes: Path = RUTA_AJUSTES,
                 leer_operaciones=leer_activos):
        self.raiz, self.lanzar, self.ruta_ajustes, self.leer_operaciones = raiz, lanzar, ruta_ajustes, leer_operaciones
        self.ruta_deseados = ruta_ajustes.with_name("vigilante.json")  # lo que vigilante.py debe mantener en marcha
        self.procesos, self.deteniendo, self.estados = {}, set(), {}
        self.vistas, self.lectura, self.proxima_lectura = {}, None, 0.0
        self.lector = ThreadPoolExecutor(max_workers=1)  # un solo hilo: la librería de MT5 tiene una conexión por proceso
        guardados = leer_json(ruta_ajustes)
        raiz.title(TITULO)
        raiz.protocol("WM_DELETE_WINDOW", self.cerrar)

        marco = ttk.Frame(raiz, padding=14)
        marco.grid(sticky="nsew")
        raiz.columnconfigure(0, weight=1)
        ttk.Label(marco, text=f"Cuenta {'REAL' if CUENTA.real else 'demo'} {CUENTA.login} · {CUENTA.server}",
                  foreground=ROJO if CUENTA.real else GRIS).grid(sticky="w", pady=(0, 8))

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
        campos = ((("Duración (horas)", "horas"), ("Pérdida diaria máx. (%)", "perdida_diaria"), ("Drawdown máx. (%)", "drawdown")),
                  (("Riesgo abierto máx. (%)", "riesgo_abierto"), ("Posiciones máx.", "max_posiciones")))
        for fila, linea in enumerate(campos):
            for columna, (texto, clave) in enumerate(linea):
                ttk.Label(abajo, text=texto).grid(row=fila, column=2 * columna, padx=(0 if columna == 0 else 14, 4), pady=2, sticky="e")
                ttk.Entry(abajo, textvariable=self.comunes[clave], width=6, justify="right").grid(row=fila, column=2 * columna + 1)
        ttk.Button(abajo, text="Iniciar marcados", command=self.iniciar).grid(row=0, column=7, padx=(14, 6))
        ttk.Button(abajo, text="Detener todos", command=self.detener_todos).grid(row=0, column=8)

        ttk.Label(marco, foreground=GRIS, text="Deja un umbral vacío para no operar ese lado. Cerrar esta ventana no detiene los "
                                               "bots.\nAl detener un bot, sus posiciones abiertas conservan el stop y el objetivo.\n"
                                               "El riesgo abierto y las posiciones máximas cuentan todos los activos de la cuenta; "
                                               "vacíos, no se aplican.\nCon la duración vacía, el bot corre hasta que lo detengas.",
                  justify="left").grid(sticky="w")
        self.vigilante = ttk.Label(marco)
        self.vigilante.grid(sticky="w", pady=(6, 0))
        self.refrescar()
        self._atender_vistas()

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
        fila = Fila(activo, variables, casilla, entradas, estado, probabilidades, ttk.Label(tabla, foreground=GRIS),
                    ttk.Button(tabla, text="Ver", width=6), ttk.Button(tabla, text="Detener", width=9))
        fila.nota.grid(row=n + 1, column=1, columnspan=9, sticky="w", padx=6, pady=(0, 6))
        fila.ver.configure(command=lambda: self.ver(fila))
        fila.ver.grid(row=n, column=8, padx=(6, 0))
        fila.detener.configure(command=lambda: self.detener(fila))
        fila.detener.grid(row=n, column=9, padx=(6, 0))
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
            self._apuntar(lambda f=fila, o=orden: desear(self.ruta_deseados, f.activo.simbolo, o,
                                                        time.time() + horas_de(o) * 3600 if horas_de(o) else None))
        self.refrescar(repetir=False)

    def detener(self, fila: Fila) -> None:
        fila.activo.carpeta.mkdir(parents=True, exist_ok=True)
        self._apuntar(lambda: dejar_de_desear(self.ruta_deseados, fila.activo.simbolo))  # antes: que no lo relancen
        fila.activo.detener.touch()
        self.deteniendo.add(fila.activo.simbolo)
        self.refrescar(repetir=False)

    def _apuntar(self, cambio) -> None:
        """Un fallo al apuntar para el vigilante no impide iniciar ni detener: solo se avisa."""
        try:
            cambio()
        except Exception as e:
            messagebox.showwarning(TITULO, f"No se pudo actualizar la lista del vigilante: {e}")

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
        texto, color = texto_vigilante(self.ruta_deseados.with_name("vigilante.txt"), ahora)
        self.vigilante.configure(text=texto, foreground=color)
        if repetir:
            self.raiz.after(INTERVALO_MS, self.refrescar)

    def ver(self, fila: Fila) -> None:
        """Abre la vista de operaciones del activo, o la trae al frente si ya estaba abierta."""
        simbolo = fila.activo.simbolo
        if simbolo in self.vistas:
            self.vistas[simbolo].ventana.deiconify()
            self.vistas[simbolo].ventana.lift()
            return
        self.vistas[simbolo] = VistaOperaciones(self.raiz, fila.activo, lambda: self.cerrar_vista(simbolo))
        self.proxima_lectura = 0.0  # que no espere al siguiente intervalo

    def cerrar_vista(self, simbolo: str) -> None:
        self.vistas.pop(simbolo).ventana.destroy()

    def _atender_vistas(self) -> None:
        """La lectura de MT5 va en otro hilo, para que una terminal lenta nunca congele la ventana."""
        if self.lectura is not None and self.lectura.done():
            for simbolo, datos in self.lectura.result().items():
                if simbolo in self.vistas:
                    self.vistas[simbolo].mostrar(datos)
            self.lectura, self.proxima_lectura = None, time.monotonic() + INTERVALO_VISTAS_S
        if self.lectura is None and self.vistas and time.monotonic() >= self.proxima_lectura:
            activos = [f.activo for f in self.filas if f.activo.simbolo in self.vistas]
            self.lectura = self.lector.submit(self.leer_operaciones, activos)
        self.raiz.after(TICK_VISTAS_MS, self._atender_vistas)

    def cerrar(self) -> None:
        self.guardar()
        self.lector.shutdown(wait=False)
        self.raiz.destroy()


def crear_raiz(intentos: int = 5) -> tk.Tk:
    """En Windows, Tcl a veces no consigue leer sus propios archivos al arrancar y al reintentar funciona."""
    for intento in range(1, intentos + 1):
        try:
            return tk.Tk()
        except tk.TclError:
            if intento == intentos:
                raise
            time.sleep(0.2)


def main() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sin esto Windows la dibuja borrosa en pantallas con escala
    except (AttributeError, OSError):
        pass
    raiz = crear_raiz()
    Ventana(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
