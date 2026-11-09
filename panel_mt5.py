"""Publica el estado del bot para el indicador RF_Bot_Panel, que lo dibuja sobre el gráfico de MT5.

Solo observa: se alimenta del log del bot, y un fallo aquí nunca interrumpe la operativa.
"""
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

# La carpeta Common\Files es la única que el indicador puede leer sin depender de la terminal concreta.
RUTA = Path(os.environ.get("APPDATA", ".")) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "rf_bot_panel.txt"
PLANTILLA_PROBABILIDADES = "P_long=%.3f P_short=%.3f"  # el mensaje de trading_bot.evaluar_entrada
INTERVALO_LATIDO_S = 5  # el indicador avisa si pasan 20 s sin novedades
SIN_LIMITE = "--:--"  # hora de fin de una sesión sin límite (el indicador la muestra tal cual)


class PanelMT5(logging.Handler):
    """Se engancha al logger del bot y reescribe el archivo del panel con cada novedad y cada latido."""

    def __init__(self, simbolo: str, cuenta: int, magic: int, fin: float, ruta: Path = RUTA):
        super().__init__(logging.INFO)
        self.ruta = ruta
        self.umbral_long = self.umbral_short = None
        self.datos = {"estado": "en_marcha", "simbolo": simbolo, "cuenta": cuenta, "magic": magic,
                      "fin": SIN_LIMITE if math.isinf(fin) else datetime.fromtimestamp(fin).strftime("%H:%M"),
                      "umbral_long": "", "umbral_short": ""}
        self._ultimo_latido = -math.inf

    def fijar_umbrales(self, umbral_long, umbral_short) -> None:
        """None significa que ese lado está desactivado en el modelo."""
        self.umbral_long, self.umbral_short = umbral_long, umbral_short
        self.datos.update(umbral_long="" if umbral_long is None else f"{umbral_long:.2f}",
                          umbral_short="" if umbral_short is None else f"{umbral_short:.2f}")
        self._publicar()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            hora = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            if record.msg == PLANTILLA_PROBABILIDADES:
                p_long, p_short = record.args
                self.datos.update(p_long=f"{p_long:.3f}", p_short=f"{p_short:.3f}", evaluada=hora)
                nota, nivel = self._decision(p_long, p_short), "INFO"
            else:
                nota, nivel = " ".join(record.getMessage().split()), record.levelname  # el archivo va por líneas
            self.datos.update(nota=nota, nota_hora=hora, nota_nivel=nivel)
            self._publicar()
        except Exception:
            self.handleError(record)

    def latido(self) -> None:
        """Lo llama el bucle del bot en cada vuelta. Sin latidos, el panel avisa de que el bot no responde."""
        if time.monotonic() - self._ultimo_latido >= INTERVALO_LATIDO_S:
            self._publicar()

    def detener(self) -> None:
        self.datos["estado"] = "detenido"
        self._publicar()

    def _decision(self, p_long: float, p_short: float) -> str:
        """El mismo criterio que trading_bot.evaluar_entrada, en palabras."""
        compra = self.umbral_long is not None and p_long >= self.umbral_long
        venta = self.umbral_short is not None and p_short >= self.umbral_short
        if compra and venta:
            return "Señales opuestas: no se opera"
        if compra or venta:
            return f"Señal de {'compra' if compra else 'venta'}"
        return "Sin señal: ninguna probabilidad llega a su umbral"

    def _publicar(self) -> None:
        self._ultimo_latido = time.monotonic()
        self.datos["actualizado"] = int(time.time())
        temporal = self.ruta.with_suffix(".tmp")
        try:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            with open(temporal, "w", encoding="utf-8", newline="\r\n") as f:
                f.writelines(f"{clave}={valor}\n" for clave, valor in self.datos.items())
            os.replace(temporal, self.ruta)  # el indicador nunca ve un archivo a medias
        except OSError:
            pass  # p. ej. el indicador lo tiene abierto: reintenta el próximo latido; si sigue, el panel lo avisa
