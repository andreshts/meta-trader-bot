"""Avisos del bot por Telegram: ejecuciones, cierres, límites alcanzados, errores y paradas.

Solo observa: se alimenta del log del bot y envía desde otro hilo, así que una red lenta o caída nunca frena
ni interrumpe la operativa. Sin TELEGRAM_TOKEN y TELEGRAM_CHAT_ID en `.env` no hace nada.

Puesta en marcha:
  1. En Telegram, hablar con @BotFather, enviar /newbot y copiar el token a TELEGRAM_TOKEN en `.env`.
  2. Escribirle cualquier mensaje al bot recién creado.
  3. python alertas.py --chat      muestra el identificador de ese chat: copiarlo a TELEGRAM_CHAT_ID.
  4. python alertas.py --probar    envía un mensaje de prueba.
"""
import argparse
import json
import logging
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import config  # noqa: F401  (carga .env)

API = "https://api.telegram.org/bot{token}/{metodo}"
TIMEOUT_S = 10
MAX_CARACTERES = 4000  # Telegram admite 4096 por mensaje
PAUSA_AVISOS_S = 3600        # un mismo aviso o error no se repite antes de este tiempo
PAUSA_BLOQUEOS_S = 6 * 3600  # un mismo motivo para no abrir entradas se recuerda cada este tiempo
MAX_CLAVES = 500

# Fragmentos de los mensajes de trading_bot.py y demo_exness_bot.py (tests/test_alertas.py comprueba que existen).
SIEMPRE = (" ejecutada", "confirmada al reconciliar", "Deal de %s", "Bot iniciado", "Entrada pendiente (orden %s) resuelta",
           "Fin de la sesión programada", "Detenido desde la ventana", "Bot detenido por el usuario", "Conexión recuperada")
BLOQUEOS = ("Sin nuevas entradas: %s", "Sin entrada: %s")
IGNORADOS = ("Spread de %s puntos", "Noticia de alto impacto", "EXIGIR_VENTAJA_SIGNIFICATIVA = False", "Umbral manual")
ICONOS = {logging.INFO: "ℹ️", logging.WARNING: "⚠️", logging.ERROR: "❌", logging.CRITICAL: "❌"}
NUMEROS = re.compile(r"\d[\d.,:]*")

registro = logging.getLogger("alertas")  # distinto del logger del bot: lo que falle aquí no vuelve a entrar aquí


class ErrorTelegram(Exception):
    """Nunca lleva el token en su texto."""


def credenciales(entorno=os.environ) -> tuple:
    """(token, chat) o (None, None) si falta alguno."""
    token, chat = (entorno.get("TELEGRAM_TOKEN") or "").strip(), (entorno.get("TELEGRAM_CHAT_ID") or "").strip()
    return (token, chat) if token and chat else (None, None)


def _llamar(token: str, metodo: str, parametros: dict, intentos: int = 3) -> dict:
    datos = urllib.parse.urlencode(parametros).encode()
    motivo = "sin respuesta"
    for intento in range(intentos):
        espera = 2 ** intento
        try:
            with urllib.request.urlopen(API.format(token=token, metodo=metodo), data=datos, timeout=TIMEOUT_S) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                cuerpo = json.loads(e.read())
            except ValueError:
                cuerpo = {}
            motivo = f"Telegram respondió {e.code}: {cuerpo.get('description', e.reason)}"
            if e.code == 429:
                espera = min(30, int(cuerpo.get("parameters", {}).get("retry_after", espera)))
            elif e.code < 500:
                break  # token o chat incorrectos: reintentar no lo arregla
        except (urllib.error.URLError, OSError, ValueError) as e:
            motivo = f"no se pudo contactar con Telegram ({type(e).__name__})"
        if intento + 1 < intentos:
            time.sleep(espera)
    raise ErrorTelegram(motivo) from None  # sin la excepción original, que puede llevar la URL con el token


def enviar_telegram(texto: str, token: str, chat: str) -> None:
    _llamar(token, "sendMessage", {"chat_id": chat, "text": texto[:MAX_CARACTERES], "disable_web_page_preview": "true"})


def _pausa(record: logging.LogRecord):
    """None si el mensaje no se avisa, 0 si se avisa siempre, o los segundos mínimos entre dos avisos iguales."""
    plantilla = record.msg if isinstance(record.msg, str) else str(record.msg)
    if any(f in plantilla for f in IGNORADOS):
        return None
    if any(f in plantilla for f in SIEMPRE):
        return 0
    if any(f in plantilla for f in BLOQUEOS):
        return PAUSA_BLOQUEOS_S
    return PAUSA_AVISOS_S if record.levelno >= logging.WARNING else None


class AlertasTelegram(logging.Handler):
    """Se engancha al logger del bot. `enviar(texto)` hace el envío real; se llama desde un hilo propio."""

    def __init__(self, etiqueta: str, enviar, reloj=time.monotonic):
        super().__init__(logging.INFO)
        self.etiqueta, self._enviar, self._reloj = etiqueta, enviar, reloj
        self._ultimo = {}  # mensaje sin números -> (momento del último aviso, veces repetido desde entonces)
        self._cola = queue.Queue(maxsize=200)
        threading.Thread(target=self._trabajar, daemon=True, name="alertas-telegram").start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            pausa = _pausa(record)
            if pausa is None:
                return
            texto = " ".join(record.getMessage().split())
            if record.exc_info and record.exc_info[1] is not None:
                texto += f" ({type(record.exc_info[1]).__name__}: {record.exc_info[1]})"
            if pausa:
                clave, ahora = NUMEROS.sub("#", texto), self._reloj()
                ultimo, repetidos = self._ultimo.get(clave, (-math.inf, 0))
                if ahora - ultimo < pausa:
                    self._ultimo[clave] = (ultimo, repetidos + 1)
                    return
                if len(self._ultimo) >= MAX_CLAVES:
                    self._ultimo.clear()
                self._ultimo[clave] = (ahora, 0)
                if repetidos:
                    texto += f" (repetido {repetidos} veces desde el aviso anterior)"
            self.avisar(f"{ICONOS.get(record.levelno, '')} {texto}".strip())
        except Exception:
            self.handleError(record)

    def avisar(self, texto: str) -> None:
        """Encola un aviso sin esperar. Si la cola está llena (Telegram caído mucho tiempo), se descarta."""
        try:
            self._cola.put_nowait(f"[{self.etiqueta}] {texto}")
        except queue.Full:
            pass

    def _trabajar(self) -> None:
        while True:
            texto = self._cola.get()
            try:
                self._enviar(texto)
            except Exception as e:
                detalle = str(e) if isinstance(e, ErrorTelegram) else type(e).__name__
                registro.warning("No se pudo enviar un aviso a Telegram: %s", detalle)
            finally:
                self._cola.task_done()

    def cerrar(self, espera_s: float = 15.0) -> None:
        """Da tiempo a que salgan los avisos encolados antes de que termine el proceso."""
        limite = time.monotonic() + espera_s
        while self._cola.unfinished_tasks and time.monotonic() < limite:
            time.sleep(0.05)


def crear(etiqueta: str):
    """El manejador listo para añadir al logger del bot, o None si Telegram no está configurado."""
    token, chat = credenciales()
    if token is None:
        return None
    return AlertasTelegram(etiqueta, lambda texto: enviar_telegram(texto, token, chat))


def chats(token: str) -> list:
    """(identificador, nombre) de los chats que han escrito al bot hace poco."""
    vistos = {}
    for novedad in _llamar(token, "getUpdates", {}).get("result", []):
        chat = (novedad.get("message") or novedad.get("channel_post") or {}).get("chat")
        if chat:
            vistos[chat["id"]] = chat.get("title") or chat.get("username") or chat.get("first_name") or ""
    return list(vistos.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    accion = parser.add_mutually_exclusive_group(required=True)
    accion.add_argument("--chat", action="store_true", help="muestra los chats que han escrito al bot")
    accion.add_argument("--probar", action="store_true", help="envía un mensaje de prueba")
    args = parser.parse_args()
    token = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Falta TELEGRAM_TOKEN en .env")
    try:
        if args.chat:
            encontrados = chats(token)
            if not encontrados:
                raise SystemExit("Ningún chat todavía: escríbele un mensaje al bot en Telegram y repite")
            for identificador, nombre in encontrados:
                print(f"TELEGRAM_CHAT_ID={identificador}   ({nombre})")
        else:
            _, chat = credenciales()
            if chat is None:
                raise SystemExit("Falta TELEGRAM_CHAT_ID en .env (python alertas.py --chat lo muestra)")
            enviar_telegram("✅ Avisos del bot de trading configurados", token, chat)
            print("Mensaje de prueba enviado")
    except ErrorTelegram as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
