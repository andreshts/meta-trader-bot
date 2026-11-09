"""Lo que comparten todos los bots de una misma cuenta: el máximo de equity y el turno para abrir.

Cada activo corre en su proceso. El turno es un bloqueo de archivo que el sistema operativo libera si el proceso
muere; el máximo de equity vive en un único archivo, que es la referencia para el drawdown de todos.
"""
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path


class CuentaOcupada(Exception):
    """Otro bot de la cuenta tiene el turno y no lo soltó a tiempo."""


def _bloquear(archivo) -> None:
    archivo.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(archivo.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(archivo, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _desbloquear(archivo) -> None:
    if os.name == "nt":
        import msvcrt
        archivo.seek(0)
        msvcrt.locking(archivo.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def turno(ruta: Path, espera_s: float = 10.0, pausa_s: float = 0.05):
    """Exclusión entre los bots de la cuenta. CuentaOcupada si no se consigue en `espera_s`."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a+") as archivo:
        limite = time.monotonic() + espera_s
        while True:
            try:
                _bloquear(archivo)
                break
            except OSError:
                if time.monotonic() >= limite:
                    raise CuentaOcupada(f"otro bot de la cuenta tiene el turno ({ruta})") from None
                time.sleep(pausa_s)
        try:
            yield
        finally:
            _desbloquear(archivo)


def leer_equity_max(ruta: Path):
    """El máximo guardado, o None si el archivo no existe. ValueError si existe pero no es utilizable."""
    try:
        texto = ruta.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    datos = json.loads(texto)  # JSONDecodeError es un ValueError
    valor = datos.get("equity_max") if isinstance(datos, dict) else None
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor) or valor <= 0:
        raise ValueError(f"equity_max inválido: {valor!r}")
    return float(valor)


def _guardar_equity_max(ruta: Path, valor: float) -> None:
    temporal = ruta.with_name(f"{ruta.name}.{os.getpid()}.tmp")
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump({"equity_max": valor}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporal, ruta)


def subir_equity_max(ruta: Path, bloqueo: Path, equity: float, espera_s: float = 1.0) -> float:
    """El máximo de equity de la cuenta tras tener en cuenta `equity`.

    El archivo manda: si no existe (primer arranque o reinicio deliberado del drawdown) se crea con la equity actual.
    Solo se escribe al superar el máximo, y con el turno tomado, para que dos bots no se pisen.
    """
    maximo = leer_equity_max(ruta)
    if maximo is not None and equity <= maximo:
        return maximo
    with turno(bloqueo, espera_s):
        maximo = leer_equity_max(ruta)  # otro bot pudo subirlo mientras tanto
        if maximo is None or equity > maximo:
            _guardar_equity_max(ruta, equity)
            maximo = equity
    return maximo
