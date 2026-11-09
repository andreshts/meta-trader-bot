"""Cuenta de Exness en la que operan el lanzador, la ventana y el entrenamiento.

Sin variables de entorno es la demo de siempre. Para otra cuenta (demo o real), en `.env`:

    EXNESS_LOGIN=12345678
    EXNESS_SERVER=Exness-MT5Real8
    EXNESS_TERMINAL=C:\\Program Files\\MetaTrader 5 EXNESS\\terminal64.exe   (opcional)
    EXNESS_CUENTA_REAL=1          (obligatorio si la cuenta es real; sin él, una cuenta real detiene el bot)
    EXNESS_EXIGIR_VENTAJA=0       (opcional; por defecto 1 en real y 0 en demo)

La contraseña nunca se escribe: la terminal debe tener ya la sesión iniciada en esa cuenta.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import config  # carga .env

SI, NO = {"1", "si", "sí", "true"}, {"0", "no", "false"}


@dataclass(frozen=True)
class Cuenta:
    login: int
    server: str
    terminal: str
    real: bool = False
    exigir_ventaja: bool = False  # si es True, el bot no arranca con un modelo sin ventaja significativa

    @property
    def tipo(self) -> str:
        return "real" if self.real else "demo"

    def carpeta(self, base: Path) -> Path:
        """Registros y estado de riesgo: la demo original conserva los suyos y cada otra cuenta tiene carpeta propia."""
        return base if self.login == DEMO.login else base / str(self.login)


DEMO = Cuenta(480569059, "Exness-MT5Trial11", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")


def _booleano(entorno, nombre: str, por_defecto: bool) -> bool:
    texto = (entorno.get(nombre) or "").strip().lower()
    if not texto:
        return por_defecto
    if texto not in SI | NO:
        raise SystemExit(f"{nombre}={texto!r} no vale: usa 1 o 0")
    return texto in SI


def desde_entorno(entorno=os.environ) -> Cuenta:
    login, server = (entorno.get("EXNESS_LOGIN") or "").strip(), (entorno.get("EXNESS_SERVER") or "").strip()
    real = _booleano(entorno, "EXNESS_CUENTA_REAL", False)
    if not login and not server:
        if real:
            raise SystemExit("EXNESS_CUENTA_REAL=1 exige indicar EXNESS_LOGIN y EXNESS_SERVER")
        return DEMO
    if not login or not server:
        raise SystemExit("EXNESS_LOGIN y EXNESS_SERVER van juntos: falta uno de los dos")
    if not login.isdigit():
        raise SystemExit(f"EXNESS_LOGIN={login!r} no es un número de cuenta")
    return Cuenta(int(login), server, (entorno.get("EXNESS_TERMINAL") or "").strip() or DEMO.terminal, real,
                  _booleano(entorno, "EXNESS_EXIGIR_VENTAJA", real))


CUENTA = desde_entorno()
LOGS = CUENTA.carpeta(config.BASE_DIR / "logs" / "exness")
