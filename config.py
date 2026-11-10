"""Parámetros globales del bot (ver Entrenamiento.md y Ejecucion.md).

Las credenciales nunca se escriben aquí: se leen de variables de entorno o de `.env`.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# --- Mercado ---
SYMBOL = "EURUSD"
TIMEFRAME = "M15"  # nombre de la constante mt5.TIMEFRAME_<TIMEFRAME>
TIMEFRAME_SEGUNDOS = {"M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400}[TIMEFRAME]

# --- Datos ---
# Inicio fijo del histórico de entrenamiento (fecha del servidor): cada reentrenamiento usa el mismo inicio, así que
# los resultados no cambian porque la ventana avance unas velas. Antes de 2009 el spread de las velas es un valor fijo
# y el volumen de ticks viene de otra fuente. Requiere "Máx. barras en la ventana" = Ilimitado en MT5 (y reiniciarla).
HISTORICO_DESDE = "2010-01-01"
BUFFER_VELAS = 200        # velas cerradas que se descargan en vivo
VELAS_CONTIGUAS_MIN = 60  # en vivo, las últimas velas cerradas deben ser consecutivas (salvo fin de semana)
WARMUP = 100              # filas iniciales descartadas por el suavizado de Wilder (ATR/RSI)

# --- Target / gestión del trade (deben coincidir entre entrenamiento y ejecución) ---
SL_MULTIPLICADOR_ATR = 1.5
TP_MULTIPLICADOR_ATR = 3.0
MAX_BARS_EN_TRADE = 48    # barrera temporal N del target y time stop en vivo (12 h en M15)
COMISION_PUNTOS = 0       # comisión ida y vuelta en puntos de precio (MetaQuotes-Demo: 0; ECN típico: ~7 en EURUSD)
SLIPPAGE_PUNTOS = 1       # deslizamiento supuesto en cada ejecución (entrada y salida), en puntos
# El swap y el nivel mínimo de stops se leen del símbolo al entrenar (symbol_info).

# --- Entrenamiento ---
WF_VENTANAS = 5           # ventanas walk-forward; el histórico se divide en WF_VENTANAS + 3 segmentos
MIN_TRADES_UMBRAL = 30    # operaciones mínimas en validación para aceptar un umbral
MODEL_PATH = BASE_DIR / "models" / "random_forest_mt5.pkl"
MODELOS_HISTORICO_DIR = BASE_DIR / "models" / "historico"  # copia con fecha de cada modelo entrenado
# Período final reservado (fecha del servidor): no entra en el walk-forward ni se usa para elegir variantes.
# Todo lo anterior ya se usó en experimentos, así que empieza después del último. Moverlo consume el holdout.
HOLDOUT_DESDE = "2026-10-30"

# --- Ejecución ---
VOLUMEN = 0.01
RIESGO_PCT = None         # ej. 0.005 = 0.5% del equity por operación; None = VOLUMEN fijo
MAX_SPREAD_PUNTOS = 25
MAX_RETRASO_ENTRADA_S = 60
DESVIACION_SLIPPAGE = 20
AJUSTAR_SL_TP_A_EJECUCION = True  # tras entrar, el SL y el TP se recolocan a su distancia del precio real de ejecución
MAGIC_NUMBER = 202601
PERMITIR_CUENTA_REAL = False  # el bot se detiene en cuentas reales salvo que se active a propósito
EXIGIR_VENTAJA_SIGNIFICATIVA = True  # el bot no arranca sin ventaja significativa en el walk-forward
T_MINIMO_TEST = 2.0       # esperanza agregada mínima en errores estándar (~97.7% de confianza a una cola)
MIN_TRADES_TEST = 30      # trades fuera de muestra mínimos para que el t sea fiable
WF_MIN_PROP_VENTANAS_POSITIVAS = 0.6  # estabilidad: proporción mínima de ventanas con esperanza > 0
SEMILLAS_ROBUSTEZ = (42, 1, 2)  # el walk-forward se repite con cada semilla del bosque y todas deben pasar
MAX_LOTES = 1.0           # tope de volumen por operación, también con RIESGO_PCT
MAX_PERDIDA_DIARIA_PCT = 0.02  # sin nuevas entradas si el bot pierde hoy (día del servidor) este % del balance
MAX_DRAWDOWN_PCT = 0.10   # sin nuevas entradas si la equity cae este % desde su máximo (persistido)
GESTION = None            # gestión de la posición abierta: None (stop y objetivo fijos), "breakeven", "trailing" o "parcial"
GESTION_ACTIVACION_R = 1.0  # a este avance, en múltiplos del riesgo, se mueve el stop a la entrada o se cierra la mitad
CERRAR_AL_LIMITE = False  # al alcanzar uno de esos dos límites, el bot cierra además su posición abierta
# Riesgo a nivel de cuenta, para cuando varios bots (uno por activo) comparten cuenta. Con None no se aplica.
MAGICS_CUENTA = None          # magics de todos los bots de la cuenta; None = solo MAGIC_NUMBER
ESTADO_CUENTA_PATH = None     # máximo de equity compartido: el drawdown es el mismo para todos los bots
BLOQUEO_CUENTA_PATH = None    # turno entre bots: dos no comprueban la exposición y abren a la vez
ESPERA_TURNO_CUENTA_S = 10    # si otro bot no suelta el turno en este tiempo, no se entra en esta vela
MAX_POSICIONES_CUENTA = None  # posiciones abiertas a la vez entre todos los bots
MAX_RIESGO_ABIERTO_PCT = None  # pérdida si todas las posiciones tocan su SL, como fracción de la equity
# Reglas de cartera (medidas en cartera.py). Todas desactivadas por defecto.
SIMBOLOS_CORRELACIONADOS = None  # símbolos del grupo de este activo: sin entrada si otro ya tiene una posición del mismo lado
REDUCIR_EN_DRAWDOWN = False   # con la equity REDUCCION_DRAWDOWN_PCT o más por debajo de su máximo, el lote se multiplica por REDUCCION_FACTOR
REDUCCION_DRAWDOWN_PCT = 0.05
REDUCCION_FACTOR = 0.5
TAMANO_POR_CONFIANZA = False  # si la probabilidad supera su umbral en CONFIANZA_MARGEN o más, el lote se multiplica por CONFIANZA_FACTOR
CONFIANZA_MARGEN = 0.05
CONFIANZA_FACTOR = 1.5
SLEEP_S = 2
REINTENTO_CONEXION_S = 5
AVISO_SIN_CONEXION_S = 60  # un corte más corto solo queda en el log como información
REINTENTO_CIERRE_S = 10   # pausa entre intentos de cierre por time stop si uno falla
ESPERA_RECONCILIACION_S = 1  # espera antes de mirar las posiciones tras un resultado incierto de order_send
PENDIENTE_SIN_RASTRO_S = 600  # una entrada incierta sin rastro en posiciones, órdenes ni historial tras este
                              # tiempo se da por no enviada; None = exigir siempre resolución manual
LOG_PATH = BASE_DIR / "logs" / "trading_bot.log"
OPERACIONES_PATH = BASE_DIR / "logs" / "operaciones.csv"
ESTADO_PATH = BASE_DIR / "logs" / "estado_bot.json"
LOCK_PATH = BASE_DIR / "logs" / "trading_bot.lock"
PAUSA_PATH = BASE_DIR / "logs" / "PAUSA"  # si existe, no se abren nuevas entradas

# --- Credenciales (opcionales si la terminal ya tiene la sesión iniciada) ---
MT5_PATH = os.environ.get("MT5_PATH") or None
MT5_LOGIN = int(os.environ["MT5_LOGIN"]) if os.environ.get("MT5_LOGIN") else None
MT5_PASSWORD = os.environ.get("MT5_PASSWORD") or None
MT5_SERVER = os.environ.get("MT5_SERVER") or None
