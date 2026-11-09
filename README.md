# MT5 AI Trading Bot 🚀

Bot de trading algorítmico automatizado que combina Machine Learning (**Random Forest**) para predecir escenarios de alta probabilidad en los mercados financieros, integrándose en tiempo real con la terminal de **MetaTrader 5** mediante su API oficial de Python.

---

## 📐 Estructura del Proyecto

```text
.
├── config.py                 # Parámetros globales (símbolo, timeframe, riesgo, SL/TP, filtros)
├── features.py               # build_features(): cálculo único de features para entrenamiento y vivo
├── mt5_utils.py              # Conexión con la terminal y descarga de velas
├── train_model.py            # Extracción histórica, etiquetado Triple Barrier y entrenamiento Long/Short
├── trading_bot.py            # Bucle de ejecución en tiempo real (detección de nueva vela)
├── cuenta.py                 # Cuenta de Exness (demo o real) para el lanzador por activo, la ventana y train_exness
├── riesgo_cuenta.py          # Máximo de equity y turno de apertura compartidos por los bots de una cuenta
├── alertas.py                # Avisos por Telegram de ejecuciones, límites, errores y paradas
├── tendencia.py              # Backtest de seguimiento de tendencia por reglas (Donchian) en varios mercados
├── models/
│   ├── random_forest_mt5.pkl # Modelos Long/Short calibrados + umbrales, informe y costes del entrenamiento
│   └── historico/            # Copia con fecha de cada modelo entrenado
├── logs/                     # trading_bot.log, operaciones.csv, estado_bot.json, trading_bot.lock, PAUSA
├── tests/                    # Pruebas automáticas (pytest), con un MT5 simulado para el bot
├── .env.example              # Plantilla de credenciales de MT5 (copiar a .env, que nunca se versiona)
├── Entrenamiento.md          # Metodología de ML, features, target, costes, walk-forward y holdout
├── Ejecucion.md              # Lógica operativa en tiempo real, gestión de riesgos, límites y MT5
├── Plan_de_mejoras.md        # Mejoras por fases: hecho, descartado y pendiente
├── requirements.txt          # Dependencias de Python (versiones exactas)
├── requirements-dev.txt      # + pytest
└── README.md                 # Documentación general del proyecto
```

---

## 📚 Documentación Técnica

* **[Entrenamiento.md](Entrenamiento.md):** Detalla la preparación de datos, la convención temporal compartida con la ejecución, la ingeniería de características (retornos, ATR, RSI, SMAs), el etiquetado alineado al riesgo (*Triple Barrier*) con modelos Long y Short separados, los costes del backtest (spread, comisión, deslizamiento y swap), la validación walk-forward con purga, el período final reservado (*holdout*) y la calibración de umbrales.
* **[Plan_de_mejoras.md](Plan_de_mejoras.md):** Mejoras por fases (riesgo, features, régimen de mercado, re-entrenamiento y microestructura), con casillas para marcar lo hecho, lo descartado y lo pendiente, y el protocolo con el que se evalúa cada una.
* **[Ejecucion.md](Ejecucion.md):** Describe el flujo en tiempo real, la sincronización por eventos de nueva vela (*New Bar*) en hora del servidor, el dimensionamiento de lotes (fijo o por riesgo), el cálculo dinámico de SL/TP, el time stop, la reconciliación de órdenes, los límites de pérdida y las reglas de seguridad en MT5.

---

## 🚀 Flujo de Trabajo Recomendado

1. **Configuración Inicial:** Definir el símbolo (ej. `EURUSD`), timeframe (`M15`) y parámetros de riesgo en `config.py`. Si la terminal de MT5 ya tiene la sesión iniciada, `mt5.initialize()` no necesita credenciales; si hacen falta, se leen de variables de entorno (`.env`), nunca de `config.py`. Para entrenar con el histórico desde 2010 (`HISTORICO_DESDE`), en MT5 hay que poner **Herramientas → Opciones → Gráficos → "Máx. barras en la ventana" → Ilimitado** y reiniciar la terminal.
2. **Pruebas automáticas** (no necesitan MT5):
   ```bash
   python -m pytest tests
   ```
3. **Entrenamiento y Validación:**
   ```bash
   python train_model.py
   ```
   Extrae el histórico de MT5, calcula las variables predictivas, entrena los clasificadores Long y Short con validación walk-forward purgada (5 ventanas, 3 semillas), los evalúa en el período reservado (*holdout*) y guarda el modelo de la última ventana con sus umbrales y el informe completo.
4. **Ejecución en Cuenta Demo:**
   ```bash
   python trading_bot.py
   ```
   Carga el modelo, verifica que sus parámetros coinciden con `config.py` y espera la apertura de cada nueva vela para evaluar entradas con Stop Loss, Take Profit y time stop automáticos.

---

## 📊 Estado Actual

La infraestructura está completa: descarga, features, etiquetado Triple Barrier con costes (spread, comisión, deslizamiento y swap), walk-forward purgado con 3 semillas, holdout reservado, calibración, backtest mono-posición, y un bot con time stop, reconciliación de órdenes, verificación de cuenta, instancia única, límites de pérdida y registro de operaciones. La mecánica de órdenes se probó en Demo, y la recuperación ante fallos, con un MT5 simulado (`tests/`).

**No se ha encontrado ventaja.** Entrenado con todo EURUSD M15 desde 2010, el walk-forward (10,5 años fuera de muestra, de 2016 a 2026, con costes y purga por el final de cada etiqueta) da `t` = **−2.47 / −1.47 / −1.49** con las semillas 42 / 1 / 2, con profit factor 0.91. Con solo los últimos 4 años salía entre +1.4 y +1.7, y unas pocas velas de desplazamiento lo cambiaban más de 3 puntos: era suerte, no una ventaja. El holdout lleva 3 operaciones (todas perdedoras), lejos de las 30 necesarias. Una prueba independiente del mismo procedimiento, sin cambios, en 2010–2022 (8 años de test que ningún experimento en M15 había usado) da `t` = −2.21 / −1.25 / −0.30: confirma que no hay ventaja. Por eso el bot exige que el walk-forward pase con todas las semillas **y** que el modelo pase el holdout, que empieza el 2026-10-30 y necesita al menos 30 trades nuevos. Las features predicen muy bien *si* habrá movimiento (ROC-AUC 0.87–0.90), pero no *en qué dirección* (0.51–0.53). Se evaluaron además 13 features extra, 8 configuraciones de SL/TP/N, un breakout sin dirección y 12 combinaciones de mercado y timeframe, sin éxito (detalle en [Entrenamiento.md](Entrenamiento.md)). Por diseño, `trading_bot.py` se niega a arrancar mientras el modelo no tenga ventaja significativa (`EXIGIR_VENTAJA_SIGNIFICATIVA`).

**Seguimiento de tendencia, tampoco.** Las reglas clásicas de canal de Donchian (55/20, stop de 2 ATR), en D1 y H4, sobre 6 mercados (4 pares de Forex, oro y plata), con reglas y criterio fijados de antemano, dan de 2010 a 2023 `t` = −0.27 por trade y −0.09 en la cartera mensual. Ninguna de las 5 comprobaciones llega a `t` = 1 (`python tendencia.py`). Con una cartera diversificada de 13 mercados en 4 sectores (divisas, ETFs de índices, metales y bonos), con el mismo riesgo por sector, da `t` = −0.06 por trade (`python tendencia.py --diversificada`). MetaQuotes-Demo no tiene histórico limpio desde 2010 de índices CFD ni de materias primas. Detalle en la sección 6 de [Entrenamiento.md](Entrenamiento.md).

En cuenta Demo, el bot corrió 8 horas el 2026-11-01 sin errores. Evaluó 32 velas sin ninguna señal, y sus probabilidades coinciden con las recalculadas sobre el histórico.

---

## ⚠️ Advertencia de Riesgo
El trading automatizado en mercados de divisas y CFDs conlleva un alto nivel de riesgo. Este bot debe ser probado exhaustivamente en entornos de simulación y **cuentas Demo** antes de considerar cualquier uso con capital real.
