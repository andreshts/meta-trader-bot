# Documentación del Modelo y Entrenamiento 🧠

Este documento describe la metodología de preparación de datos, la ingeniería de características (*feature engineering*), la alineación del target con la ejecución en mercado y el proceso de validación/entrenamiento del modelo de Machine Learning.

---

## 📊 1. Fuente de Datos y Preprocesamiento

Los datos son extraídos directamente desde la terminal de **MetaTrader 5** mediante la API de Python (`mt5.copy_rates_from_pos`).

* **Temporalidad recomendada (Timeframe):** M15 o H1 (evita el ruido excesivo de M1/M5 y reduce el impacto relativo del spread).
* **Ventana de datos históricos:** 99,999 velas para entrenamiento (~4 años en M15; el tope es el `maxbars` de la terminal, 100,000, menos la vela en formación) y un búfer de mínimo **200 velas cerradas** para el cálculo de indicadores en vivo.
* **Solo velas cerradas:** Se descarga desde la posición `1` (`copy_rates_from_pos(symbol, tf, 1, n)`). La posición `0` es la vela en formación y nunca entra en el cálculo de features.
* **Hora del servidor:** El campo `time` es un entero (segundos) en la hora del servidor del broker, no en UTC real. Se conserva tal cual; si se añaden filtros de sesión, se expresan en la hora del broker.

### ⏱️ Convención Temporal (idéntica en entrenamiento y en vivo)

| Momento | Qué ocurre |
| :--- | :--- |
| Cierre de la vela `t` | Las features se calculan con datos hasta este cierre, inclusive. |
| Apertura de la vela `t+1` | Entrada (en vivo: orden a mercado al detectar la nueva vela). |
| Velas `t+1 … t+N` | Evaluación de las barreras del target. |
| Apertura de la vela `t+N+1` | Cierre por time stop (en vivo: al detectar esa vela). |

Las features **no** llevan un `shift(1)` adicional: la fila `t` solo contiene información disponible al cierre de `t`, y el target mira exclusivamente hacia adelante desde `t+1`. Aplicar `shift(1)` en entrenamiento y no en vivo (o al revés) desfasa una vela el modelo respecto a la ejecución.

> [!IMPORTANT]
> El cálculo de features vive en una única función `build_features(df, point)` (en `features.py`) que importan tanto `train_model.py` como `trading_bot.py`. La lógica nunca se duplica.

### 🔥 Periodo de Calentamiento (*Warm-up*)

* `dropna()` elimina las filas incompletas de los indicadores con ventana fija (la SMA50 es la más larga: ~50 filas). Los `NaN` iniciales no contaminan las medias móviles de pandas; simplemente se descartan.
* ATR y RSI usan el suavizado de Wilder (`ewm(alpha=1/14, adjust=False)`), que **no** produce `NaN` al inicio sino valores sesgados por el valor semilla. `dropna()` no los detecta, así que se descartan explícitamente las primeras **100 filas** (`df.iloc[100:]`), lo que deja el peso de la semilla por debajo del 0.1%.
* Las últimas `N` filas también se descartan: su target no tiene el horizonte completo.
* Se descartan las filas en las que el bot no entraría, para que no entrenen al modelo ni cuenten en el backtest: la vela de entrada (`t+1`) tiene un `spread` mayor que `MAX_SPREAD_PUNTOS`, o el SL quedaría más cerca del precio que el nivel mínimo de stops del símbolo (`trade_stops_level`) más el spread. Es una aproximación: el `spread` de la vela es un valor agregado y en vivo el filtro usa el del tick, que en el cambio de día suele ser mayor.
* En vivo, el búfer de 200 velas cubre el mismo calentamiento, por lo que el ATR y el RSI de la última vela coinciden en la práctica con los calculados sobre el histórico completo.

---

## 🛠️ 2. Ingeniería de Características (*Feature Engineering*)

Las features deben ser estrictamente **estacionarias** (no precios brutos) para que el modelo generalice ante diferentes regímenes de mercado. El escalado no es necesario para un Random Forest (los árboles son invariantes a la escala); dividir el RSI entre 100 es solo por legibilidad.

| Feature | Descripción | Cálculo | Justificación |
| :--- | :--- | :--- | :--- |
| `returns` | Rendimiento percentual | `close.pct_change()` | Estacionariedad del precio. |
| `dist_sma10` | Distancia relativa a SMA corta | `(close - SMA10) / SMA10` | Impulso y desviación de corto plazo. |
| `dist_sma50` | Distancia relativa a SMA media | `(close - SMA50) / SMA50` | Dirección y sesgo de la tendencia general. |
| `volatilidad` | Volatilidad rodante de retornos | `returns.rolling(10).std()` | Detección de compresión vs. expansión de volatilidad. |
| `atr_rel` | ATR relativo al precio | `ATR(14) / close` | Normaliza el rango real del precio en porcentaje. |
| `rsi` | Relative Strength Index (14) | `RSI(14) / 100` | Oscilador de momentum acotado entre 0 y 1. |
| `volume_ratio` | Ratio de volumen de ticks | `tick_volume / tick_volume.rolling(20).mean()` | Confirma si el movimiento tiene respaldo institucional. |
| `spread_ratio` | Costo relativo del spread | `(spread * point) / close` | Enseña al modelo a no operar en momentos de baja liquidez. |
| `hora_sin`, `hora_cos` | Hora del día (cíclica) | `sin/cos(2π · hora / 24)`, hora del servidor | La volatilidad del FX depende de la sesión (Asia, Londres, Nueva York, cambio de día). Codificada en círculo para que las 23:00 queden junto a las 00:00. |
| `dia_semana` | Día de la semana | `(time // 86400 + 3) % 7` (lunes = 0) | Diferencias de comportamiento entre días (inicio y cierre de semana). |

> [!NOTE]
> **Por qué estas tres y no otras:** se evaluaron 13 features candidatas en 9 combinaciones con el mismo walk-forward (momentum a 1 h/4 h/1 día, distancia a SMA200/SMA800, régimen de volatilidad, posición en el rango diario y forma de la vela). Solo las de tiempo mejoraron con claridad el ROC-AUC (de 0.52/0.55 a 0.55/0.56 en Long/Short) y fueron las más importantes del modelo combinado. El momentum empeoró el resultado y el resto no aportó discriminación, además de exigir un búfer en vivo de ~1000 velas. Ninguna combinación alcanzó ventaja significativa.

> [!NOTE]
> El `spread` de las velas históricas es un valor agregado por vela, no el spread en el instante de la entrada. Para que la feature tenga la misma distribución en vivo, `spread_ratio` se calcula siempre con el campo `spread` de la última vela cerrada. El spread real del tick solo se usa en el filtro `MAX_SPREAD_PUNTOS` de [Ejecucion.md](Ejecucion.md).

---

## 🎯 3. Definición de la Variable Objetivo (*Target*)

> [!IMPORTANT]
> **Alineación Target-Ejecución:** No se debe predecir la dirección de una sola vela aislada (`t+1`) si la orden va a operar con un Stop Loss de 1.5× ATR y un Take Profit de 3.0× ATR. El target debe reflejar las condiciones reales con las que se cerrará el trade.

Se adopta una variante adaptada del **Triple Barrier Method (Método de Triple Barrera)** con los mismos parámetros que usa la ejecución (`config.py`):

| Parámetro | Valor | Barrera |
| :--- | :--- | :--- |
| `TP_MULTIPLICADOR_ATR` | `3.0` | Superior (Take Profit). |
| `SL_MULTIPLICADOR_ATR` | `1.5` | Inferior (Stop Loss). |
| `MAX_BARS_EN_TRADE` (`N`) | `48` velas (12 horas en M15) | Temporal (Time Stop). |

El ATR de las barreras es el `ATR(14)` al cierre de `t`, el mismo que usa la orden en vivo para fijar SL y TP.

> [!NOTE]
> **Configuraciones evaluadas:** con las 11 features y el mismo walk-forward se probaron 8 combinaciones de SL/TP/N (1.5/3.0/48, 1.5/3.0/24, 1.0/2.0/24, 1.0/1.0/16, 1.0/1.0/32, 1.5/1.5/32, 2.0/2.0/48 y 1.0/1.5/24). Ninguna alcanzó ventaja significativa: el mejor `t` fue +1.67 (1.0/2.0/24, con una ventana de 0 trades y otra de 3), aproximadamente lo que se espera por azar al probar tantas variantes. Se mantiene 1.5/3.0/48: cambiar a la "mejor" sería elegir ruido.
>
> Con horizontes cortos el ROC-AUC sube mucho (0.62/0.63 con 1.5/3.0/24) sin mejorar la esperanza. Con `N` corto muchas etiquetas `0` son time stops (el precio no se movió), y el modelo aprende a distinguir *si habrá movimiento* (volatilidad, sobre todo por la hora), no *en qué dirección*. Eso sube el AUC pero no el beneficio.

> [!NOTE]
> **Movimiento vs. dirección:** separando las dos preguntas (walk-forward, 11 features), el modelo predice muy bien *si* el precio recorrerá ±1.5 ATR en 24–48 velas (ROC-AUC 0.87–0.90, estable en las 5 ventanas), pero apenas *hacia dónde* (0.51–0.53). Una estrategia que no necesita dirección, un breakout OCO con buy stop y sell stop a ±0.5 ATR, pierde −0.074 R ± 0.018 por trade sin filtro (t = −4.0) y −0.127 R filtrada por el modelo. En EURUSD M15 los rompimientos no se prolongan lo suficiente para cubrir el spread. Conclusión: con estas features la ventaja tendría que venir de la dirección, y no la hay.

> [!NOTE]
> **Otros mercados:** el mismo pipeline, sin ajustes, en 12 variantes fijadas de antemano (EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD y XAGUSD, en M15 y H1). Ninguna tiene ventaja: el mejor `t` es +0.99 (EURUSD M15) y varias pierden de forma significativa (XAGUSD M15 t = −3.52, XAGUSD H1 −2.21, AUDUSD M15 −2.19). La dirección sigue siendo casi impredecible: ROC-AUC 0.495–0.517 en H1 y 0.50–0.56 en M15. En M15 es algo mayor justo en los pares más caros de operar (AUDUSD, GBPUSD), lo que apunta a ruido de microestructura que el spread se come. En H1 el coste baja a la mitad (−0.03 R por trade), pero la dirección vale 0.50.
>
> **Filtro de spread en el backtest:** en XAGUSD M15, un festivo con precio congelado (ATR 0.0023) y spread de 129 puntos produjo un trade de −37 R que el bot en vivo nunca habría abierto. Desde entonces `preparar_datos` quita esas entradas (ver sección 1).
>
> **Robustez:** con el filtro, el walk-forward de EURUSD M15 dio +0.088 R ± 0.044 (t = 1.98), justo por debajo del umbral. No se sostiene. Con cambios que no deberían importar (otra descarga de las mismas velas, otra semilla del Random Forest, 4 o 6 ventanas), `t` va de −2.12 a +1.25, con media +0.13. Solo la semilla, con los mismos datos, lo mueve de −0.22 a +0.68. El modelo es inestable y la elección de umbrales amplifica esa inestabilidad. Un resultado aislado cerca de `t = 2` no significa nada sin esta comprobación.

### Dos Modelos Binarios: Long y Short

Una etiqueta `0` del modelo de compra significa "la compra no ganó", **no** "la venta gana". Usar `1 - P(compra)` como probabilidad de venta abriría cortos con un modelo que nunca aprendió cortos. Por eso se entrenan dos clasificadores independientes, cada uno con su propia triple barrera.

Las velas de MT5 son precios **bid**: una compra entra al ask y se cierra al bid; una venta entra al bid y se cierra al ask. `spread` es el de cada vela, en puntos.

| | Modelo Long (`y_long`) | Modelo Short (`y_short`) |
| :--- | :--- | :--- |
| Precio de entrada | `open[t+1] + spread * point` (ask) | `open[t+1]` (bid) |
| Take Profit | `entrada + 3.0 * ATR`; tocado si `high >= TP` | `entrada - 3.0 * ATR`; tocado si `low + spread * point <= TP` |
| Stop Loss | `entrada - 1.5 * ATR`; tocado si `low <= SL` | `entrada + 1.5 * ATR`; tocado si `high + spread * point >= SL` |
| Etiqueta `1` | Toca TP antes que SL dentro de `N` velas | Toca TP antes que SL dentro de `N` velas |
| Etiqueta `0` | Toca SL primero, o vence el tiempo sin tocar TP | Toca SL primero, o vence el tiempo sin tocar TP |

Reglas de desempate:
* **Vela ambigua:** si en una misma vela se tocan TP y SL, cuenta como SL. Sin datos de ticks no se sabe cuál ocurrió primero, y el criterio conservador evita inflar la tasa de acierto.
* **Time stop:** el trade se cierra en la **apertura de la vela `t+N+1`** (bid para el long, ask para el short) y la etiqueta es `0`, aunque esté en ganancia. Es el precio al que cierra el bot, que envía la orden al detectar esa vela. Tras un fin de semana, ese precio puede diferir bastante del cierre del viernes.

### Costes del Backtest

El resultado `R` de cada trade descuenta:

| Coste | Origen | Valor en EURUSD (MetaQuotes-Demo) |
| :--- | :--- | :--- |
| Spread | `spread` de cada vela: entrada al ask en compras y salida al ask en ventas | Variable por vela |
| Comisión | `COMISION_PUNTOS`, ida y vuelta | `0` en esta demo (comprobado en el historial de deals); en cuentas ECN ronda 7 puntos |
| Deslizamiento | `SLIPPAGE_PUNTOS`, en la entrada y en la salida | `1` punto supuesto por ejecución; `logs/operaciones.csv` registra el real |
| Swap | `symbol_info` (`swap_long`, `swap_short` en puntos) por cada rollover a las 00:00 del servidor tras un día de lunes a viernes, triple el día `swap_rollover3days` | −0.7 / −1.0 puntos por noche, triple los miércoles |

`train_model.py` lee el swap y `trade_stops_level` del símbolo al entrenar, y los guarda en `informe["costes"]`. Solo se modela el swap expresado en puntos (`swap_mode = 1`); con otros modos se avisa y se ignora.

*(Alternativa: un único modelo multiclase `1: Buy`, `0: Hold`, `-1: Sell`. Con estas barreras ambas señales son excluyentes, pero dos modelos permiten calibrar el umbral de cada lado por separado.)*

---

## 🧪 4. Validación Temporal y Prevención de Fugas de Datos

> [!WARNING]
> En series temporales financieras **está estrictamente prohibido usar validación cruzada aleatoria (`train_test_split(shuffle=True)`)**. Mezclar datos del futuro con el pasado produce resultados artificialmente optimistas que fracasan en el mercado real.

Se aplica una **validación walk-forward anclada** (`WF_VENTANAS = 5`). El histórico se divide en `WF_VENTANAS + 3` segmentos iguales (8 segmentos de ~6 meses con 99,999 velas M15) y en cada ventana `k`:

| Bloque | Segmentos | Uso |
| :--- | :--- | :--- |
| **Train** | `0 … k+1` (crece en cada ventana) | Aprendizaje del modelo. |
| **Validation** | `k+2` | Calibración de probabilidades y elección de umbrales. |
| **Test** | `k+3` (la última ventana absorbe el resto) | Fuera de muestra: backtest con los umbrales ya fijados. |

Los 5 tests son consecutivos y no se solapan: juntos cubren los últimos 5/8 del histórico (~2.5 años) y ningún trade se evalúa con un modelo que haya visto esos datos. Un único split puede salir bien por casualidad; el walk-forward mide si el **procedimiento** funciona en periodos distintos.

El modelo que se guarda es el de la última ventana, entrenado con los datos más recientes.

### Purga entre Bloques (*Purging*)

La etiqueta de la fila `t` usa precios hasta `t+N`, así que las últimas `N` filas de cada bloque dependen de precios del bloque siguiente. Se eliminan **`N` velas (48)** en cada frontera Train→Validation y Validation→Test.

### Período Final Reservado (*Holdout*)

El walk-forward se ha reutilizado en decenas de experimentos (features, SL/TP, mercados, filtros), así que ya no es una prueba limpia: alguna variante puede salir bien por azar. Por eso hay un **holdout** que no se usa para elegir nada:

* `HOLDOUT_DESDE = "2026-10-30"` (fecha del servidor). Todo lo anterior ya se usó en los experimentos, así que el holdout empieza después del último.
* El walk-forward solo usa datos anteriores a esa fecha, menos `N` filas de purga, para que ninguna etiqueta mire dentro del holdout.
* El modelo guardado, entrenado solo con datos anteriores, se evalúa una vez en el holdout con sus umbrales ya fijados.
* El bot exige que el holdout también pase: al menos 30 trades y `t >= 2`. Mientras no se acumulen suficientes datos nuevos, el bot no arranca, y es lo correcto.
* **Mover `HOLDOUT_DESDE` consume el holdout:** los datos que pasan al walk-forward dejan de ser una prueba limpia. Solo debe hacerse después de tomar una decisión con él, y fijando un holdout nuevo hacia delante.
* Como la descarga son siempre las últimas 99,999 velas, el inicio del walk-forward avanza con el tiempo y sus resultados cambian un poco al reentrenar.

---

## ⚙️ 5. Algoritmo y Configuración

Se utiliza `RandomForestClassifier` de Scikit-Learn por su capacidad no lineal y tolerancia al ruido, reforzado con restricciones para evitar la memorización. Se entrena una instancia para Long y otra para Short con los mismos hiperparámetros:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=150,          # Número de árboles para estabilidad en el ensamble
    max_depth=5,               # Poda de profundidad para prevenir sobreajuste
    min_samples_split=50,      # Requiere masa crítica de muestras para crear ramas
    min_samples_leaf=25,       # Evita nodos terminales atípicos o ruidosos
    max_features="sqrt",       # Submuestreo de características por árbol
    random_state=42,
    n_jobs=-1
)
```

### Calibración y Umbrales

Con `max_depth=5` y `min_samples_leaf=25`, las probabilidades del Random Forest se concentran cerca de la proporción de etiquetas `1`. Con TP a 3×ATR esa proporción depende mucho del horizonte. En EURUSD M15 (oct 2024 a oct 2026) es ~15% con `N=12`, ~23% con `N=24` y ~28% con `N=48`, frente al 33.3% necesario para no perder. Por eso se usa `N=48`: el modelo solo tiene que mejorar un poco la tasa base, no duplicarla. Un umbral fijo como `0.60` podría no alcanzarse nunca.

1. **Calibrar** las probabilidades de cada modelo en el Validation Set (sigmoide o isotónica) para que sean interpretables.
2. **Elegir** `umbral_long` y `umbral_short` en el Validation Set, maximizando la esperanza por trade (en múltiplos de riesgo, R) con un mínimo de operaciones (`MIN_TRADES_UMBRAL`, 30) para no ajustar ruido. Cada umbral del grid (0.20–0.90) se evalúa con un backtest mono-posición de ese lado. Si ninguno da esperanza positiva, el lado queda desactivado (`None`) y el bot no opera en esa dirección.
3. **Medir** en el Test de la ventana con esos umbrales ya fijados. El agregado de todas las ventanas es lo que decide el despliegue.

### Artefacto del Modelo

`models/random_forest_mt5.pkl` guarda un único diccionario con todo lo que la ejecución necesita para reproducir las condiciones del entrenamiento. Los modelos y umbrales son los de la última ventana, y `informe["walk_forward"]` guarda el resultado de todas:

```python
bundle = {
    "model_long": model_long,
    "model_short": model_short,
    "umbral_long": 0.42,          # ejemplo: sale del Validation Set
    "umbral_short": 0.40,         # ejemplo: sale del Validation Set
    "features": FEATURES,         # lista y orden exacto de columnas
    "sl_mult": 1.5,
    "tp_mult": 3.0,
    "n_barras": 48,
    "symbol": "EURUSD",
    "timeframe": "M15",
}
```

`trading_bot.py` comprueba al arrancar que estos parámetros coinciden con `config.py` y se detiene si no (ver [Ejecucion.md](Ejecucion.md)). Cada entrenamiento guarda además una copia con fecha en `models/historico/`, para poder volver a un modelo anterior.

### Métricas de Evaluación Clave:
* **Precision en clase positiva frente al punto de equilibrio:** con relación 1:2, un trade es rentable en promedio si acierta más del **33.3%** (`1 / (1 + 2)`). Spread, comisión, deslizamiento y swap ya se descuentan del R de cada trade (ver *Costes del Backtest*), así que el listón es ese 33.3% sobre la etiqueta, no el 50%. Se prioriza la precisión sobre el recall (es preferible operar pocas veces pero con alta probabilidad de acierto).
* **Curva ROC-AUC:** Asegurar capacidad discriminativa superior a 0.55 en el set de prueba out-of-sample.
* **Payoff Ratio, Profit Factor y Drawdown simulados:** Comprobar que la esperanza matemática (considerando comisiones y spread) sea positiva, con un número de operaciones suficiente para que el resultado no sea casualidad.
* **Significancia de la esperanza (criterio de despliegue):** una media positiva no basta. Con resultados de +2 R / −1 R, cada trade tiene una desviación de ~1.4 R, así que con 150 trades el error estándar de la media es ~±0.11 R. `train_model.py` calcula `t = esperanza / error estándar` sobre **todos los trades fuera de muestra del walk-forward**, y `trading_bot.py` solo arranca si `t >= 2` con al menos 30 trades **y** la esperanza es positiva en al menos el 60% de las ventanas (`WF_MIN_PROP_VENTANAS_POSITIVAS`). Todo el walk-forward se repite con cada semilla de `SEMILLAS_ROBUSTEZ` (42, 1, 2) y **todas** deben cumplirlo, porque la semilla del bosque basta para mover `t` de −0.22 a +0.68 con los mismos datos. El modelo guardado es el de la primera semilla, y además debe pasar en el holdout (sección 4). En las pruebas con ruido puro, una ventana suelta dio +0.39 R (t = 1.65) por azar; el agregado (t = −1.32) y la estabilidad (20% de ventanas positivas) lo bloquearon.
