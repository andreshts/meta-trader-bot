# Plan de Mejoras por Fases 🗺️

Seguimiento de las mejoras propuestas para el bot: qué está hecho, qué se probó y descartó, y qué falta. Marca cada casilla al terminarla (`- [x]`) y anota el resultado en la línea de debajo.

**Leyenda:** `[x]` hecho · `[ ]` pendiente · ❌ probado y descartado (no mejoró) · 🔒 bloqueado por otra fase

---

## 📏 Protocolo para Evaluar Cualquier Mejora del Modelo

Una mejora del modelo (feature, régimen, filtro, re-entrenamiento) solo cuenta como **hecha** si pasa este protocolo. Si no lo pasa, se marca ❌ con su resultado: un "no mejora" bien medido también es un resultado.

1. **Fijar la variante antes de mirar resultados:** qué se añade, con qué parámetros y qué se espera. No se ajustan parámetros viendo el test.
2. **Walk-forward desde 2010** (`train_model.py`: 5 ventanas, purga por el final de cada etiqueta, 3 semillas), comparando contra la línea base, **con los mismos datos**.
3. **Criterio:** que suba el ROC-AUC de forma estable entre ventanas **y** que la esperanza agregada mejore con las 3 semillas. Solo se considera ventaja real con `t >= 2` en todas las semillas, y además en el holdout (≥ 30 trades).
4. **Contar las pruebas.** Cada variante evaluada sobre los mismos datos aumenta la probabilidad de un falso positivo. Llevamos **unas 49 variantes** (38 en EURUSD desde 2010 y 11 en el banco de experimentos sobre los 5 activos de Exness). Con tantas, un `t` cercano a 2 aislado no significa nada (ver [Entrenamiento.md](Entrenamiento.md)).
5. **No tocar el holdout** (desde `HOLDOUT_DESDE = 2026-10-30`) para elegir variantes. Solo decide el despliegue final.
6. **Pruebas automáticas** (`python -m pytest tests`) para el código nuevo, en particular contra fugas de datos del futuro, y documentar el resultado en [Entrenamiento.md](Entrenamiento.md).

**Línea base actual** (EURUSD M15 desde 2010, 11 features, SL 1.5 / TP 3.0 / N 48; modelo del 2026-10-31):
- **Walk-forward con 10,5 años de test** (2016-04 a 2026-10): `t` = **−2.47 / −1.47 / −1.49** con las semillas 42 / 1 / 2. Profit factor 0.91 y −0.064 R ± 0.026 por trade con la semilla 42.
- ROC-AUC de la última ventana: 0.552 / 0.557.
- Holdout: 2 operaciones.
- Ninguna ventaja demostrada: es la referencia contra la que comparar cada mejora.
- Antes, con solo 2022–2026, salía `t` entre +1.4 y +1.7, y la prueba independiente en 2010–2022 dio −2.21 / −1.25 / −0.30.

---

## ✅ Fase 0 — Base Construida

- [x] Descarga desde MT5 con inicio fijo (`HISTORICO_DESDE = 2010-01-01`), sincronización y aviso si la terminal recorta el histórico.
- [x] Etiquetado Triple Barrier con costes reales: spread, deslizamiento, swap y cierre por tiempo en la apertura siguiente.
- [x] Modelos Long/Short separados con `predict_proba` calibrado y umbrales elegidos en validación (no `predict` 0/1).
- [x] Walk-forward anclado con purga por el final de la etiqueta, 3 semillas y holdout reservado.
- [x] Bloqueo de despliegue: el bot no arranca sin ventaja significativa y robusta.
- [x] Bot: time stop, reconciliación de órdenes inciertas, intención guardada antes de enviar, instancia única, verificación de cuenta y validación de velas antes de entrar.
- [x] Límites de riesgo: pérdida diaria, drawdown persistido, `MAX_LOTES` y pausa manual (`logs/PAUSA`).
- [x] Suite de pruebas automáticas (109 tests) con MT5 simulado y red de seguridad sobre `logs/` y `models/`.

---

## 🛡️ Fase 1 — Riesgo y Ejecución

No depende de encontrar ventaja: protege la cuenta en cualquier caso.

- [ ] **1.1 Validación operativa en Demo** (desde el domingo 17:00 CR)
  - Hecho cuando: varias semanas de operación sin incidencias y una comparación de `logs/operaciones.csv` (deslizamiento, spread y swap reales) con lo que supone el backtest.
  - Resultado parcial (2026-11-01):
    - El bot corrió 8 h en dos sesiones (18:27–20:27 y 21:02–03:01 CR) sin errores ni desconexiones.
    - Evaluó 32 velas, todas a tiempo, y no hubo ninguna señal.
    - Las probabilidades en vivo coinciden con las recalculadas sobre el histórico (diferencia ≤ 0.0005, el redondeo del log).
    - Pendiente: aún no se ha enviado ninguna orden, y al arrancar se lee una vela vieja de la caché (sin efecto, porque se descarta por retraso).
- [x] **1.2 Tamaño de posición por volatilidad** (`RIESGO_PCT`, en [trading_bot.py](trading_bot.py)): `lotes = equity × RIESGO_PCT / (distancia al SL / tick_size × tick_value)`, con tope `MAX_LOTES`.
- [ ] **1.3 Activar el lote por riesgo en Demo**
  - Hoy está en `None` (0.01 fijo).
  - Hecho cuando: se elige un `RIESGO_PCT` (por ejemplo 0.25%), se activa en la Demo y se comprueba en `operaciones.csv` que el riesgo en dinero es constante entre operaciones.
  - Resultado:
- [x] **1.4 Filtro de calendario económico** (2026-11-10: servicio [mql5/RF_Calendario.mq5](mql5/RF_Calendario.mq5), [calendario.py](calendario.py) y filtro en el lanzador, activo por defecto)
  - Resultado: el servicio entrega 17.398 eventos de USD y EUR desde 2021-05 (3.567 de alto impacto, unos 620 al año), con las horas en UTC comprobadas con el NFP. En el backtest ([experimentos.py](experimentos.py), 5 activos, 3 semillas) **no da ventaja**: como filtro (`noticias_filtro`), `t` medio por activo −1.54 / +0.32 / +0.16 / −0.75 / −0.61 (mejora a la base en el S&P 500 y el DAX); como feature (`noticias_feature`, tiempo hasta la próxima noticia y desde la última), −0.14 / −0.48 / −1.26 / −0.62 / −0.47 (mejora solo en el DAX). El ROC-AUC no cambia. El filtro se mantiene como protección de la ejecución: el backtest usa el spread de la vela y no modela el deslizamiento en una noticia.: sin nuevas entradas de 15 minutos antes a 15 minutos después de noticias de alto impacto (NFP, FOMC, IPC, BCE).
  - La API de Python de MT5 **no** da acceso al calendario: está solo en MQL5 (`CalendarValueHistory`). Opciones: un pequeño Expert Advisor en MQL5 que exporte los eventos a un CSV que lea el bot, o una fuente externa.
  - Hecho cuando: el bot lee los eventos, los registra en el log, bloquea entradas en la ventana y tiene pruebas automáticas. Para medir su efecto en el backtest hace falta el histórico de eventos; si no lo hay, se valida solo en Demo.
  - Resultado:
- ❌ **1.6 Gestión de la posición abierta** (`config.GESTION`, `--gestion`; 2026-11-10). Tres reglas, fijadas antes de ver resultados, que no cambian la señal sino la salida. El stop solo se mueve al cerrar una vela y nunca se aleja:
  - `breakeven`: stop a la entrada cuando el precio llega a +1 R;
  - `trailing`: stop a 1,5 ATR del mejor precio alcanzado;
  - `parcial`: media posición cerrada a mercado a +1 R y stop a la entrada para el resto.
  - Resultado: ninguna mejora de forma consistente ([experimentos.py](experimentos.py), 5 activos, 3 semillas; `t` medio por activo en EUR/USD, oro, S&P 500, Nasdaq y DAX; la base da −0.08 / +0.60 / −0.48 / −0.47 / −1.91):
    - `breakeven`: −0.22 / +0.18 / −0.40 / −0.36 / −2.18;
    - `trailing`: +0.26 / −1.14 / 0 / −1.11 / 0 (en el S&P 500 y el DAX ningún umbral es rentable en validación y deja de operar);
    - `parcial`: −0.40 / −0.22 / −0.49 / −0.53 / 0.
  - Las tres quedan implementadas en el bot y probadas (la regla del bot coincide con la de la simulación), **desactivadas por defecto**.
- [x] **1.7 Reglas de cartera** (2026-11-10; [cartera.py](cartera.py) las mide sobre los trades fuera de muestra del walk-forward de los 5 activos a la vez, 3 semillas, y el bot las aplica con `--limite-correlacion`, `--reducir-en-drawdown` y `--por-confianza`, desactivadas por defecto). Fijadas antes de ver resultados:
  - **correlación:** sin entrada si otro índice (S&P 500, Nasdaq, DAX) ya tiene una posición del mismo lado;
  - **reducción:** medio lote mientras la cartera esté 10 R o más bajo su máximo (en el bot, la equity un 5 % o más);
  - **confianza:** lote y medio si la probabilidad supera su umbral en 0.05 o más.
  - Resultado (medias de las 3 semillas; unos 2.200 trades):

    | Regla | Total R | `t` por semilla (1 / 2 / 42) | Drawdown máx. R |
    | :--- | ---: | :--- | ---: |
    | base | −31.6 | −0.90 / −0.85 / +0.22 | 99.8 |
    | correlación | −28.8 | −0.90 / −0.85 / +0.34 | 99.7 |
    | reducción | −22.0 | −1.16 / −0.72 / −0.05 | 57.5 |
    | confianza | −27.9 | −0.80 / −0.77 / +0.20 | 96.7 |
    | las tres | −15.2 | −1.11 / −0.56 / +0.25 | 56.3 |

  - La reducción es la única con un efecto claro: el drawdown máximo baja un 42 %. No cambia el signo del resultado: ninguna regla de tamaño lo hace. La correlación descarta un 1 % de los trades y la confianza apenas mueve nada: la probabilidad por encima del umbral no distingue mejores trades.
- [ ] 🔒 **1.5 Criterio de Kelly**: con una ventaja nula o negativa, Kelly da un tamaño de 0. Bloqueado hasta que exista una ventaja demostrada (Fase 4) y un intervalo de confianza para estimarla.

---

## 🧪 Fase 2 — Features Nuevas

Cada una se evalúa con el protocolo, por separado y luego en combinación con las que mejoren.

### Ya evaluadas

- [x] **Hora del día (seno/coseno) y día de la semana.** Integradas: subieron el ROC-AUC de 0.52/0.55 a 0.55/0.56.
- [x] **ATR relativo al precio** (`atr_rel`). Integrada desde el inicio.
- ❌ **Momentum a 1 h, 4 h y 1 día** (`mom_4`, `mom_16`, `mom_96`). Empeoró el resultado (`t` = −1.67).
- ❌ **Tendencia de temporalidad superior aproximada** (`dist_sma200` ≈ 2 días, `dist_sma800` ≈ 8 días). No mejoró el ROC-AUC.
- ❌ **Régimen de volatilidad** (`atr_regimen` = ATR / su media de 500 velas, `vol_ratio`). No mejoró.
- ❌ **Posición en el rango diario y forma de la vela.** No mejoró.
- ❌ **2.3 ADX(14) con +DI y −DI** (`adx`, `plus_di`, `minus_di` de Wilder en `features.adx_wilder()`, con pruebas anti-fuga). Mejoró la esperanza respecto a la línea base en las 3 semillas (`t` = −1.89 / −0.20 / +0.00 frente a −2.47 / −1.47 / −1.49), pero no subió el ROC-AUC (0.542 / 0.557 en la última ventana; media 0.559 / 0.576) ni alcanzó ventaja significativa. La función queda implementada y probada en `features.py` para la Fase 3.
- ❌ **2.4 Multi-timeframe real H1 y H4 cerrado** (`rsi_h1`, `dist_sma20_h1`, `rsi_h4`, `dist_sma20_h4` en `features.mtf_cerrado()`, usando exclusivamente bloques superiores cerrados al cierre de `t`, verificado con test de perturbación intra-hora). No subió el ROC-AUC (0.547 / 0.557 en la última ventana; media 0.555 / 0.573) y dio `t` = −1.11 / −2.25 / −1.60. Combinada con 2.3 (18 features) dio `t` = −0.01 / −0.52 / −1.48 y ROC-AUC 0.545 / 0.557.

### Pendientes

- ❌ **2.1 Banderas de sesión**: Asia, Londres, Nueva York y el solape Londres–NY, por la hora local de cada plaza (con sus cambios de horario).
  - Resultado: sin ventaja. Evaluada el 2026-11-10 con el banco de experimentos ([experimentos.py](experimentos.py)) en los 5 activos de Exness (unas 100.000 velas M15 cada uno, walk-forward hasta 2025-12-31, 3 semillas). `t` medio por activo (EUR/USD, oro, S&P 500, Nasdaq, DAX): −0.86 / +0.07 / −0.78 / +0.28 / −2.35; mejora a la base solo en el Nasdaq y el ROC-AUC no cambia. `hora_sin` y `hora_cos` ya recogen esa información.
- ❌ **2.2 Squeeze**: anchura de las Bandas de Bollinger(20, 2) entre la anchura del Canal de Keltner(20, 1.5×ATR), y una bandera de squeeze activo.
  - Resultado: sin ventaja. Evaluada el 2026-11-10 con el banco de experimentos ([experimentos.py](experimentos.py)) en los 5 activos de Exness (unas 100.000 velas M15 cada uno, walk-forward hasta 2025-12-31, 3 semillas). `t` medio por activo: −0.30 / +0.50 / −0.12 / −0.41 / −1.50; mejora algo a la base en 3 de 5, pero ninguna semilla de ningún activo llega a `t` = 1 y el ROC-AUC no cambia.

---

## 🧭 Fase 3 — Régimen de Mercado

Depende de 2.3 (ADX). Clasifica el mercado en tendencia alcista, tendencia bajista o rango.

- [ ] **3.1 Clasificador por reglas**: ADX y +DI/−DI para tendencia y dirección, y cambio de volatilidad para separar rango y expansión. Umbrales fijados de antemano (por ejemplo ADX > 25 = tendencia) y no ajustados viendo el test.
  - Hecho cuando: está implementado con pruebas y se mide qué parte de las velas cae en cada estado.
  - Resultado:
- [ ] **3.2 Hidden Markov Model** (alternativa a 3.1, con la dependencia `hmmlearn`).
  - ⚠️ **Riesgo de fuga:** el HMM se ajusta **solo con datos de train** de cada ventana, y el estado de la vela `t` se calcula con el **filtrado hacia delante** (solo datos hasta `t`). Viterbi y el suavizado usan datos futuros y no valen.
  - Hecho cuando: hay pruebas contra fugas y estados estables entre ventanas.
  - Resultado:
- [ ] **3.3 Usos del régimen**, cada uno evaluado con el protocolo:
  - ❌ a) como feature más del modelo (régimen por reglas con ADX > 25 y el DI dominante, más el ADX): `t` medio por activo −2.12 / +0.11 / −1.40 / +0.11 / −2.45; mejora a la base solo en el Nasdaq.
  - ❌ b) como filtro, sin entradas en rango (ADX ≤ 25): −2.52 / +0.33 / −0.89 / −0.54 / −0.40; mejora solo en el DAX, donde casi deja de operar (74 trades).
  - Las dos, evaluadas el 2026-11-10 con [experimentos.py](experimentos.py) en los 5 activos de Exness.
  - [ ] c) con SL/TP distintos por régimen. Multiplica las variantes probadas, así que hay que contarlas en el protocolo.
  - Resultado:

---

## 🔁 Fase 4 — Modelo y Re-entrenamiento

- [x] **4.1 Probabilidades con umbral de confianza** (`predict_proba`, calibración y umbral elegido en validación).
  - Nota: un umbral fijo de 65–70% **no sirve aquí**. Solo un ~28% de las operaciones llega al TP, y para no perder basta acertar el 33.3%, así que el modelo casi nunca daría 0.65. El umbral tiene que salir de la validación, como ya se hace.
- [x] **4.2 Walk-forward anclado** (train creciente) para evaluar.
- [ ] **4.3 Ventana deslizante frente a anclada**: entrenar solo con los últimos X meses (por ejemplo 12) y comparar con la anclada usando el protocolo.
  - 3 meses de M15 son unas 6.000 filas, pocas para un Random Forest con esta tasa de aciertos. Conviene probar 6, 12 y 24 meses, fijados de antemano.
  - Resultado (2026-11-10, ❌ con 12 meses): en los 5 activos de Exness, `t` medio −0.76 / +0.67 / −1.60 / −1.17 / −0.72; mejora a la base en el oro (+0.07) y el DAX. Con unas 100.000 velas por activo, 6 meses dejan muy pocas filas y 24 meses son casi la ventana anclada, así que no se evaluaron.
- [ ] **4.4 Robustez por desplazamiento de ventanas**: repetir el walk-forward con los cortes desplazados (por ejemplo, 3 desplazamientos de un mes) además de las 3 semillas, y exigir que todos pasen.
  - Motivo: desplazar la ventana 92 velas movió el `t` más de 3 puntos.
  - Hecho cuando: está integrado en `train_model.py` y en el bloqueo del bot, con pruebas.
  - Resultado:
- [ ] 🔒 **4.5 Re-entrenamiento periódico automático** (por ejemplo, semanal con el Programador de tareas): reentrena, guarda la copia en `models/historico/` y el bot solo carga el modelo nuevo si pasa el bloqueo.
  - Bloqueado hasta que haya un modelo con ventaja: reentrenar un modelo sin ventaja no la crea.
  - Hecho cuando: está programado, con registro de cada reentrenamiento y vuelta automática al modelo anterior si el nuevo no pasa.

---

## 🔬 Fase 5 — Microestructura (Datos más allá de las Velas)

La información que no está en OHLC es la más prometedora, porque la dirección no se puede predecir con velas (ROC-AUC 0.51–0.53 en 4 mercados y 2 timeframes).

- [x] **5.1 Inventario de ticks** ([inventario_ticks.py](inventario_ticks.py), 2026-11-10, en Exness-MT5Trial11 con los 5 activos): un día de muestra por semestre desde 2019 y los tres últimos miércoles.
  - **Profundidad:** el primer tick disponible es del 2026-01-01 en todos los activos; antes no hay nada (comprobado mes a mes en 2025, con reintentos). Son unos 10 meses.
  - **Campos:** solo cotizaciones, bid y ask. Ningún tick trae `last`, volumen ni las banderas de compra o venta, así que no hay operaciones reales con las que medir un desequilibrio.
  - **Densidad, muy desigual en el tiempo:**

    | Activo | 2026-01-14 | 2026-07-08 | 2026-11-04 |
    | :--- | ---: | ---: | ---: |
    | EUR/USD | 26/min | 26/min | 49/min |
    | Oro | 319/min | 309/min | 274/min |
    | S&P 500 | 104/min | 78/min | 442/min |
    | Nasdaq | 169/min | 206/min | 3.145/min |
    | DAX | 29/min | 102/min | 117/min |

  - **Coste:** unos 450 MB de disco por 5 días de muestra de 4 activos (la terminal guarda el mes entero de cada día pedido).
- [ ] 🔒 **5.2 Features de ticks**: desequilibrio de ticks al alza/baja, velocidad de ticks, dinámica del spread dentro de la vela y saltos de precio.
  - Bloqueado por 5.1 (2026-11-10): con los ticks de Exness no se puede evaluar. Solo hay 10 meses, que además son exactamente el holdout del banco de experimentos (desde 2026-01-01); la densidad cambia hasta 18 veces dentro del mismo año, así que una feature de velocidad de ticks mediría el cambio del proveedor y no el mercado; y sin operaciones, el desequilibrio solo se puede aproximar por el sentido de cada cambio de precio.
  - Para desbloquearlo hace falta otra fuente con años de ticks (por ejemplo un histórico descargable de cotizaciones), comprobando antes que sus velas M15 coinciden con las de Exness, o grabar los ticks de Exness desde ahora y esperar a tener al menos dos años.
- ❌ **5.3 Libro de órdenes** (`market_book_get`): en Forex retail normalmente no está disponible. Comprobar si el broker lo ofrece; si no, se descarta.
  - Resultado (2026-11-10): Exness no lo ofrece. `market_book_add` falla en EUR/USD y en el Nasdaq. Descartado.
- [ ] **5.4 Datos externos**: calendario económico (también de la Fase 1.4), activos relacionados (índice del dólar, diferenciales de tipos, oro) y sentimiento.
  - Cada fuente nueva necesita su propio histórico alineado en hora del servidor.
  - Resultado parcial (2026-11-10, ❌ activos relacionados): rentabilidad a 1 h y 4 h de los otros cuatro activos de Exness como features. `t` medio por activo −2.87 / −0.08 / +0.08 / −0.69 / −0.89; mejora a la base en el S&P 500 y el DAX, sin acercarse a `t` = 2, y el ROC-AUC no cambia. El calendario económico se evaluó en 1.4 (❌ como feature). Queda por evaluar el sentimiento.

---

## 📈 Fase 6 — Seguimiento de Tendencia Multi-mercado (otra Fuente de Ventaja)

La dirección de EURUSD M15 no se puede predecir con velas (Fases 2 y 5), y en plazos cortos el spread se come los movimientos. Esta fase cambia de estrategia en vez de modelo: reglas clásicas de seguimiento de tendencia en temporalidades largas y en varios mercados. Está documentada con décadas de datos (p. ej. Moskowitz, Ooi y Pedersen, 2012) y tiene una explicación económica (infrarreacción y sobrerreacción a la información).

- ❌ **6.1 Donchian 55/20 en D1, cartera de 6 mercados** ([tendencia.py](tendencia.py))
  - **Fijado el 2026-11-02, antes de ver ningún resultado.** No se ajusta ningún parámetro: las reglas son las del "Sistema 2" de las Tortugas (años 80), así que todo el histórico desde 2010 es fuera de muestra.
  - **Mercados:** EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD y XAGUSD (los mismos de la prueba de otros mercados), con el mismo riesgo (1 R) por operación.
  - **Entrada:** cierre de la vela `t` por encima del máximo de las 55 velas anteriores (long) o por debajo del mínimo (short). Se ejecuta en la apertura de `t+1` al ask (long) o al bid (short), con el deslizamiento de `config.SLIPPAGE_PUNTOS`.
  - **Stop inicial:** 2 × ATR(20) de Wilder de la vela `t`, fijo y en el servidor. Se dispara dentro de la vela (con el bid el long, con el ask el short) y, si hay hueco, en la apertura.
  - **Salida:** cierre por debajo del mínimo de las 20 velas anteriores (long) o por encima del máximo (short), ejecutada en la apertura siguiente.
  - Una posición por mercado, sin piramidar. Tras una salida se puede entrar al lado contrario.
  - **R** = resultado / distancia al stop inicial, neto de spread, deslizamiento, comisión y swap de cada rollover (triple el miércoles).
  - **Periodos:** desarrollo de 2010 a 2023. **Holdout** del 2024-01-01 a hoy: se mira una sola vez, al final. Los trades que cruzan el corte no cuentan en ninguno de los dos.
  - **Métricas:**
    - por trade: esperanza, `t` y profit factor;
    - de cartera: resultado diario marcado a mercado y sumado por mes, con `t` mensual, Sharpe anual y drawdown máximo en R. Es la medida principal, porque los trades de mercados correlacionados no son independientes.
  - **Criterio de éxito (se tienen que cumplir todos):**
    1. **Desarrollo:** `t ≥ 2` por trade (con ≥ 30 trades) y `t ≥ 2` mensual de cartera. Con 13,6 años, esto equivale a un Sharpe ≥ 0,54.
    2. **Estabilidad:** resultado positivo en ≥ 60% de los 8 segmentos iguales del desarrollo.
    3. **Robustez:** esperanza > 0 en el desarrollo con los vecinos 20/10 y 100/50, y con un swap pesimista (2% anual del precio en contra, en los dos lados). Los swaps de MetaQuotes-Demo son mucho menores que los de un broker real, y con operaciones de semanas el swap pesa.
    4. **Holdout:** esperanza > 0. Se informa su `t`, pero no se exige `t ≥ 2`: en 2,8 años eso pediría un Sharpe > 1,2, que ningún seguimiento de tendencia tiene.
  - **Informativo (no decide):** las mismas reglas en H4, costes ×3 y el resultado por mercado.
  - **Variantes:** 1 principal y 5 de comprobación (H4, 20/10, 100/50, swap pesimista, costes ×3). Se cuentan aparte de las ~38 del modelo de M15, porque es otra hipótesis con otros datos.
  - **Limitación:** el swap histórico real no está disponible, así que se aplica el swap actual del símbolo a todo el periodo.
  - **Resultado (2026-11-02): sin ventaja.** No cumple ni el desarrollo ni la robustez:
    - **Principal:** 360 trades con −0.032 R ± 0.121 (`t` = −0.27). La cartera da `t` mensual −0.09 (Sharpe −0.03), con solo 3 de 8 segmentos positivos y un drawdown máximo de 66.7 R.
    - **Holdout:** +0.100 R (`t` = 0.27, 79 trades), gracias al oro y la plata en 2024–2026.
    - **Comprobaciones:** ninguna de las 5 llega a `t` = 1. La mejor, 100/50, da +0.206 R (`t` = 0.70) en desarrollo y −0.254 R en el holdout.
    - **Costes:** no son el problema. Con costes ×3 la esperanza solo baja de −0.032 a −0.040 R, porque en D1 el spread pesa muy poco. El swap pesimista la baja a −0.084 R.
    - **Cuándo ganó:** de 2012 a 2016 (+50 R entre los segmentos 2 y 3) y perdió de 2017 a 2023. Coincide con la mala década del seguimiento de tendencia (2011–2019) y con que Forex es el sector donde peor funciona.
    - Detalle en [Entrenamiento.md](Entrenamiento.md), sección 6.
- ❌ **6.2 Donchian 55/20 en D1, cartera diversificada** ([tendencia.py](tendencia.py) `--diversificada`)
  - **Fijado el 2026-11-02, antes de ver el histórico de los mercados nuevos y cualquier resultado.** Las reglas son las de 6.1 sin cambios.
  - **Candidatos:**
    - **Divisas:** los 7 pares principales (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD).
    - **Índices:** todos los símbolos del grupo `Indexes`.
    - **Metales:** los del grupo `Metals` cotizados en dólares (sin cruces como XAUEUR).
    - **ETFs fijados de antemano**, para sectores sin CFD en MT5: TLT e IEF (bonos del Tesoro de EE. UU.), USO (petróleo), UNG (gas natural) y DBA (agricultura).
  - **Un símbolo por subyacente:** si hay varias versiones del mismo índice (p. ej. US500 y US500M), se queda la de histórico más largo y, a igualdad, la que no lleva sufijo.
  - **Inclusión, solo por calidad de datos:**
    - histórico D1 que empieza a más tardar el 2010-02-01;
    - ningún hueco de más de 15 días entre velas consecutivas;
    - ningún cambio de más del 40% entre cierres consecutivos (indica un split no ajustado o datos erróneos).
    - Los excluidos se listan con el motivo.
  - **Enmienda (2026-11-02, tras el inventario de datos y antes de ver ningún resultado):** ningún CFD del grupo `Indexes` cumple la calidad de datos. Empiezan entre 2010-08 (CHINA50) y 2022, y los de 2011–2012 tienen huecos de 48–56 días. Para no dejar la cartera sin renta variable, el sector de índices se representa con 4 ETFs fijados ahora: SPY (EE. UU.), EFA (desarrollados fuera de EE. UU.), EWJ (Japón) y EEM (emergentes). Pasan por el mismo filtro de calidad.
  - **Sectores y pesos:** 5 sectores (divisas, índices, metales, bonos y materias primas), con el mismo riesgo total cada uno y repartido a partes iguales entre sus mercados. Así los índices, que son muchos y están muy correlacionados, no dominan la cartera. Los pesos se normalizan para que la media por mercado sea 1 R, y los R de los trades y de la cartera son ponderados.
  - **Periodos y criterio:** los mismos 4 puntos de 6.1 (desarrollo con `t ≥ 2` por trade y mensual, ≥ 60% de segmentos positivos, robustez con 20/10, 100/50 y swap pesimista, y holdout positivo).
  - **Informativo (no decide):** el mismo peso por mercado y costes ×3. H4 no se evalúa: era informativo en 6.1 y muchos ETFs no tienen H4 desde 2010.
  - **Variantes:** 1 principal y 5 de comprobación. Con 6.1 suman 12 en la Fase 6.
  - **Resultado (2026-11-02): sin ventaja.** No cumple ningún punto del criterio.
    - **Cartera que dejó el filtro:** 13 mercados en 4 sectores.
      - Divisas: los 7 pares.
      - Índices: EFA y EEM. SPY empieza en 2024, y EWJ tiene un contrasplit 1:4 sin ajustar (+303%).
      - Metales: oro y plata. Los industriales empiezan en 2023, y el platino y el paladio en 2015.
      - Bonos: TLT e IEF.
      - Materias primas, ninguno: USO tiene el contrasplit 1:8 de 2020 sin ajustar (+731%), UNG empieza en 2025 y DBA en 2026.
    - **Principal:** 789 trades con −0.005 R ± 0.080 (`t` = −0.06). La cartera da `t` mensual 0.11 (Sharpe 0.03), con 3 de 8 segmentos positivos y un drawdown máximo de 67.3 R.
    - **Comprobaciones:** ninguna llega a `t` = 0.1. Con 20/10, −0.021 R; con 100/50, +0.006 R; con el swap pesimista, −0.096 R.
    - **Holdout:** −1.121 R en bruto. Ese número está distorsionado por 8 velas de IEF con spreads erróneos del servidor (53–59 dólares, en noviembre y diciembre de 2024). Con ese spread corregido queda en −0.015 R (`t` = −0.05), y sigue sin pasar.
    - **Por mercado:** los bonos son lo mejor del desarrollo (TLT +0.25 R con `t` = 0.95; IEF +0.19 R). USDCHF es lo peor (−0.37 R, `t` = −2.48).
    - Detalle en [Entrenamiento.md](Entrenamiento.md), sección 6.

---

## 📒 Registro de Cambios del Plan

| Fecha | Cambio |
| :--- | :--- |
| 2026-10-31 | Plan creado a partir de la lista de propuestas; marcado lo ya hecho y lo ya descartado. |
| 2026-10-31 | Evaluadas 2.3 (`ADX(14)`, `+DI`, `-DI`), 2.4 (`Multi-timeframe real` H1/H4 cerrado) y su combinación en el walk-forward desde 2010: marcadas ❌ como features del modelo; `adx_wilder` y `mtf_cerrado` quedan en `features.py` con pruebas anti-fuga (111 tests). |
| 2026-11-02 | Añadida la Fase 6 (seguimiento de tendencia por reglas), con reglas y criterio fijados antes de ver resultados. Evaluada 6.1 (Donchian 55/20 en D1, 6 mercados, y 5 comprobaciones): ❌ sin ventaja. `tendencia.py` con 23 pruebas nuevas (134 tests). |
| 2026-11-10 | Banco de experimentos ([experimentos.py](experimentos.py), registro en `experimentos/registro.csv`): evalúa cada variante en los 5 activos de Exness con el walk-forward de `train_model.py`, 3 semillas y holdout desde 2026-01-01 sin tocar. Evaluadas 2.1, 2.2, 3.3a, 3.3b, 4.3 (12 meses) y 5.4 (activos relacionados): ❌ todas. La base da `t` medio −0.08 / +0.60 / −0.48 / −0.47 / −1.91, con mucha dispersión entre semillas (EUR/USD: +2.23 / −1.34 / −1.12). |
| 2026-11-10 | Calendario económico (1.4): servicio MQL5, filtro en el bot y dos variantes en el banco, ❌ sin ventaja. Gestión de la posición (1.6: breakeven, trailing y parcial): medida en el banco, ❌ ninguna mejora; implementada en el bot y desactivada por defecto. |
| 2026-11-10 | Reglas de cartera (1.7): `cartera.py` y opciones del bot; la reducción en drawdown baja el drawdown máximo un 42 %, las otras dos apenas cambian nada. |
| 2026-11-10 | Informe de rendimiento ([informe.py](informe.py)): estadísticas en R y desgloses por activo, lado, salida y sesión, con la esperanza del backtest al lado; semanal por Telegram desde el vigilante y bajo demanda con `/informe`. |
| 2026-11-10 | Fase 5: inventario de ticks de Exness (5.1, hecho: solo desde 2026-01-01 y solo bid/ask), features de ticks bloqueadas por falta de histórico (5.2) y libro de órdenes descartado (5.3). |
| 2026-11-02 | Añadida y evaluada 6.2 (cartera diversificada por sectores; enmienda de los índices con ETFs por falta de datos): ❌ sin ventaja. `tendencia.py --diversificada`, con filtro de calidad, pesos por sector y aviso de spreads erróneos (138 tests). |
