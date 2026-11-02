# Plan de Mejoras por Fases 🗺️

Seguimiento de las mejoras propuestas para el bot: qué está hecho, qué se probó y descartó, y qué falta. Marca cada casilla al terminarla (`- [x]`) y anota el resultado en la línea de debajo.

**Leyenda:** `[x]` hecho · `[ ]` pendiente · ❌ probado y descartado (no mejoró) · 🔒 bloqueado por otra fase

---

## 📏 Protocolo para Evaluar Cualquier Mejora del Modelo

Una mejora del modelo (feature, régimen, filtro, re-entrenamiento) solo cuenta como **hecha** si pasa este protocolo. Si no lo pasa, se marca ❌ con su resultado: un "no mejora" bien medido también es un resultado.

1. **Fijar la variante antes de mirar resultados:** qué se añade, con qué parámetros y qué se espera. No se ajustan parámetros viendo el test.
2. **Walk-forward desde 2010** (`train_model.py`: 5 ventanas, purga por el final de cada etiqueta, 3 semillas), comparando contra la línea base, **con los mismos datos**.
3. **Criterio:** que suba el ROC-AUC de forma estable entre ventanas **y** que la esperanza agregada mejore con las 3 semillas. Solo se considera ventaja real con `t >= 2` en todas las semillas, y además en el holdout (≥ 30 trades).
4. **Contar las pruebas.** Cada variante evaluada sobre los mismos datos aumenta la probabilidad de un falso positivo. Llevamos **unas 38 variantes**. Con tantas, un `t` cercano a 2 aislado no significa nada (ver [Entrenamiento.md](Entrenamiento.md)).
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
  - Resultado:
- [x] **1.2 Tamaño de posición por volatilidad** (`RIESGO_PCT`, en [trading_bot.py](trading_bot.py)): `lotes = equity × RIESGO_PCT / (distancia al SL / tick_size × tick_value)`, con tope `MAX_LOTES`.
- [ ] **1.3 Activar el lote por riesgo en Demo**
  - Hoy está en `None` (0.01 fijo).
  - Hecho cuando: se elige un `RIESGO_PCT` (por ejemplo 0.25%), se activa en la Demo y se comprueba en `operaciones.csv` que el riesgo en dinero es constante entre operaciones.
  - Resultado:
- [ ] **1.4 Filtro de calendario económico**: sin nuevas entradas de 15 minutos antes a 15 minutos después de noticias de alto impacto (NFP, FOMC, IPC, BCE).
  - La API de Python de MT5 **no** da acceso al calendario: está solo en MQL5 (`CalendarValueHistory`). Opciones: un pequeño Expert Advisor en MQL5 que exporte los eventos a un CSV que lea el bot, o una fuente externa.
  - Hecho cuando: el bot lee los eventos, los registra en el log, bloquea entradas en la ventana y tiene pruebas automáticas. Para medir su efecto en el backtest hace falta el histórico de eventos; si no lo hay, se valida solo en Demo.
  - Resultado:
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

- [ ] **2.1 Banderas de sesión**: Asia, Londres, Nueva York y el solape Londres–NY, en hora del servidor (GMT+2/+3), con atención a los cambios de horario de verano de Europa y EE. UU.
  - Parte de esta información ya la capturan `hora_sin` y `hora_cos`. Hay que comprobar si aportan algo más.
  - Hecho cuando: pasa el protocolo, o se marca ❌.
  - Resultado:
- [ ] **2.2 Squeeze**: anchura de las Bandas de Bollinger(20, 2) entre la anchura del Canal de Keltner(20, 1.5×ATR), y una bandera de squeeze activo.
  - Hecho cuando: pasa el protocolo, o se marca ❌.
  - Resultado:

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
  - [ ] a) como feature más del modelo;
  - [ ] b) como filtro: sin entradas en rango;
  - [ ] c) con SL/TP distintos por régimen. Multiplica las variantes probadas, así que hay que contarlas en el protocolo.
  - Resultado:

---

## 🔁 Fase 4 — Modelo y Re-entrenamiento

- [x] **4.1 Probabilidades con umbral de confianza** (`predict_proba`, calibración y umbral elegido en validación).
  - Nota: un umbral fijo de 65–70% **no sirve aquí**. Solo un ~28% de las operaciones llega al TP, y para no perder basta acertar el 33.3%, así que el modelo casi nunca daría 0.65. El umbral tiene que salir de la validación, como ya se hace.
- [x] **4.2 Walk-forward anclado** (train creciente) para evaluar.
- [ ] **4.3 Ventana deslizante frente a anclada**: entrenar solo con los últimos X meses (por ejemplo 12) y comparar con la anclada usando el protocolo.
  - 3 meses de M15 son unas 6.000 filas, pocas para un Random Forest con esta tasa de aciertos. Conviene probar 6, 12 y 24 meses, fijados de antemano.
  - Resultado:
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

- [ ] **5.1 Inventario de ticks**: medir cuánto histórico de ticks da MetaQuotes-Demo (`copy_ticks_range`), con qué campos (bid, ask, flags, volumen) y con qué calidad.
  - Hecho cuando: hay una tabla de profundidad y calidad por año, como la de velas.
  - Resultado:
- [ ] **5.2 Features de ticks** (si 5.1 da suficiente histórico): desequilibrio de ticks al alza/baja, velocidad de ticks, dinámica del spread dentro de la vela y saltos de precio.
  - Hecho cuando: pasa el protocolo, o se marca ❌.
  - Resultado:
- [ ] **5.3 Libro de órdenes** (`market_book_get`): en Forex retail normalmente no está disponible. Comprobar si el broker lo ofrece; si no, se descarta.
  - Resultado:
- [ ] **5.4 Datos externos**: calendario económico (también de la Fase 1.4), activos relacionados (índice del dólar, diferenciales de tipos, oro) y sentimiento.
  - Cada fuente nueva necesita su propio histórico alineado en hora del servidor.
  - Resultado:

---

## 📒 Registro de Cambios del Plan

| Fecha | Cambio |
| :--- | :--- |
| 2026-10-31 | Plan creado a partir de la lista de propuestas; marcado lo ya hecho y lo ya descartado. |
| 2026-10-31 | Evaluadas 2.3 (`ADX(14)`, `+DI`, `-DI`), 2.4 (`Multi-timeframe real` H1/H4 cerrado) y su combinación en el walk-forward desde 2010: marcadas ❌ como features del modelo; `adx_wilder` y `mtf_cerrado` quedan en `features.py` con pruebas anti-fuga (111 tests). |
