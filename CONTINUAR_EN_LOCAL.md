# Prompt para continuar el proyecto en el entorno local

> Cómo usarlo: clonar el repositorio (sección 0), abrir Claude Code **en la raíz del repo** y pegar todo lo que está debajo de la línea `=== PROMPT ===` como primer mensaje.

```bash
git clone https://github.com/Alejob12/proyecto-1-machine-learning.git
cd proyecto-1-machine-learning
git checkout claude/exciting-mccarthy-ykf118      # NO está fusionada en main
python3.13 -m venv .venv && source .venv/bin/activate   # usar 3.12 o 3.13 si 3.14 no trae ruedas de scikit-learn 1.9.1
pip install -r requirements.txt
```

=== PROMPT ===

# Contexto: continuar la Parte 1 de la competencia Kaggle de Aprendizaje de Máquina (Uniandes 2026-20)

Eres mi asistente de programación. Vienes de una sesión larga en la nube que ya no existe; todo lo que necesitas está en este mensaje y en el repositorio. Trabajamos en la rama `claude/exciting-mccarthy-ykf118` (no hay pull request; no abras uno si no te lo pido). Responde siempre en español.

## 1. Qué es el proyecto
Competencia Kaggle de curso: clasificar el **sentimiento global** de reseñas de productos en español en `negativo` / `neutral` / `positivo`. Métrica: **accuracy**. `data/train.csv` (12.000 reseñas con `id,text,label`; negativo 4.212, positivo 4.212, neutral 3.576), `data/eval.csv` (3.000 sin etiqueta = test de Kaggle), `data/sample_submission.csv` (`id,answer`). El enunciado está en `proyecto202620.pdf`; las instrucciones del repo, en `CLAUDE.md` (léelo primero).

Las reseñas tienen ~54 palabras y 4 oraciones: relleno logístico (envío, dónde se guarda) + opiniones + conectores ("aun así", "eso sí", "por otro lado", "al final") + a veces un cierre indeciso ("tengo sentimientos encontrados", "cada quien que juzgue", "ni fu ni fa"). La etiqueta es el sentimiento **global**: depende del orden (gana casi siempre la última opinión tras un conector contrastivo), de la negación y de los conectores, no de contar palabras polares.

## 2. REGLAS DURAS de la Parte 1 (incumplirlas = nota 0 en la parte; no negociables)
- Solo clasificadores de **scikit-learn**. Prohibido: redes neuronales (tampoco `MLPClassifier`), transformers, embeddings preentrenados (word2vec, fastText, BERT), xgboost/lightgbm, datos o léxicos externos.
- Representación del texto: bolsa de palabras, TF-IDF, n-gramas (palabras o caracteres) y *features hechas a mano derivadas del texto* (regex, listas **estructurales** de conectores/negadores/frases de cierre, segmentación en oraciones/cláusulas, posición, conteos, marcado de negación) y léxicos **aprendidos de los datos de entrenamiento dentro de `fit`**. **No** se permiten listas escritas a mano de palabras o plantillas positivas/negativas (léxicos de sentimiento) dentro del modelo; solo se pueden usar para analizar.
- **Nunca** usar `eval.csv` para entrenar ni ajustar nada (solo para predecir al final). Todo lo aprendido debe ocurrir dentro de `fit` de un `Pipeline`/estimador (sin fuga entre pliegues).
- Entregables en Bloque Neón (semana 11): **un** notebook + **un** modelo `.joblib` correspondientes al envío con **mejor score público** del grupo en Kaggle. Debe ser replicable solo con el notebook y el `.joblib`: funciones y clases definidas con `def`/`class` dentro del notebook (sin `lambda`, sin `.py` externo). Semillas fijas.
- Mínimo 5 envíos distintos a Kaggle (ya hay 13).
- `parte1.ipynb` es el esqueleto del usuario: **no lo modifiques**. Todo el trabajo va en archivos nuevos.
- Pendiente de confirmar con la profesora o el monitor: partir el texto (última oración, tras el último conector, segmentos) antes de aplicar TF-IDF. Recuérdamelo si vamos a entregar uno de esos modelos. NB-SVM (ponderar la bolsa de palabras con el log-count ratio de Naive Bayes, Wang y Manning 2012) también hay que justificarlo en la sustentación: es una ponderación supervisada de la bolsa de n-gramas, aprendida solo en `fit`.

## 3. Objetivo y expectativas realistas
Objetivo del usuario: **quedar mínimo en el top 5** del leaderboard público. Captura de hoy: 1.º Grupo30 0,92111 · 2.º 0,92000 · 3.º Grupo 46 0,91777 · 4.º y 5.º 0,91555 · 6.º 0,91444 · 7.º 0,91333. **Nuestro mejor envío público: 0,90666** (no sé a qué archivo de `kaggle_iteraciones/` corresponde; pregúntame y anótalo).

Aritmética que hay que tener presente: los scores avanzan de 1/900 ≈ 0,00111, así que el público son ~900 reseñas (0,90666 = 816 aciertos; 0,91555 = 824). **Faltan 8 aciertos.** El error estándar del score público a ~0,91 es ≈ ±0,0095: dos modelos casi iguales difieren fácilmente en 5-10 reseñas del público solo por azar. Además, la nota de la rúbrica es por **percentil del score privado** (≥ percentil 75 = 100 % del rubro), no por el puesto público. Por eso: prioriza mejoras **reales** (validación cruzada repetida), no perseguir el público, y avísame con honestidad cuando una "mejora" sea solo ruido. El techo con estos datos es ≈ 0,91-0,93 (ver sección 6).

## 4. Estado actual: 13 iteraciones, todas con notebook, envío y modelo
Validación cruzada = `StratifiedKFold(5, shuffle=True, random_state=42)` sobre `X_ent` (80 % de `train.csv`, `train_test_split(test_size=0.2, stratify=label, random_state=42)`). "Prueba" = el 20 % restante (`X_prueba`), **ya mirado 3 veces** (iter10, 12 y 13): no lo uses más para elegir; usa CV con semillas nuevas.

| # | Qué es | CV | Prueba |
|---|---|---|---|
| 01 | Bolsa de palabras + Naive Bayes | 0,7047 | |
| 02 | TF-IDF + regresión logística | 0,7252 | |
| 03 | TF-IDF + Random Forest | 0,6760 | |
| 04 | Logística ajustada (GridSearchCV) | 0,7407 | |
| 05 | Stacking clásico (LR+SVM+NB) | 0,7420 | |
| 06 | Solo la última oración | 0,8509 | |
| 07 | Solo lo que va tras el último conector | 0,8316 | |
| 08 | Texto completo + tras conector + última oración (FeatureUnion + LR) | 0,8869 | |
| 09 | Stacking de un modelo por vista (meta logístico) | 0,8942 | |
| 10 | Stacking de 12 bases + metamodelo `HistGradientBoosting` | 0,9012 | 0,9083 |
| 11 | Vistas de palabras con NB-SVM + negación marcada + última cláusula | 0,9076 | |
| 12 | NB-SVM también en los n-gramas de caracteres (13 bases) | 0,9093 | 0,9075 |
| **13** | 12 + `tras_conector_sr` + `polaridad_segmentos` (15 bases) | **0,9123 ± 0,0071** | **0,9142** |

Los 13 envíos son distintos entre sí (≥ 131 de 3.000 predicciones de diferencia). La iteración 11 no se evaluó en la prueba.

**Anota los scores públicos reales de cada archivo** en `kaggle_iteraciones/README.md` (columna "Score público Kaggle", hoy vacía); pídeme los que falten.

## 5. Mapa de archivos (rutas relativas a la raíz del repo)
- `CLAUDE.md`, `PLAN.md`, `PLAN_PARTE1.md`, `proyecto202620.pdf`: contexto y enunciado.
- `data/`: `train.csv`, `eval.csv`, `sample_submission.csv`.
- `parte1.ipynb`: esqueleto del usuario (NO tocar). `parte1convencional.ipynb` + `models/parte1_convencional.joblib` + `submissions/conv01..05_*.csv`: solución "convencional" (CV 0,7407, prueba 0,7392).
- `notebooks_iteraciones/iterNN_*.ipynb`: un notebook **autocontenido** por iteración (limpieza, vistas, modelo, CV, envío, `.joblib`, prueba de recarga). Se ejecutan desde la raíz o desde su carpeta. **`iter13_sin_relleno_polaridad.ipynb` es la referencia: contiene todo el código del mejor modelo.**
- `kaggle_iteraciones/iterNN_*.csv`: los envíos listos (`id,answer`). `kaggle_iteraciones/README.md`: tabla de iteraciones, historia y advertencias.
- `models/iteraciones/iterNN_*.joblib`: modelo entrenado con las 12.000 reseñas de cada iteración (recargar exige ejecutar antes las celdas de definiciones del notebook correspondiente).
- `experimentos/` (nuevo; es el laboratorio de la sesión anterior, ya con rutas relativas):
  - `harness.py`: `from harness import *` carga datos y `X_ent/y_ent/X_prueba/y_prueba`, `CV`, `corregir_texto`, `oraciones`, `CONECTORES`, `CIERRE`, todas las vistas y modelos base (iter08 e iter10), `INDECISA_ENT` (reseñas con cierre indeciso) y `evaluar(modelo, nombre, n_jobs)` (CV estándar; devuelve acc, std, pliegues, acierto en indecisas/no indecisas/neutral y predicciones fuera de pliegue `oof`).
  - `modelo_final2.py` = iteración 13 (autocontenido, `construir_modelo()`); `modelo_final_r1.py` = iteración 12 (13 bases).
  - `cachef.py`: caché exacta de predicciones de los modelos base por pliegue externo e interno (`cachear`, `cargar`, `evaluar_meta`, `pareado`) para probar nuevos metamodelos o subconjuntos de bases en segundos. Equivale exactamente a evaluar un `StackingClassifier`. El directorio `experimentos/cache/` NO está en git (~300 MB); se regenera.
  - `datos/analisis_final.pkl`: las 9.600 reseñas de `X_ent` con categorías de error (`cat` ∈ dura, indec_otro, mixta_contr, una_pol, sin_op, otro), `pred`, `indec`, `tc` (texto corregido). `datos/oof_iter13_cv42.csv`, `datos/oof_iter10_cv2026.csv`: predicciones fuera de pliegue.

Prueba rápida de que todo funciona (fija 1 hilo ANTES de importar numpy; ver sección 8):
```python
import os
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]: os.environ[v]="1"
import sys; sys.path.insert(0, "experimentos")
from harness import *
import modelo_final2 as m
r = evaluar(m.construir_modelo(), "iter13", n_jobs=4)   # ~10 min; en la nube dio 0.9123 ± 0.0071
```
**Primero reproduce esa línea base en esta máquina** (BLAS distinto, p. ej. Accelerate en Mac, puede cambiar decimales) y compara todo candidato contra *esa* línea base local, no contra 0,9123.

## 6. Cómo está construida la iteración 13 (el modelo a superar)
`StackingClassifier` con 15 modelos base y metamodelo `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, early_stopping=False, random_state=42)`; CV interno `StratifiedKFold(5, shuffle=True, random_state=42)`. Cada modelo base mira una parte distinta de la reseña (todas pasan por `corregir_texto`: reparar codificación dañada, minúsculas, quitar tildes/ñ):
- **13 bases NB-SVM** (negación marcada `no_X` hasta la puntuación → uni+bigramas **binarios** → escalado por log-count ratio de NB por clase (una contra el resto) → `LogisticRegression(C=1)`): `nb_tres` (completo + tras último conector + última oración unidas), `completo`, `tras_conector`, `ultima`, `primera`, `penultima`, `con_conector` (cada palabra pegada al conector que abre su oración), `ultima_opinion` (última oración con palabra de opinión aprendida en `fit`), `antes_del_conector`, `ultima_clausula`; más `cierre` (indicador de frase de indecisión, TF-IDF+LR) y dos de **n-gramas de caracteres** `char_wb` 2-5 binarios+NB (`caracteres_tras_conector`, `caracteres_ultima`).
- `tras_conector_sr`: quita las oraciones finales sin palabras de opinión y toma desde el último conector (palabras de opinión aprendidas en `fit`: razón de frecuencia polar/neutral ≥ 4) → TF-IDF → LR(C=3).
- `polaridad_segmentos`: segmenta (oraciones cortadas en cada conector), aprende en `fit` un clasificador de segmentos negativo/nada/positivo con **supervisión débil** (segmentos abiertos por conector contrastivo heredan la etiqueta de la reseña; los de reseñas neutrales = "nada"; 2 rondas de autoentrenamiento; cross-fitting interno), marca cada palabra (`P_`, `N_`, `Nult_`, `Ppri_`, `C_`, `O_`) → TF-IDF → LR(C=3).

Qué mueve la accuracy (para no repetir exploraciones): **darle al modelo el final de la reseña por separado** es lo que importa (iter05→06: +10,9 pts; vistas unidas: +5,5); stacking+boosting +1,4; NB-SVM con negación +0,8; los dos modelos de la 13 +0,3 (no significativo por separado: p = 0,18/0,41/0,78 en semillas 42/2026/7; en la semilla 99, que no se usó para elegir, gana a la iteración 12 en 5/5 pliegues, 0,9134 vs 0,9088, p = 0,036).

## 7. Lo que ya se sabe del techo y lo que NO funciona
- **~14 % de las reseñas (≈1.380-1.420 de 9.600) tienen una etiqueta prácticamente aleatoria**: mixtas (una opinión positiva y una negativa) con cierre indeciso, o unidas por conectores aditivos ("por otro lado", "por cierto", "de paso", "además"). Modelos entrenados solo con ellas (BoW, NB-SVM, caracteres, kNN) dan 0,48-0,51; ninguna variable (aspecto, plantilla, longitud, orden, conector, tipo de cierre) pasa de ~55 %; chi² todos no significativos. Techo teórico ≈ 0,926-0,933; realista ≈ 0,91-0,925.
- Errores del modelo 12 por categoría (871 en CV): `dura` 446/947, `indec_otro` 190/432, `mixta_contr` 106/3.320, `una_pol` 107/2.258, `sin_op` 18/2.463. Es decir, ~73 % de lo que falla está en el pozo aleatorio; solo ~50-60 errores son "resolubles" (una opinión escrita de forma rara: "quedó delicada", "se mantuvo fiable", errores de tipeo "danno", "fliz").
- No pasaron el protocolo (ruido o peor): bases de polaridad por oración con features de secuencia + HGB (0,66-0,87 solas); `posicion_conector`; `ultima_opinion` corregida (palabras de opinión por razón polar/neutral; mejora una_pol pero empeora las mixtas); `OpinionesPorRango`; NB-SVM de caracteres sobre el texto completo; "experto en indecisas" (base entrenada solo con reseñas de cierre indeciso: indistinguible de añadir una base de ruido); variantes del metamodelo (HGB con otros hiperparámetros, ExtraTrees, RandomForest, votación de 3, HGB+ET): dentro del ruido; metamodelo logístico: peor (0,89-0,90); quitar de entrenamiento las reseñas mixtas con plantillas escritas a mano (viola la regla del léxico de sentimiento).
- Fuente principal de ruido: **la semilla del CV interno del stacking** (pasar de 42 a 1 cambia ~+107 errores sumando 3 CV externos; ±0,3 pts). Por eso el 0,9123 de la 13 puede estar algo optimista (se eligió con la semilla interna 42). Esperable en datos nuevos: ≈ 0,905-0,915.

## 8. Trampas técnicas (ya pisadas)
1. **Hilos BLAS**: `polaridad_segmentos` usa `liblinear` + autoentrenamiento con umbrales; con distinto número de hilos de OpenBLAS cambian ~3 % de las predicciones. Fija `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1` **antes** de importar numpy/sklearn (la primera celda de `iter13` ya lo hace). Genera el envío y el `.joblib` en la misma ejecución.
2. Dentro de `StackingClassifier`, los modelos base reciben `y` **codificada como 0/1/2** (orden alfabético negativo/neutral/positivo). Cualquier transformer que compare `y == "neutral"` debe decodificar (`etiquetas_texto`). `UltimaOpinion` hereda esa peculiaridad (es el comportamiento medido; no lo cambies sin remedir).
3. `HistGradientBoostingClassifier(early_stopping="auto")` se activa con >10.000 filas: al entrenar con las 12.000 de `train.csv` el modelo sería distinto del validado. Siempre `early_stopping=False`.
4. Si el código está definido en `__main__` (notebook) y entrenas el stack con `n_jobs>1`, `joblib.dump` falla (`PicklingError: not the same object`). Entrena el modelo final con `n_jobs=None`; usa `n_jobs>1` solo en `cross_validate`.
5. Recargar un `.joblib` exige haber ejecutado antes las celdas de definiciones (el pickle referencia nombres de `__main__`).
6. Tiempos (4 CPU): CV de un stack de 13-15 bases ≈ 6-10 min con `n_jobs=4`; ajuste final con 12.000 reseñas ≈ 10-13 min en 1 hilo; ejecutar el notebook 13 completo ≈ 24 min sin competencia por CPU.
7. `cachef.py` busca primero `experimentos/cache_base/` (opcional); si falta, recalcula bases (lento la primera vez). Nunca sobrescribas cachés ajenas.

## 9. Protocolo de decisión (obligatorio para aceptar cualquier cambio)
El ruido es del tamaño de las mejoras. Un cambio solo cuenta si, **comparado con la línea base local en los mismos pliegues**: (a) reduce el total de errores sumando CV externos de 3 semillas (42, 2026, 7: `StratifiedKFold(5, shuffle=True, random_state=s)` sobre `X_ent`) y mejora en ≥ 2 de 3; y (b) se confirma en una semilla que no se usó para elegir (p. ej. 99), idealmente con McNemar pareado. Si compartes el stack, compara también contra la media de varias semillas internas, no solo contra la 42. Reporta siempre errores exclusivos de cada modelo. No uses `X_prueba` para elegir.

## 10. Plan sugerido (ordenado por probabilidad de ayudar de verdad; sé crítico y mide antes de gastar horas)
1. **Reducir varianza del stack (la más prometedora para el score privado)**: `VotingClassifier(voting="soft")` o promedio manual de K stacks de la 13 que solo cambien la semilla del CV interno del `StackingClassifier` (y `random_state` del HGB); probar también CV interno de 10 pliegues. Evidencia previa: promediar 2 stacks bajó ~40 errores respecto a la media de los dos individuales (pero no frente al stack con semilla 42, que pudo ser afortunado). Compara contra la **media** de los stacks individuales.
2. **Afinar las bases NB-SVM** (alpha del suavizado, `C` de cada LR, `min_df`, binario vs conteo) con el protocolo de 3 semillas usando la caché; y más diversidad solo si es barata (p. ej. `LinearSVC` o `ComplementNB` por vista).
3. **Errores resolubles** (≈ 50-60 por CV): leer los errores fuera del pozo (`experimentos/datos/analisis_final.pkl`, `cat` en una_pol / mixta_contr / sin_op) y buscar patrones de una sola opinión escrita de forma rara o con errores de tipeo; cualquier idea debe usar solo léxicos aprendidos en `fit`.
4. Elegir el envío final: entre candidatos con CV equivalente, **prioriza CV promedio sobre el score público** (±1 pt de ruido); pero recuerda que la regla del curso exige entregar el notebook y el `.joblib` del envío con **mejor score público**.
No vale la pena: buscar reglas para el pozo aleatorio (ya demostrado sin señal), cambiar de metamodelo, ni subir envíos "a la lotería" para escalar el público (el rubro se calcula con el privado).

## 11. Convenciones para trabajar
- Una iteración nueva = notebook autocontenido `notebooks_iteraciones/iter14_<nombre>.ipynb` (usa `iter13_*.ipynb` como plantilla: configuración con hilos=1, datos y corrección, definiciones, modelo, CV, envío a `kaggle_iteraciones/iter14_<nombre>.csv`, modelo a `models/iteraciones/iter14_<nombre>.joblib`, recarga y comprobación), con markdown que explique qué cambia, por qué y el resultado con números reales; y una fila nueva en `kaggle_iteraciones/README.md`. Ejecuta el notebook completo (`jupyter nbconvert --to notebook --execute <nb> --inplace`) para que quede con salidas antes de hacer commit.
- Cada notebook se valida con el formato de envío: columnas `id,answer`, 3.000 filas, ids de `sample_submission.csv`, etiquetas exactas `negativo`/`neutral`/`positivo`.
- Los notebooks se califican por documentación (15 %) y calidad del proceso (15 %): explica decisiones y resultados negativos en markdown.
- Commits en la rama `claude/exciting-mccarthy-ykf118`, mensajes en español y descriptivos; sin pull request salvo que lo pida. No subas cachés ni archivos temporales.
- Sé honesto con los números: si algo no mejora, dilo y no lo vendas como mejora.

## 12. Empieza así
1. Lee `CLAUDE.md`, `kaggle_iteraciones/README.md` y `experimentos/modelo_final2.py`; confirma que el entorno funciona (sección 5) y reproduce la línea base local de la iteración 13.
2. Pregúntame qué archivo de `kaggle_iteraciones/` sacó 0,90666 y cuáles son los scores públicos de los demás; actualiza la tabla.
3. Propón un plan corto (qué probar, tiempo estimado y cómo lo validarás con el protocolo de la sección 9) y espera mi visto bueno antes de lanzar experimentos largos.
