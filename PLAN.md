# Plan de trabajo: clasificación de sentimiento de reseñas

Pasos para ir del entendimiento del problema a la entrega final de las dos partes de la competencia. Marca cada casilla al completarla. Las reglas que dan 0 están resumidas en [CLAUDE.md](CLAUDE.md); el enunciado completo es [proyecto202620.pdf](proyecto202620.pdf).

---

## 0. Lo que ya sabemos de los datos

Hallazgos de una primera revisión de `train.csv` (12.000 reseñas, ninguna repetida y ninguna compartida con `eval.csv`):

- **Clases casi balanceadas:** positivo 4212, negativo 4212, neutral 3576. Un clasificador que siempre diga "positivo" saca ~35 %.
- **Las reseñas parecen generadas con plantillas.** Mezclan frases de relleno (envío, dónde lo usan, color, garantía) con una o dos frases de opinión sobre un aspecto ("el ensamblaje", "el sistema de carga", "el manual").
- **Gana la última opinión.** Cuando hay una opinión buena y una mala, la etiqueta la decide la que va después del conector contrastivo:
  - *"El ensamblaje me salió resistente… **Aun así**, la instalación quedó mediocre"* → `negativo`
  - *"El peso me decepcionó… **Eso sí**, el sistema de carga volvería a comprarlo"* → `positivo`
- **Los conectores no sirven por sí solos.** "aun así", "eso sí", "al final" y "con todo" aparecen casi igual en positivas y negativas (p. ej. "eso sí": 603 neg / 515 pos). Lo que importa es **qué viene después** del conector. Esto explica por qué una bolsa de palabras simple se queda corta.
- **Neutral = sin opiniones.** Las neutrales casi nunca tienen conectores contrastivos ("al final": solo 24 neutrales frente a ~580 de cada polaridad). Son solo frases de logística y descripción.
- **Ruido ortográfico:** ~1/3 de las reseñas no tienen tildes ("compre", "llego") y hay errores de tipeo ("installacion", "abi"). Hay que normalizar tildes y considerar n-gramas de caracteres.
- **"no" es poco informativo por sí solo:** aparece en todas las clases, a menudo en frases de relleno ("no faltó ningún tornillo").

> Hipótesis central del proyecto: **el sentimiento lo determina la polaridad de la última cláusula de opinión.** Casi todo lo que sigue sirve para que el modelo capture eso.

---

## 1. Preparación (una sola vez)

- [ ] Crear el entorno virtual en la raíz del repositorio:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install pandas numpy scikit-learn joblib matplotlib seaborn jupyter
  pip freeze > requirements.txt
  ```
  Si scikit-learn no instala en Python 3.14, crea el entorno con Python 3.12 (`brew install python@3.12`, luego `python3.12 -m venv .venv`).
- [ ] Registrar a todos los integrantes en Kaggle y unirse a la competencia de la Parte 1 (el enlace se publica en Bloque Neón). Formar el equipo en Kaggle.
- [ ] Anotar el puntaje de la **línea base** del curso que aparece en el leaderboard. Hay que superarla (10 % de la nota).
- [ ] Revisar en Bloque Neón las fechas exactas de las semanas 9, 11, 14 y 16.
- [ ] Invitar a los compañeros al repositorio de GitHub y acordar quién hace qué. El aporte individual vale 15 % y se nota en el historial de commits.
- [ ] Crear `envios.md` con una bitácora de envíos a Kaggle (la Parte 1 exige al menos **5 envíos distintos**):

  | # | Fecha | Modelo / cambio | Accuracy CV | Score público Kaggle |
  |---|-------|-----------------|-------------|----------------------|

---

## 2. Parte 1: ML clásico (solo scikit-learn, sin redes neuronales ni embeddings preentrenados)

Todo en el notebook [parte1.ipynb](parte1.ipynb), en la raíz del proyecto, con celdas markdown que expliquen cada decisión. La documentación vale 15 % y la calidad del proceso otro 15 %.

### 2.1 Análisis exploratorio
- [ ] Cargar los datos y mostrar la distribución de clases y de longitudes (palabras y oraciones) por clase.
- [ ] Leer ~20 ejemplos por clase y documentar la estructura de las plantillas (relleno + opinión + conector + opinión final).
- [ ] Contar conectores por clase (la tabla de la sección 0) y mostrar que no separan positivo de negativo.
- [ ] **Validar la hipótesis de la última cláusula:** entrenar un modelo solo con la última oración y compararlo con uno que use solo la primera. Si la última gana por mucho, queda justificado el diseño de features de la sección 2.4.

### 2.2 Esquema de validación (antes de modelar)
- [ ] `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre `train.csv`. Esa es la métrica para decidir. El leaderboard público es solo una muestra y el puntaje privado define la nota.
- [ ] Meter todo (preprocesamiento + vectorizador + modelo) dentro de un `Pipeline` para que el vectorizador se ajuste solo con los folds de entrenamiento.
- [ ] Fijar `random_state=42` en todo lo que tenga azar. Los resultados deben ser replicables.

### 2.3 Línea base propia → envío #1
- [ ] `TfidfVectorizer(ngram_range=(1,2), strip_accents="unicode", sublinear_tf=True)` con `LogisticRegression`, `LinearSVC`, `MultinomialNB` y `ComplementNB`. Comparar con CV.
- [ ] **No quitar stopwords.** "no", "pero", "aun", "sí" y "al final" son justo las palabras clave.
- [ ] Generar `submissions/sub01_tfidf_lr.csv` con columnas `id,answer`, enviarlo y anotarlo en la bitácora.

### 2.4 Features que capturan el orden (la mejora principal)
Siguen siendo bolsa de palabras/TF-IDF/n-gramas, es decir, técnicas permitidas, pero aplicadas a **partes** del texto. *Confirmar con el profesor o monitor que segmentar el texto antes de vectorizar está permitido.*

- [ ] **Segmentación:** escribir una función que parta la reseña en oraciones (`.`, `!`, `?`) y además corte en los conectores contrastivos ("aun así", "eso sí", "pero", "aunque", "sin embargo", "al final", "a fin de cuentas", "con todo", "igual", "con el tiempo", "en cambio").
- [ ] **Vistas del texto:** vectorizar por separado y unir con `ColumnTransformer` o `FeatureUnion`:
  1. el texto completo,
  2. **la última cláusula después del último conector contrastivo** (la más importante),
  3. la última oración,
  4. la primera mitad frente a la segunda mitad del texto.
- [ ] **Marcado de negación:** anteponer `NEG_` a los tokens que siguen a "no" / "nunca" / "tampoco" hasta el siguiente signo de puntuación ("no me gustó" → `no NEG_me NEG_gustó`).
- [ ] **N-gramas de caracteres:** `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5))` para resistir errores de tipeo y la falta de tildes.
- [ ] Features numéricas simples: número de oraciones, si hay conector contrastivo, posición del último conector.
- [ ] Medir con CV **cada** feature nueva por separado, documentar cuál ayuda y enviar a Kaggle las mejoras reales (envíos #2–#4).

> ⚠️ Para que el modelo se pueda guardar con `joblib`, las funciones de preprocesamiento deben estar definidas con `def` a nivel de módulo (no `lambda`) y deben estar definidas en el notebook **antes** de hacer `joblib.load`. Defínanlas dentro del propio notebook y no en un `.py` aparte: a Bloque Neón solo se suben el notebook y el modelo, así que un archivo externo rompería la replicabilidad.

### 2.5 Ajuste de hiperparámetros
- [ ] `GridSearchCV` o `RandomizedSearchCV` sobre el pipeline completo: `C`, `ngram_range`, `min_df`, `max_df`, `sublinear_tf`, el peso de cada vista (`transformer_weights`) y el tipo de modelo.
- [ ] Probar también `SGDClassifier`, `RidgeClassifier` y, si da tiempo, un `VotingClassifier` o `StackingClassifier` de los mejores.

### 2.6 Análisis de errores
- [ ] Matriz de confusión con predicciones fuera de fold (`cross_val_predict`).
- [ ] Leer ~30 errores y clasificarlos (¿conector que no detecté? ¿negación? ¿neutral confundido?). Ajustar la segmentación según lo que salga y volver a 2.4.
- [ ] Documentar en el notebook qué se intentó, qué funcionó y qué no. La evidencia de iteración cuenta para la nota.

### 2.7 Modelo final y entrega (semana 11)
- [ ] Reentrenar el mejor pipeline con **todo** `train.csv`.
- [ ] Predecir `eval.csv` y verificar el archivo: 3.000 filas, columnas `id,answer`, solo `negativo`/`neutral`/`positivo`.
- [ ] Guardar el modelo: `joblib.dump(pipeline, "models/parte1_final.joblib")`.
- [ ] **Prueba de replicabilidad:** reiniciar el kernel, ejecutar el notebook de principio a fin, cargar el `.joblib` y comprobar que las predicciones son idénticas al envío.
- [ ] Confirmar que ese modelo es exactamente el del **mejor score público** del grupo en Kaggle.
- [ ] Subir a Bloque Neón: el notebook y el archivo `.joblib`.
- [ ] Preparar la sustentación: cada integrante debe poder explicar cualquier parte del notebook.

---

## 3. Parte 2: Deep learning + aumentación de datos

Notebook nuevo en la raíz, p. ej. `parte2.ipynb`. Usa GPU de Kaggle Notebooks o Google Colab.

### 3.1 Reglas que no se pueden romper
- **Nunca** usar `eval.csv` para entrenar: ni pseudoetiquetado, ni preentrenamiento sobre sus textos, ni ajustar el tokenizador con él.
- La aumentación es obligatoria y hay que mostrar el desempeño **con y sin** ella.
- Los datos externos solo se pueden usar si su licencia permite redistribuirlos, y hay que citarlos con fuente, enlace y licencia.

### 3.2 Modelos (de menor a mayor complejidad)
- [ ] Mantener la misma partición de validación de la Parte 1 (o un holdout estratificado del 20 % con `random_state=42`) para comparar de forma justa.
- [ ] **Línea base DL:** MLP sobre TF-IDF (rápido y sirve de punto de comparación).
- [ ] **Secuenciales desde cero (Keras):** `Embedding` → `Bidirectional(LSTM/GRU)` → `Dense(3, softmax)`, y una `Conv1D` con max-pooling. Estos modelos leen la secuencia completa, así que deberían captar el efecto de "la última opinión gana".
- [ ] **Transfer learning (lo más prometedor):** fine-tuning de un modelo en español, por ejemplo BETO (`dccuchile/bert-base-spanish-wwm-cased`), RoBERTa-BNE (`PlanTL-GOB-ES/roberta-base-bne`) o `xlm-roberta-base`. Pocas épocas (2–4), `lr≈2e-5`, `max_length=128` (las reseñas tienen como máximo 110 palabras).
- [ ] Registrar cada experimento (arquitectura, hiperparámetros, accuracy de validación) en una tabla del notebook y en la bitácora de envíos.

### 3.3 Aumentación de datos (aplicar solo al fold de entrenamiento, nunca a validación)
Ideas pensadas para la estructura de estas reseñas, ordenadas de más a menos segura para conservar la etiqueta:
- [ ] **Insertar, quitar o reordenar frases de relleno** (logística, color, garantía) de reseñas neutrales en otras reseñas. No cambian la opinión, así que conservan la etiqueta. Hay que mantener la última cláusula de opinión al final.
- [ ] **Cambiar el aspecto:** reemplazar el sustantivo del aspecto ("el ensamblaje" ↔ "la batería" ↔ "el envío"). Conserva la etiqueta.
- [ ] **Ruido ortográfico:** quitar tildes y meter errores de tipeo aleatorios, igual que en los datos reales.
- [ ] **Reemplazo de sinónimos** de adjetivos de opinión manteniendo la polaridad ("excelente" ↔ "buenísimo").
- [ ] *(Avanzado)* **Inversión contrastiva:** en reseñas tipo "A. Aun así, B." intercambiar A y B e invertir la etiqueta. Le enseña al modelo explícitamente que el orden importa, pero hay que validar a mano una muestra.
- [ ] Experimento obligatorio: el mismo modelo **sin aumentación** frente a **con aumentación** (y, si se puede, cada técnica por separado). Mostrar los resultados en una tabla o gráfica en el notebook.

### 3.4 Modelo final y entrega (semana 14)
- [ ] Reentrenar el mejor modelo con todo `train.csv` (+ aumentación), predecir `eval.csv` y enviar a Kaggle.
- [ ] Guardar el modelo (`model.save("models/parte2_final.keras")` o `save_pretrained(...)` si es un transformer).
- [ ] El notebook debe incluir el análisis y la justificación de cada decisión, evidencia de iteración y la comparación con y sin aumentación.
- [ ] Documento adicional (`REFERENCIAS.md` o PDF):
  - los datos externos usados, con fuente, enlace y licencia (o "no se usaron datos externos"),
  - **el uso de IA generativa en el proyecto** (incluido el uso de asistentes como Claude para planear o programar).
- [ ] Subir a Bloque Neón: el notebook, el modelo y el documento de referencias.
- [ ] La competencia cierra en la semana 16: seguir mejorando y enviando hasta entonces, y asegurarse de que la entrega corresponda al mejor score público.

---

## 4. Estructura sugerida del repositorio

```
├── data/                  # train.csv, eval.csv, sample_submission.csv (no modificar)
├── parte1.ipynb           # notebooks en la raíz del proyecto
├── parte2.ipynb
├── models/                # .joblib / .keras finales
├── submissions/           # subNN_descripcion.csv
├── envios.md              # bitácora de envíos a Kaggle
├── requirements.txt
└── PLAN.md
```
