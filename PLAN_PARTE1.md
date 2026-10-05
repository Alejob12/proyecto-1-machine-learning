# Plan de trabajo: Parte 1 (ML clásico)

Plan centrado solo en la Parte 1. Acompaña al notebook [parte1.ipynb](parte1.ipynb): cada paso indica en qué sección del notebook se trabaja. Para la visión general de las dos partes, ver [PLAN.md](PLAN.md).

**Cierre de la competencia:** semana 9 · **Entrega en Bloque Neón:** semana 11 (notebook + modelo `.joblib`)

---

## Reglas que dan 0 (revisarlas antes de cada idea nueva)

- Solo clasificadores de **scikit-learn**.
- **Nada** de redes neuronales, transformers ni embeddings preentrenados (word2vec, fastText, BERT…).
- Representación: bolsa de palabras, TF-IDF, n-gramas.
- Se entrega **un solo modelo**: el del mejor score público del grupo en Kaggle.
- Debe ser **replicable** solo con el notebook y el `.joblib` que se suben.

---

## Punto de partida: cifras de referencia

El notebook es solo un esqueleto: trae los títulos y el texto de cada sección, y **el código lo escribe el grupo**. Estas cifras salieron de una prueba previa del enfoque (validación cruzada de 5 folds sobre `train.csv`) y sirven para saber si van bien encaminados:

| Experimento | Accuracy CV |
|---|---|
| Predecir siempre la clase mayoritaria | 0.351 |
| TF-IDF texto completo + LogisticRegression | 0.739 |
| Solo la **primera** oración | 0.518 |
| Solo la **última** oración | 0.856 |
| Solo la última cláusula (después del último conector) | 0.866 |
| Texto completo + última cláusula | 0.890 |
| + última oración (+ negación) | **0.895** |

**Conclusión:** la hipótesis se confirma. La información está al final de la reseña, y el aporte de este proyecto es pasar de ~0.74 a ~0.89 haciendo que una bolsa de palabras "vea" el orden. La clase `neutral` ya casi no tiene errores (recall 0.996). Todo el margen de mejora está en separar **positivo de negativo**.

El trabajo del grupo es: implementar cada sección, documentar cada decisión, subir los envíos, analizar los errores restantes (~10 %) y mejorar a partir de ahí.

---

## Paso 1: Preparación (día 1)

- [ ] Crear el entorno virtual e instalar las dependencias fijadas:
  ```bash
  cd "Proyecto 1 Machine Learning"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  jupyter notebook parte1.ipynb
  ```
  `requirements.txt` fija `scikit-learn==1.9.1`. No lo cambien: un `.joblib` puede no cargar con otra versión de scikit-learn.
- [ ] Todos los integrantes: registrarse en Kaggle, unirse a la competencia (enlace en Bloque Neón) y formar el equipo.
- [ ] Anotar el score de la **línea base del curso** en la sección 10 del notebook.
- [ ] Repartir secciones del notebook entre los integrantes. El aporte individual vale 15 % y se ve en los commits de GitHub.
- [ ] Llenar nombre del grupo e integrantes en la primera celda.

## Paso 2: EDA (§1–3 del notebook)

- [ ] §1: imports, semilla fija (`SEED = 42`) y rutas a `data/`, `submissions/` y `models/`.
- [ ] §2: cargar `train.csv` y `eval.csv`.
- [ ] §3: graficar la distribución de clases y la longitud de las reseñas por clase, imprimir ejemplos de cada clase y contar cuántas veces aparece cada conector por clase.
- [ ] Completar la celda **"Conclusiones del EDA"**: balance de clases, estructura de las reseñas, por qué los conectores solos no separan las clases (la tabla de conectores lo muestra) y el ruido ortográfico.
- [ ] Opcional: agregar gráficas (número de oraciones por clase, posición del último conector).

## Paso 3: Modelos base → envío #1 (§4–5)

- [ ] §4: crear el `StratifiedKFold` de 5 folds y una función que evalúe cualquier pipeline con validación cruzada y guarde el resultado en una tabla de experimentos.
- [ ] §5: probar TF-IDF sobre el texto completo con varios clasificadores (`LogisticRegression`, `LinearSVC`, `MultinomialNB`, `ComplementNB`, `SGDClassifier`). No quitar stopwords.
- [ ] Explicar en markdown por qué se usa `StratifiedKFold` y por qué todo va dentro de un `Pipeline` (evita fuga de información entre folds).
- [ ] Completar **"Conclusiones de los modelos base"**: ¿por qué la bolsa de palabras se estanca en ~0.74? (pista: cuenta "excelente" y "decepcionante" igual sin importar cuál va al final).
- [ ] Escribir una función que entrene con todo `train.csv`, prediga `eval.csv`, valide el formato (`id,answer`, 3.000 filas, etiquetas válidas) y guarde el CSV en `submissions/`.
- [ ] Generar `submissions/sub01_tfidf_lr.csv` → subirlo a Kaggle → anotar el score en la bitácora (§10).

## Paso 4: Features de orden → envíos #2 y #3 (§6)

- [ ] **Confirmar con el profesor o monitor** que segmentar el texto antes de aplicar TF-IDF está permitido. Guardar la respuesta (correo o foro) por si la piden en la sustentación.
- [ ] Escribir las funciones de segmentación: normalizar (minúsculas, sin tildes), partir en oraciones, extraer la última oración, extraer la última cláusula (lo que va después del último conector) y marcar negaciones. Cada integrante debe poder explicarlas.
- [ ] §6.1: entrenar un modelo solo con la primera oración, otro solo con la última y otro solo con la última cláusula, y comparar.
- [ ] §6.2: combinar las vistas con `FeatureUnion`, agregando una a la vez y midiendo cada una.
- [ ] Completar las conclusiones de §6.1 (validación de la hipótesis) y §6.2 (ablación).
- [ ] Envío #2: "texto completo + última cláusula".
- [ ] Envío #3: la mejor combinación de la ablación.

> ⚠️ Toda función que vaya dentro del pipeline se define con `def` **dentro del notebook** (no `lambda`, no un `.py` aparte), o el `.joblib` no podrá cargarse a partir de lo entregado.

## Paso 5: Análisis de errores (§8): aquí está la mejora real

- [ ] §8: obtener predicciones fuera de fold (`cross_val_predict`), mostrar la matriz de confusión y leer **al menos 30 errores**. Llenar la tabla de tipos de error del notebook.
- [ ] Patrones a buscar (vistos en una primera revisión):
  - **Conectores que faltan en la lista:** p. ej. *"Por cierto, quedé encantado con el sonido"* → hay que agregar "por cierto". Buscar otros ("total", "la verdad", "eso sí", "al menos", "lo bueno es que"…).
  - **Frases de relleno después de la opinión final:** *"…Cada quien que juzgue, yo nada más cuento mi experiencia."* En esos casos la "última oración" no es la última **opinión**.
  - **Opiniones sin conector:** dos opiniones seguidas sin "pero" ni "aun así".
  - Casos ambiguos o posible ruido en las etiquetas (no perder tiempo en ellos).
- [ ] Cada cambio se mide con CV en §6.2. Solo se envía a Kaggle lo que mejore → **envío #4**.

### Ideas para subir más (en orden de esfuerzo)
1. **Ampliar la lista de conectores** según los errores (barato y efectivo).
2. **Última oración de opinión:** saltar las oraciones de relleno al buscar la última. Las oraciones de las reseñas `neutral` son, por definición, relleno: se puede usar su vocabulario para detectar si una oración es relleno. *Ojo:* si esto se aprende de los datos, tiene que pasar **dentro** del pipeline (un transformer con `fit`) para no filtrar información entre folds.
3. **Penúltima cláusula** como vista adicional, para que el modelo vea el contraste completo ("A, aun así B").
4. **Features numéricas:** número de oraciones, ¿hay conector?, posición relativa del último conector.
5. **Otros clasificadores sobre las mismas features:** `LinearSVC`, `SGDClassifier`, y luego `VotingClassifier`/`StackingClassifier` con los mejores.

## Paso 6: Ajuste de hiperparámetros → envío #5 (§7)

- [ ] Hacer una búsqueda (`RandomizedSearchCV` o `GridSearchCV`) sobre el mejor pipeline de §6.
- [ ] Parámetros a explorar (`C`, `ngram_range` de cada vista, `min_df`, `sublinear_tf`, `transformer_weights` de la `FeatureUnion`); empezar con pocas iteraciones y subir a 20–50 cuando haya tiempo.
- [ ] Envío #5 con el mejor modelo ajustado. Con esto ya se cumple el **mínimo de 5 envíos distintos**.
- [ ] Seguir iterando y enviando hasta la semana 9. El percentil final se calcula con el **score privado**, así que hay que confiar más en la CV que en pequeñas subidas del público.

## Paso 7: Modelo final y entrega (§9–10, semana 11)

- [ ] §9: reentrenar con todo `train.csv` el modelo del **mejor score público** de Kaggle (no necesariamente el último), generar el envío y guardarlo con `joblib.dump` en `models/parte1_final.joblib`.
- [ ] Escribir la prueba de replicabilidad: cargar el `.joblib` y comprobar que sus predicciones son idénticas a las del envío.
- [ ] **Reiniciar el kernel y ejecutar todo** (*Kernel → Restart & Run All*). Debe terminar sin errores.
- [ ] Verificar que `submissions/sub_parte1_final.csv` es idéntico al archivo que dio el mejor score en Kaggle.
- [ ] Completar la bitácora de envíos y las **conclusiones finales** (incluida la limitación de la bolsa de palabras, que motiva la Parte 2).
- [ ] Revisar que cada sección tenga su markdown explicativo. La documentación vale 15 % y la calidad del proceso otro 15 %.
- [ ] Subir a Bloque Neón: `parte1.ipynb` (con salidas) + `models/parte1_final.joblib`.
- [ ] Ensayar la sustentación: cualquier integrante debe poder explicar cualquier celda.

---

## Plan de envíos sugerido

| # | Qué enviar | CV esperada |
|---|---|---|
| 1 | TF-IDF texto completo + LogisticRegression | ~0.739 |
| 2 | + vista de última cláusula | ~0.890 |
| 3 | Mejor combinación de la ablación (§6.2) | ~0.895 |
| 4 | Mejoras del análisis de errores (conectores, relleno) | > 0.895 |
| 5 | Mejor modelo con hiperparámetros ajustados / ensamble | lo más alto posible |
