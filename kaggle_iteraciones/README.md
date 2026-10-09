# Envíos a Kaggle por iteración (Parte 1)

Cada archivo `.csv` de esta carpeta es un envío listo para subir (`id,answer`, 3.000 filas). El notebook que lo genera está en [`notebooks_iteraciones/`](../notebooks_iteraciones/) con el mismo nombre, y el modelo entrenado queda en `models/iteraciones/`. Cada notebook es autocontenido: se puede ejecutar de principio a fin desde la raíz del proyecto o desde su carpeta.

Súbanlos **en orden**. Los catorce envíos son distintos entre sí: cualquier par difiere en al menos 131 de las 3.000 predicciones.

| # | Archivo | Qué cambia | Accuracy CV | Score público Kaggle |
|---|---|---|---|---|
| 01 | `iter01_bolsa_naive_bayes.csv` | Bolsa de palabras + Naive Bayes | 0,7047 | 0,68888 |
| 02 | `iter02_tfidf_logistica.csv` | TF-IDF + regresión logística | 0,7252 | 0,73555 |
| 03 | `iter03_tfidf_random_forest.csv` | TF-IDF + Random Forest (empeora) | 0,6760 | 0,67888 |
| 04 | `iter04_logistica_ajustada.csv` | Logística ajustada con GridSearchCV | 0,7407 | 0,75000 |
| 05 | `iter05_stacking_clasico.csv` | Stacking logística + SVM + NB (techo de la bolsa de palabras) | 0,7420 | 0,75222 |
| 06 | `iter06_ultima_oracion.csv` | Solo la última oración | 0,8509 | 0,84777 |
| 07 | `iter07_tras_ultimo_conector.csv` | Solo lo que va después del último conector (empeora) | 0,8316 | 0,83555 |
| 08 | `iter08_tres_vistas.csv` | Texto completo + tras el conector + última oración | 0,8869 | 0,88111 |
| 09 | `iter09_stacking_de_vistas.csv` | Stacking de un modelo por vista (metamodelo logístico) | 0,8942 | 0,89111 |
| 10 | `iter10_stacking_boosting.csv` | Stacking ampliado + metamodelo HistGradientBoosting | 0,9012 | 0,89888 |
| 11 | `iter11_nbsvm_negacion_clausulas.csv` | NB-SVM + negación marcada + última cláusula en las vistas de palabras | 0,9076 | no enviado |
| 12 | `iter12_nbsvm_completo.csv` | NB-SVM también en los n-gramas de caracteres | 0,9093 | 0,89333 |
| 13 | `iter13_sin_relleno_polaridad.csv` | Dos modelos base más: final sin relleno y polaridad por segmento | 0,9123 | **0,90666** |
| 14 | `iter14_bases_sin_indecisas.csv` | Una segunda copia de cada modelo base, entrenada sin las reseñas indecisas (29 modelos base) | **0,9148** | pendiente |

Accuracy CV: validación cruzada de 5 pliegues sobre el 80 % de `train.csv` (mismo corte y semilla que `parte1convencional.ipynb`). Línea base trivial: 0,351. Las iteraciones 01 a 13 se midieron en otra máquina; la 13, reejecutada en la máquina de la 14, da 0,9130 (835 errores) porque la librería numérica es otra.

Evaluación en el 20 % apartado (solo para reportar; ninguna decisión se tomó con ella): iteración 10 **0,9083**, iteración 12 **0,9075**, iteración 13 **0,9142**, iteración 14 **0,9150**.

## Cómo se llegó de la 10 a la 13
Se exploraron varias ideas en paralelo:
- buscar una regla oculta en las reseñas mixtas
- puntuar la polaridad de cada oración
- marcar la negación
- partir en cláusulas
- ponderación NB-SVM
- análisis de errores
- ajuste del ensamble

Cada cambio solo se aceptó si reducía los errores sumando **varias validaciones cruzadas con particiones distintas** (semillas 42, 2026 y 7) y se confirmó en una cuarta partición (semilla 99) que no se usó para elegir.

Lo que funcionó:
- **11 y 12:** la ponderación NB-SVM con negación marcada y la última cláusula.
- **13:** los dos modelos base nuevos.

Lo que no funcionó: ninguna variable del texto predice la etiqueta de las reseñas mixtas indecisas o con "por otro lado" (~14 % de los datos). Modelos entrenados solo con ellas no pasan de 0,48-0,51, así que el techo realista está entre 0,91 y 0,93.

## El techo de los datos y cómo se mide desde la iteración 14
Al revisar los errores de la iteración 13 se encontró que las reseñas expresan la opinión de dos maneras:
- **Con un adjetivo** ("resultó frágil", "luce impecable"). Con dos opiniones opuestas gana siempre la última, con cualquier conector: 2.206 casos, ninguna excepción.
- **Con una frase hecha** ("me encantó … de principio a fin", "no vale lo que cuesta"). Con una sola frase, la etiqueta es su polaridad. Con una positiva y una negativa, la etiqueta es casi una moneda al aire: 50,4 % positivas, y la última gana el 51,5 % de las veces (1.608 reseñas). Frases, orden, conector, cierre, aspecto, bolsa de palabras y n-gramas de caracteres quedan entre 0,49 y 0,52 en validación cruzada.

Esa **zona de moneda al aire** es el 17,8 % del entrenamiento y el 18,8 % de `eval.csv`. Ahí estaban 788 de los 835 errores de la iteración 13; fuera de la zona ya acertaba el 99,4 %. Consecuencias:
- El techo es ≈ 0,91 en validación y ≈ 0,905 en Kaggle. El 0,90666 público de la iteración 13 es lo esperable de un modelo en el techo.
- El score público usa ~900 reseñas, de las cuales ~170 están en la zona: entre modelos igual de buenos se mueve ±0,7 puntos por azar. El puesto entre los equipos de arriba es en buena parte suerte.
- Lo que un modelo gana o pierde dentro de la zona es ruido, así que desde la iteración 14 un cambio se acepta por los **errores fuera de la zona**, en las mismas cuatro particiones (42, 2026, 7 y 99).

| Semilla | Errores fuera de la zona (13 → 14) | Errores totales (13 → 14) |
|---|---|---|
| 42 | 47 → 41 | 835 → 818 |
| 2026 | 52 → 44 | 860 → 841 |
| 7 | 50 → 39 | 850 → 832 |
| 99 | 56 → 51 | 843 → 856 |

La zona se identifica con una lista de frases escrita a mano que se usa **solo para medir** ([`experimentos/zona.py`](../experimentos/zona.py)); ningún modelo la usa. Scripts: [`experimentos/analisis_zona.py`](../experimentos/analisis_zona.py) (la moneda al aire) y [`experimentos/evaluar_por_zona.py`](../experimentos/evaluar_por_zona.py) (línea base y candidatos por zona).

Probado y descartado en la 14: entrenar **solo** con las copias sin reseñas indecisas (50, 53, 58 y 57 errores fuera de la zona: peor), cambiar el metamodelo (regresión logística, bagging, ExtraTrees y promedios: dentro del ruido) y un corrector ortográfico aprendido del vocabulario (efecto mínimo).

## Antes de entregar
- **Modelo a entregar:** el del envío con **mejor score público** (regla del curso). Hoy es el 13 (0,90666); falta el score del 14. Su notebook y su `.joblib` son el par que se sube a Bloque Neón. El score público solo usa 900 reseñas (margen ≈ ±1 punto), así que entre el 12, el 13 y el 14 puede ganar cualquiera por azar: el 12, con mejor validación que el 10, sacó menos (0,89333 frente a 0,89888).
- **Iteración 13 y número de hilos:** su primera celda fija 1 hilo en las librerías numéricas antes de importar numpy/sklearn. Sin eso, el modelo reentrenado en otro computador puede cambiar algunas predicciones. Para recargar el `.joblib` hay que ejecutar antes las celdas de definiciones del notebook.
- **Iteraciones 06 a 13:** parten la reseña (última oración, lo que sigue al último conector…) antes de aplicar TF-IDF. Siguen usando solo TF-IDF y clasificadores de scikit-learn, pero `CLAUDE.md` pide **confirmarlo con la profesora o el monitor** antes de entregarlas. Las iteraciones 01 a 05 son el enfoque convencional sin dudas.
- **Iteración 03:** su modelo (`iter03_tfidf_random_forest.joblib`) pesa 34 MB porque guarda los 300 árboles; los demás pesan menos de 2,2 MB hasta la 10, unos 18-20 MB de la 11 a la 13 (vocabularios de n-gramas de caracteres) y 38 MB la 14 (29 modelos base).
