# Envíos a Kaggle por iteración (Parte 1)

Cada archivo `.csv` de esta carpeta es un envío listo para subir (`id,answer`, 3.000 filas). El notebook que lo genera está en [`notebooks_iteraciones/`](../notebooks_iteraciones/) con el mismo nombre, y el modelo entrenado queda en `models/iteraciones/`. Cada notebook es autocontenido: se puede ejecutar de principio a fin desde la raíz del proyecto o desde su carpeta.

Súbanlos **en orden**. Los trece envíos son distintos entre sí: cualquier par difiere en al menos 131 de las 3.000 predicciones.

| # | Archivo | Qué cambia | Accuracy CV | Score público Kaggle |
|---|---|---|---|---|
| 01 | `iter01_bolsa_naive_bayes.csv` | Bolsa de palabras + Naive Bayes | 0,7047 | |
| 02 | `iter02_tfidf_logistica.csv` | TF-IDF + regresión logística | 0,7252 | |
| 03 | `iter03_tfidf_random_forest.csv` | TF-IDF + Random Forest (empeora) | 0,6760 | |
| 04 | `iter04_logistica_ajustada.csv` | Logística ajustada con GridSearchCV | 0,7407 | |
| 05 | `iter05_stacking_clasico.csv` | Stacking logística + SVM + NB (techo de la bolsa de palabras) | 0,7420 | |
| 06 | `iter06_ultima_oracion.csv` | Solo la última oración | 0,8509 | |
| 07 | `iter07_tras_ultimo_conector.csv` | Solo lo que va después del último conector (empeora) | 0,8316 | |
| 08 | `iter08_tres_vistas.csv` | Texto completo + tras el conector + última oración | 0,8869 | |
| 09 | `iter09_stacking_de_vistas.csv` | Stacking de un modelo por vista (metamodelo logístico) | 0,8942 | |
| 10 | `iter10_stacking_boosting.csv` | Stacking ampliado + metamodelo HistGradientBoosting | 0,9012 | |
| 11 | `iter11_nbsvm_negacion_clausulas.csv` | NB-SVM + negación marcada + última cláusula en las vistas de palabras | 0,9076 | |
| 12 | `iter12_nbsvm_completo.csv` | NB-SVM también en los n-gramas de caracteres | 0,9093 | |
| 13 | `iter13_sin_relleno_polaridad.csv` | Dos modelos base más: final sin relleno y polaridad por segmento | **0,9123** | |

Accuracy CV: validación cruzada de 5 pliegues sobre el 80 % de `train.csv` (mismo corte y semilla que `parte1convencional.ipynb`). Línea base trivial: 0,351.

Evaluación en el 20 % apartado, que nunca se usó para elegir: iteración 10 **0,9083**, iteración 12 **0,9075**, iteración 13 **0,9142**.

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

## Antes de entregar
- **Modelo a entregar:** el del envío con **mejor score público** (regla del curso). Por validación cruzada y por la prueba, el mejor es el 13. Su notebook y su `.joblib` son el par que se sube a Bloque Neón. El score público solo usa 900 reseñas (margen ≈ ±1 punto), así que entre el 11, el 12 y el 13 puede ganar cualquiera por azar.
- **Iteración 13 y número de hilos:** su primera celda fija 1 hilo en las librerías numéricas antes de importar numpy/sklearn. Sin eso, el modelo reentrenado en otro computador puede cambiar algunas predicciones. Para recargar el `.joblib` hay que ejecutar antes las celdas de definiciones del notebook.
- **Iteraciones 06 a 13:** parten la reseña (última oración, lo que sigue al último conector…) antes de aplicar TF-IDF. Siguen usando solo TF-IDF y clasificadores de scikit-learn, pero `CLAUDE.md` pide **confirmarlo con la profesora o el monitor** antes de entregarlas. Las iteraciones 01 a 05 son el enfoque convencional sin dudas.
- **Iteración 03:** su modelo (`iter03_tfidf_random_forest.joblib`) pesa 34 MB porque guarda los 300 árboles; los demás pesan menos de 2,2 MB hasta la 10 y unos 18-20 MB de la 11 a la 13 (vocabularios de n-gramas de caracteres).
