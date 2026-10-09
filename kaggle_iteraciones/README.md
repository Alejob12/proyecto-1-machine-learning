# Envíos a Kaggle por iteración (Parte 1)

Cada archivo `.csv` de esta carpeta es un envío listo para subir (`id,answer`, 3.000 filas). El notebook que lo genera está en [`notebooks_iteraciones/`](../notebooks_iteraciones/) con el mismo nombre, y el modelo entrenado queda en `models/iteraciones/`. Cada notebook es autocontenido: se puede ejecutar de principio a fin desde la raíz del proyecto o desde su carpeta.

Súbanlos **en orden**. Los diez envíos son distintos entre sí: cualquier par difiere en al menos 135 de las 3.000 predicciones.

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
| 11 | `iter11_nbsvm_negacion_clausulas.csv` | NB-SVM + negación marcada + última cláusula en las vistas de palabras | **0,9076** | |

Accuracy CV: validación cruzada de 5 pliegues sobre el 80 % de `train.csv` (mismo corte y semilla que `parte1convencional.ipynb`). Solo la iteración 10 se evaluó en el 20 % apartado: **0,9083**. Línea base trivial: 0,351.

## Antes de entregar
- **Modelo a entregar:** el del envío con **mejor score público** (regla del curso). Se espera que sea el 10. Su notebook y su `.joblib` son el par que se sube a Bloque Neón.
- **Iteraciones 06 a 10:** parten la reseña (última oración, lo que sigue al último conector…) antes de aplicar TF-IDF. Siguen usando solo TF-IDF y clasificadores de scikit-learn, pero `CLAUDE.md` pide **confirmarlo con la profesora o el monitor** antes de entregarlas. Las iteraciones 01 a 05 son el enfoque convencional sin dudas.
- **Iteración 03:** su modelo (`iter03_tfidf_random_forest.joblib`) pesa 34 MB porque guarda los 300 árboles; los demás pesan menos de 2,2 MB.
