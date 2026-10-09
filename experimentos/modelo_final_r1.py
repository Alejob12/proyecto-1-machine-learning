"""Modelo final (Parte 1, ML clásico): stacking de 13 modelos base NB-SVM + metamodelo HistGradientBoosting.

Autocontenido: solo usa re, unicodedata, numpy y scikit-learn. Todas las funciones y clases están a nivel de
módulo (sin lambdas), así que el modelo entrenado se guarda y recarga con joblib. Todo lo aprendido de los datos
(vocabularios, log-count ratios NB, palabras de opinión, metamodelo) ocurre dentro de fit().

Cambios respecto a iter10 (12 modelos base TF-IDF + LR, metamodelo HGB):
  1. NB-SVM (Wang y Manning, 2012) en las vistas de palabras: negación marcada (no_X hasta la puntuación) ->
     unigramas + bigramas binarios -> escalado por el log-count ratio de Naive Bayes de cada clase -> LR(C=1).
  2. Vista nueva: última cláusula (corte en oraciones y antes de ", pero", " aunque", " y ", ...).
  3. Las dos vistas de n-gramas de caracteres (char_wb 2-5) también binarias + escalado NB + LR(C=1).
CV estándar (5 pliegues, SEED=42, X_ent): 0.9093 ± 0.0074 (iter10: 0.8998 ± 0.0082 en el mismo harness).
CV alternativo (StratifiedKFold(5, shuffle=True, random_state=2026)): 0.9098 ± 0.0032 (iter10: 0.8993 ± 0.0070).
"""
import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

SEED = 42  # semilla fija: resultados replicables
LABELS = ["negativo", "neutral", "positivo"]

# ----------------------------------------------------------------------------------------------
# 1. Limpieza del texto
# ----------------------------------------------------------------------------------------------
# Pares que deja una codificación dañada (UTF-8 leído como Latin-1) -> letra correcta
MOJIBAKE = {"Ã¡": "á", "Ã©": "é", "Ã\xad": "í", "Ã³": "ó", "Ãº": "ú", "Ã±": "ñ"}
# Variante en la que además la Ã perdió su tilde: "llegA3" -> "llegó"
MOJIBAKE_DEGRADADO = {"A¡": "á", "A©": "é", "A\xad": "í", "A3": "ó", "A³": "ó", "Aº": "ú", "A±": "ñ"}
PATRON_DEGRADADO = re.compile(r"(?<=[a-zñ])A[¡©\xad3³º±]")  # solo dentro de una palabra


def reparar_par(coincidencia):
    return MOJIBAKE_DEGRADADO[coincidencia.group()]


def corregir_texto(texto):
    """Repara la codificación, pasa a minúsculas y quita tildes/ñ. No modifica el original."""
    for malo, bueno in MOJIBAKE.items():
        texto = texto.replace(malo, bueno)
    texto = PATRON_DEGRADADO.sub(reparar_par, texto)
    texto = texto.replace("\xad", "")
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


# ----------------------------------------------------------------------------------------------
# 2. Segmentación: oraciones, conectores (lista escrita a mano) y frases de cierre indeciso (regex)
# ----------------------------------------------------------------------------------------------
CONECTORES = ["aun asi", "eso si", "por otro lado", "por cierto", "de paso", "al final", "con todo",
              "a fin de cuentas", "sin embargo", "en fin", "despues de todo", "pero", "aunque", "ahora",
              "ademas", "no obstante", "de todas formas", "igual", "lo unico", "nada mas"]
PATRON_CONECTOR = re.compile(r"\b(" + "|".join(CONECTORES) + r")\b")
PATRON_ORACION = re.compile(r"(?<=[.;:!?])\s+")

CIERRE = re.compile(r"sentimientos encontrados|con dudas|no me decido|cada quien|a criterio|ni lo mejor ni lo peor"
                    r"|ni tan bueno ni tan malo|hay de todo|sopes|para gustos|no se si me quedo|lo dejo a tu"
                    r"|cada cosa por su lado|nomas cuento|juzgue|no se que pensar|ahi lo dejo|mis dudas|ni fu ni fa"
                    r"|no sabria|en duda|que recomendar|de cada lado|otras no|no se bien")


def oraciones(texto):
    """Parte un texto ya corregido en oraciones."""
    return [o for o in PATRON_ORACION.split(texto.strip()) if o]


# ----------------------------------------------------------------------------------------------
# 3. Vistas del texto: cada función recibe textos crudos y devuelve una parte de cada reseña
# ----------------------------------------------------------------------------------------------
def texto_completo(textos):
    return [corregir_texto(t) for t in textos]


def ultima_oracion(textos):
    return [oraciones(corregir_texto(t))[-1] for t in textos]


def primera_oracion(textos):
    return [oraciones(corregir_texto(t))[0] for t in textos]


def penultima_oracion(textos):
    salida = []
    for t in textos:
        partes = oraciones(corregir_texto(t))
        salida.append(partes[-2] if len(partes) > 1 else "")
    return salida


def tras_ultimo_conector(textos):
    """Desde el último conector hasta el final (o la última oración si no hay conector)."""
    salida = []
    for t in textos:
        t = corregir_texto(t)
        conectores = list(PATRON_CONECTOR.finditer(t))
        salida.append(t[conectores[-1].start():] if conectores else oraciones(t)[-1])
    return salida


def antes_del_ultimo_conector(textos):
    salida = []
    for t in textos:
        t = corregir_texto(t)
        conectores = list(PATRON_CONECTOR.finditer(t))
        salida.append(t[:conectores[-1].start()] if conectores else "")
    return salida


def con_conector(textos):
    """Cada palabra lleva pegado el conector que abre su oración: 'aunasi_feliz', 'porotrolado_harto', 'nada_llego'."""
    salida = []
    for t in textos:
        tokens = []
        for o in oraciones(corregir_texto(t)):
            conector = next((c for c in CONECTORES + ["la verdad"] if o.startswith(c)), "nada").replace(" ", "")
            tokens += [f"{conector}_{p}" for p in re.findall(r"\w\w+", o)]
        salida.append(" ".join(tokens))
    return salida


def cierre_indeciso(textos):
    """Marca si la reseña tiene una frase de indecisión ("tengo sentimientos encontrados", "ni fu ni fa")."""
    return ["cierre_si" if CIERRE.search(corregir_texto(t)) else "cierre_no" for t in textos]


def etiquetas_texto(y):
    """StackingClassifier entrena los modelos base con la y codificada (0, 1, 2 en orden alfabético):
    la devolvemos a texto para que las reglas que miran 'neutral' funcionen igual dentro y fuera del stack."""
    y = np.asarray(y)
    if y.dtype.kind in "iu":
        y = np.asarray(LABELS)[y]
    return y


class UltimaOpinion(BaseEstimator, TransformerMixin):
    """Última oración que contiene alguna palabra de opinión y no es un cierre indeciso.

    Las palabras de opinión se aprenden en `fit` (las `k` de mayor peso en una regresión logística
    sobre TF-IDF), así que en cada pliegue se aprenden solo con su parte de entrenamiento.
    Nota: dentro del StackingClassifier la y llega codificada (0/1/2), así que `polares` es siempre True y
    la regresión se entrena con las 3 clases (se usa la fila de 'negativo'); se deja así porque es el
    comportamiento con el que se midió el modelo.
    """

    def __init__(self, k=400):
        self.k = k

    def fit(self, X, y):
        y = np.asarray(y)
        polares = y != "neutral"
        vectorizador = TfidfVectorizer(min_df=3)
        M = vectorizador.fit_transform(texto_completo(np.asarray(X)[polares]))
        pesos = LogisticRegression(C=1, max_iter=3000).fit(M, y[polares]).coef_[0]
        vocabulario = vectorizador.get_feature_names_out()
        self.palabras_opinion_ = set(vocabulario[np.argsort(-np.abs(pesos))[:self.k]])
        return self

    def transform(self, X):
        salida = []
        for t in X:
            partes = oraciones(corregir_texto(t))
            elegida = next((o for o in reversed(partes) if not CIERRE.search(o)
                            and set(re.findall(r"\w\w+", o)) & self.palabras_opinion_), partes[-1])
            salida.append(elegida)
        return salida


# ----------------------------------------------------------------------------------------------
# 4. Marcado de negación y segmentación en cláusulas (reglas escritas a mano)
# ----------------------------------------------------------------------------------------------
NEGADORES = {"no", "ni", "nunca", "sin", "tampoco", "jamas", "nada", "nadie", "ningun", "ninguna", "ninguno"}
PATRON_TOKEN = re.compile(r"\w+|[.,;:!?]")
PUNTUACION = set(".,;:!?")


def marcar_negacion_texto(texto):
    """'no lo recomiendo, la verdad' -> 'no no_lo no_recomiendo , la verdad' (el alcance termina en la puntuación)."""
    salida, negado = [], False
    for tok in PATRON_TOKEN.findall(texto):
        if tok in PUNTUACION:
            negado = False
            salida.append(tok)
        elif tok in NEGADORES:
            negado = True
            salida.append(tok)
        else:
            salida.append("no_" + tok if negado else tok)
    return " ".join(salida)


def marcar_negacion(textos):
    return [marcar_negacion_texto(t) for t in textos]


# Corte de cláusulas: fin de oración, o coma/espacio antes de un conector ("..., pero", " aunque ", " y ")
PATRON_CLAUSULA = re.compile(r"(?<=[.;:!?])\s+|,\s+(?=(?:pero|aunque|aun asi|sin embargo|eso si|y)\b)"
                             r"|\s+(?=(?:pero|aunque|y)\s)")


def clausulas_texto(texto_corregido):
    return [c.strip(" ,") for c in PATRON_CLAUSULA.split(texto_corregido.strip()) if c and c.strip(" ,")]


def ultima_clausula(textos):
    return [(clausulas_texto(corregir_texto(t)) or [""])[-1] for t in textos]


# ----------------------------------------------------------------------------------------------
# 5. NB-SVM (Wang y Manning, 2012): bolsa de n-gramas binaria escalada por el log-count ratio de cada clase
# ----------------------------------------------------------------------------------------------
class EscaladoNBClase(BaseEstimator, TransformerMixin):
    """Escalado NB para UNA clase (la k-ésima en orden): r = log((p/|p|) / (q/|q|)), con p = alpha + conteos
    binarios (presencia) en la clase y q = alpha + conteos en el resto (una contra el resto).
    transform devuelve X * r. Los r se aprenden en fit, solo con los datos de entrenamiento del pliegue."""

    def __init__(self, k=0, alpha=1.0):
        self.k = k
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        Xb = X.tocsr().copy()
        Xb.data = np.ones_like(Xb.data)  # presencia / ausencia
        clase = np.unique(y)[self.k]
        p = self.alpha + np.asarray(Xb[y == clase].sum(0)).ravel()
        q = self.alpha + np.asarray(Xb[y != clase].sum(0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum()))
        return self

    def transform(self, X):
        return X.tocsr().multiply(self.r_).tocsr()


def escalado_nb(alpha=1.0):
    """NB-SVM multiclase: un bloque de features escaladas por cada clase (negativo, neutral, positivo),
    concatenados por FeatureUnion: [X * r_neg | X * r_neu | X * r_pos]."""
    return FeatureUnion([(f"clase_{k}", EscaladoNBClase(k=k, alpha=alpha)) for k in range(len(LABELS))])


def bolsa_nb_palabras():
    """Unigramas y bigramas binarios (norma L2) con negación marcada, escalados por NB."""
    return [("neg", FunctionTransformer(marcar_negacion)),
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), binary=True, use_idf=False, norm="l2")),
            ("nb", escalado_nb(alpha=1.0))]


def vista_nb(funcion):
    return Pipeline([("parte", FunctionTransformer(funcion))] + bolsa_nb_palabras())


def clf_nb(funcion, C=1):
    """Modelo base: una vista del texto -> NB-SVM -> regresión logística."""
    return Pipeline([("vista", vista_nb(funcion)), ("clf", LogisticRegression(C=C, max_iter=3000))])


def clf_char_nb(funcion, C=1):
    """N-gramas de caracteres 2-5 dentro de palabra (toleran tipeos: "mdia", "cincco"), binarios + NB."""
    vista = Pipeline([("parte", FunctionTransformer(funcion)),
                      ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(2, 5), binary=True, use_idf=False,
                                                norm="l2", analyzer="char_wb")),
                      ("nb", escalado_nb(alpha=1.0))])
    return Pipeline([("vista", vista), ("clf", LogisticRegression(C=C, max_iter=3000))])


def nb_tres(C=1):
    """Tres vistas NB-SVM (texto completo, tras el último conector, última oración) unidas -> LR."""
    vistas = [("completo", vista_nb(texto_completo)),
              ("tras_conector", vista_nb(tras_ultimo_conector)),
              ("ultima", vista_nb(ultima_oracion))]
    return Pipeline([("vistas", FeatureUnion(vistas)), ("clf", LogisticRegression(C=C, max_iter=3000))])


def ultima_opinion_nb(C=1):
    return Pipeline([("parte", UltimaOpinion(k=400)),
                     ("neg", FunctionTransformer(marcar_negacion)),
                     ("bow", TfidfVectorizer(min_df=2, ngram_range=(1, 2), binary=True, use_idf=False)),
                     ("nb", escalado_nb()),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


def cierre_lr():
    """Indicador de cierre indeciso -> LR (igual que en iter10)."""
    return Pipeline([("parte", FunctionTransformer(cierre_indeciso)),
                     ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)),
                     ("clf", LogisticRegression(C=1, max_iter=3000))])


def modelos_base_nb():
    """Los 13 modelos base del stack NB-SVM (en este orden)."""
    return [("nb_tres", nb_tres()),
            ("completo", clf_nb(texto_completo)),
            ("tras_conector", clf_nb(tras_ultimo_conector)),
            ("ultima", clf_nb(ultima_oracion)),
            ("primera", clf_nb(primera_oracion)),
            ("penultima", clf_nb(penultima_oracion)),
            ("con_conector", clf_nb(con_conector)),
            ("cierre", cierre_lr()),
            ("ultima_opinion", ultima_opinion_nb()),
            ("antes_del_conector", clf_nb(antes_del_ultimo_conector)),
            ("ultima_clausula", clf_nb(ultima_clausula)),
            ("caracteres_tras_conector", clf_char_nb(tras_ultimo_conector)),
            ("caracteres_ultima", clf_char_nb(ultima_oracion))]


# ----------------------------------------------------------------------------------------------
# 6. Modelo final: stacking de 13 modelos base NB-SVM + metamodelo HistGradientBoosting
# ----------------------------------------------------------------------------------------------
def construir_modelo():
    """Estimador SIN entrenar. El StackingClassifier entrena el metamodelo con predicciones fuera de pliegue
    (CV interno de 5 pliegues) de los modelos base, así que todo lo aprendido ocurre dentro de fit()."""
    return StackingClassifier(
        modelos_base_nb(),
        final_estimator=HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    )
