"""Modelo final 2 (Parte 1, ML clásico): stacking de 15 modelos base + metamodelo HistGradientBoosting.

Autocontenido: solo usa re, unicodedata, numpy y scikit-learn. Todas las funciones y clases están a nivel de
módulo (sin lambdas), así que el modelo entrenado se guarda y recarga con joblib. Todo lo aprendido de los datos
(vocabularios, log-count ratios NB, palabras de opinión, clasificador de segmentos, metamodelo) ocurre en fit().

Cambios respecto al modelo final de la ronda 1 (13 bases NB-SVM + HGB):
  1. Base "tras_conector_sr": desde el último conector hasta el final, después de quitar las oraciones de relleno
     del final (envío, dónde se guarda...). Una oración es relleno si no tiene ninguna palabra de opinión; esas
     palabras se APRENDEN en fit (>= 4 veces más frecuentes, en proporción, en reseñas polares que en neutrales).
     -> TF-IDF -> LR(C=3).
  2. Base "polaridad_segmentos": un clasificador de segmentos (negativo / nada / positivo) aprendido en fit con
     supervisión débil (conectores contrastivos + reseñas neutrales + autoentrenamiento); cada palabra se marca con
     la polaridad de su segmento y su papel (primera / última opinión) -> TF-IDF -> LR(C=3).
     Ninguna de las dos usa léxicos de sentimiento escritos a mano: solo listas ESTRUCTURALES (conectores, cierre).
  3. El metamodelo HGB lleva early_stopping=False: con 'auto' se activaría al entrenar con >10.000 filas
     (las 12.000 de train.csv) y el modelo entregado sería distinto del validado.

Resultados (X_ent, 9.600 reseñas; nunca X_prueba / y_prueba / eval.csv):
  CV estándar REAL, evaluar(construir_modelo()) del harness (StratifiedKFold(5, shuffle, 42)):
      0.9123 ± 0.0071 (842 errores)   final ronda 1: 0.9093 ± 0.0074 (871)
  Protocolo de 3 semillas externas (CV interno con semilla 42, como el final), errores:
      semilla 42:   842 vs 871  (solo falla final2 208 / solo falla final r1 237)  acc 0.9123 vs 0.9093
      semilla 2026: 847 vs 866  (227 / 246)                                         acc 0.9118 vs 0.9098
      semilla 7:    841 vs 834  (226 / 219)                                         acc 0.9124 vs 0.9131
      total 2530 vs 2571 (-41), mejora en 2/3 semillas -> CUMPLE el protocolo.
  Chequeo con otra semilla del CV interno (1, en ambos): 2585 vs 2678 (-93), mejora en 3/3. Ver INFORME.
"""
import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
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
# 6. Base nueva (ronda 2): tras el último conector, SIN las oraciones de relleno del final
# ----------------------------------------------------------------------------------------------
class VistaSinRelleno(BaseEstimator, TransformerMixin):
    """Quita las oraciones de relleno del final de la reseña y devuelve lo que sigue al último conector
    (modo="tras_conector") o la última oración (modo="ultima").

    Palabras de opinión aprendidas en fit: p(palabra | polar) / p(palabra | neutral) >= razon (presencia binaria,
    suavizado +1). No dicen si la palabra es positiva o negativa, solo que aparece en opiniones. Una oración sin
    ninguna palabra de opinión se considera relleno. La y se decodifica porque dentro del stack llega 0/1/2."""

    def __init__(self, modo="tras_conector", razon=4.0, min_df=3):
        self.modo = modo
        self.razon = razon
        self.min_df = min_df

    def fit(self, X, y):
        y = etiquetas_texto(y)
        cv = CountVectorizer(binary=True, min_df=self.min_df, token_pattern=r"\b\w\w+\b")
        A = cv.fit_transform(texto_completo(np.asarray(X, dtype=object)))
        neu = y == "neutral"
        p_neu = (np.asarray(A[neu].sum(0)).ravel() + 1) / (neu.sum() + 2)
        p_pol = (np.asarray(A[~neu].sum(0)).ravel() + 1) / ((~neu).sum() + 2)
        self.palabras_opinion_ = set(cv.get_feature_names_out()[p_pol / p_neu >= self.razon])
        return self

    def _es_opinion(self, o):
        return bool(set(re.findall(r"\b\w\w+\b", o)) & self.palabras_opinion_)

    def transform(self, X):
        salida = []
        for t in X:
            partes = oraciones(corregir_texto(t))
            while len(partes) > 1 and not self._es_opinion(partes[-1]):
                partes = partes[:-1]  # quitar relleno del final (siempre queda al menos una oración)
            if self.modo == "ultima":
                salida.append(partes[-1])
            else:
                texto = " ".join(partes)
                conectores = list(PATRON_CONECTOR.finditer(texto))
                salida.append(texto[conectores[-1].start():] if conectores else partes[-1])
        return salida


def tras_conector_sr(C=3):
    """Modelo base: vista sin relleno (tras el último conector) -> TF-IDF -> LR."""
    return Pipeline([("parte", VistaSinRelleno(modo="tras_conector", razon=4.0)),
                     ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


# ----------------------------------------------------------------------------------------------
# 7. Base nueva (ronda 2): polaridad por segmento aprendida en fit (supervisión débil con conectores)
# ----------------------------------------------------------------------------------------------
# Conectores agrupados por función discursiva (lista ESTRUCTURAL escrita a mano, sin palabras de sentimiento)
GRUPO_CONECTOR = {
    "aun asi": "contra_fuerte", "al final": "contra_fuerte", "con todo": "contra_fuerte",
    "a fin de cuentas": "contra_fuerte", "sin embargo": "contra_fuerte", "en fin": "contra_fuerte",
    "despues de todo": "contra_fuerte", "no obstante": "contra_fuerte", "de todas formas": "contra_fuerte",
    "pero": "contra_debil", "aunque": "contra_debil", "eso si": "concesivo", "ahora": "concesivo",
    "igual": "concesivo", "lo unico": "concesivo", "nada mas": "concesivo",
    "por otro lado": "aditivo", "por cierto": "aditivo", "de paso": "aditivo", "ademas": "aditivo", "y": "aditivo",
}
LISTA_CON = sorted(GRUPO_CONECTOR, key=len, reverse=True)
PATRON_SEG = re.compile(r"(?<!hasta )(?<!por )(?<!desde )\b(" + "|".join(LISTA_CON) + r")\b")
PATRON_PALABRA = re.compile(r"\w+")


def segmentar(texto_corr):
    """Parte un texto corregido en segmentos (texto, conectores_que_lo_abren, índice_oración, es_cierre_indeciso):
    corta en cada oración y antes de cada conector de GRUPO_CONECTOR."""
    segs = []
    ors = oraciones(texto_corr)
    for k, o in enumerate(ors):
        trozos, inicio, actual_con = [], 0, []
        for m in PATRON_SEG.finditer(o):
            previo = o[inicio:m.start()].strip(" ,.;:!?")
            if previo and PATRON_PALABRA.search(previo):
                trozos.append((previo, actual_con))
                actual_con = []
            actual_con = actual_con + [m.group()]
            inicio = m.end()
        resto = o[inicio:].strip(" ,.;:!?")
        if resto and PATRON_PALABRA.search(resto):
            trozos.append((resto, actual_con))
        for t, cs in trozos:
            segs.append((t, cs, k, bool(CIERRE.search(t))))
    if not segs:
        segs = [(texto_corr, [], 0, False)]
    return segs, len(ors)


def preparar(textos):
    return [segmentar(corregir_texto(t)) for t in textos]


class PolaridadSegmentos(BaseEstimator, TransformerMixin):
    """Aprende en fit un clasificador de segmentos (negativo / nada / positivo) y devuelve el texto con cada palabra
    marcada por la polaridad de su segmento y su papel en la secuencia: 'P_bateria', 'Nult_bateria' (última opinión
    negativa), 'Ppri_...' (primera opinión positiva), 'C_...' (cierre indeciso), 'O_...' (sin opinión).

    Supervisión débil (sin léxico de sentimiento): los segmentos abiertos por un conector contrastivo heredan la
    etiqueta de la reseña y los de reseñas neutrales son 'nada'; luego `iter_auto` rondas de autoentrenamiento
    con la regla del contraste (la opinión justo antes del conector contrastivo tiene la polaridad contraria).
    En fit_transform las marcas del train salen de validación cruzada interna (cross-fitting), para que la LR
    que va después no vea marcas sobreajustadas."""

    def __init__(self, C_seg=3.0, iter_auto=2, umbral_op=0.4, umbral_pol=0.4, n_internos=5,
                 semillas=("contra_fuerte", "contra_debil")):
        self.C_seg = C_seg
        self.iter_auto = iter_auto
        self.umbral_op = umbral_op
        self.umbral_pol = umbral_pol
        self.n_internos = n_internos
        self.semillas = semillas

    def _entrenar_seg(self, filas, etiquetas):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = OneVsRestClassifier(LogisticRegression(C=self.C_seg, solver="liblinear"))
        clf.fit(vec.fit_transform(filas), etiquetas)
        return {"vec2": vec, "lr2": clf}

    def _probas(self, modelo, planos):
        """Por segmento: p(pos) - p(neg) y p(opinión) = 1 - p(nada)."""
        P = modelo["lr2"].predict_proba(modelo["vec2"].transform(planos))
        cl = list(modelo["lr2"].classes_)
        return P[:, cl.index("positivo")] - P[:, cl.index("negativo")], 1 - P[:, cl.index("nada")]

    def _ajustar_puntuador(self, segs, y):
        y = etiquetas_texto(y)  # dentro del stack la y llega 0/1/2
        f0, e0 = [], []
        for (ss, _), l in zip(segs, y):
            if l == "neutral":
                f0 += [s[0] for s in ss]
                e0 += ["nada"] * len(ss)
            else:
                for s in ss:
                    if s[1] and GRUPO_CONECTOR[s[1][-1]] in self.semillas:
                        f0.append(s[0])
                        e0.append(l)
        modelo = self._entrenar_seg(f0, e0)
        planos = [s[0] for ss, _ in segs for s in ss]
        for _ in range(self.iter_auto):
            pol, op = self._probas(modelo, planos)
            f2, e2 = list(f0), list(e0)
            k = 0
            for (ss, _), l in zip(segs, y):
                n = len(ss)
                p, o = pol[k:k + n], op[k:k + n]
                k += n
                if l == "neutral":
                    continue
                signo = 1 if l == "positivo" else -1
                opuesto = "negativo" if l == "positivo" else "positivo"
                j_con = [j for j, s in enumerate(ss) if s[1] and GRUPO_CONECTOR[s[1][-1]] in self.semillas]
                ultimo_con = j_con[-1] if j_con else -1
                for j, s in enumerate(ss):
                    if j in j_con or s[3]:
                        continue
                    if ultimo_con > 0 and j == ultimo_con - 1 and o[j] > 0.3 and p[j] * signo <= 0.3:
                        f2.append(s[0]); e2.append(opuesto)  # opinión justo antes del contraste
                    elif o[j] > self.umbral_op and abs(p[j]) > self.umbral_pol:
                        f2.append(s[0]); e2.append("positivo" if p[j] > 0 else "negativo")
                    elif o[j] < 0.1:
                        f2.append(s[0]); e2.append("nada")
            modelo = self._entrenar_seg(f2, e2)
        return modelo

    def _puntuar(self, modelo, segs):
        """Por reseña, un array (n_segmentos, 2): [p_pos - p_neg, p_opinión]."""
        planos = [s[0] for ss, _ in segs for s in ss]
        pol, op = self._probas(modelo, planos)
        M = np.column_stack([pol, op])
        out, k = [], 0
        for ss, _ in segs:
            out.append(M[k:k + len(ss)])
            k += len(ss)
        return out

    def _texto(self, segs, puntajes):
        salida = []
        for (ss, _), S in zip(segs, puntajes):
            pol, op = S[:, 0], S[:, 1]
            es_op = [(op[j] > 0.5 and not ss[j][3]) for j in range(len(ss))]
            idx = [j for j in range(len(ss)) if es_op[j]]
            toks = []
            for j, s in enumerate(ss):
                if s[3]:
                    tag = "C"
                elif es_op[j]:
                    tag = "P" if pol[j] > 0 else "N"
                else:
                    tag = "O"
                rol = ("ult" if idx and j == idx[-1] else "") + ("pri" if idx and j == idx[0] else "")
                grupo = GRUPO_CONECTOR[s[1][-1]] if s[1] else "ninguno"
                palabras = re.findall(r"\w\w+", s[0])
                toks += [f"{tag}_{w}" for w in palabras]
                if tag in "PN":
                    toks += [f"{tag}{rol}_{w}" for w in palabras]
                    toks.append(f"{tag}{rol}_con_{grupo}")
            salida.append(" ".join(toks))
        return salida

    def fit(self, X, y):
        self.modelo_ = self._ajustar_puntuador(preparar(X), y)
        return self

    def fit_transform(self, X, y=None, **kw):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        segs = preparar(X)
        puntajes = [None] * len(X)
        skf = StratifiedKFold(self.n_internos, shuffle=True, random_state=SEED)
        for tr, te in skf.split(X, y):
            m = self._ajustar_puntuador([segs[i] for i in tr], y[tr])
            for i, p in zip(te, self._puntuar(m, [segs[i] for i in te])):
                puntajes[i] = p
        self.modelo_ = self._ajustar_puntuador(segs, y)
        return self._texto(segs, puntajes)

    def transform(self, X):
        segs = preparar(X)
        return self._texto(segs, self._puntuar(self.modelo_, segs))


def polaridad_segmentos(C=3):
    """Modelo base: texto marcado con la polaridad de cada segmento -> TF-IDF -> LR."""
    return Pipeline([("est", PolaridadSegmentos()),
                     ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, token_pattern=r"\S+")),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


def modelos_base_final2(con_sr=True, con_polaridad=True):
    """Los 15 modelos base: los 13 del final de la ronda 1 + tras_conector_sr + polaridad_segmentos."""
    extra = [("tras_conector_sr", tras_conector_sr())] if con_sr else []
    if con_polaridad:
        extra.append(("polaridad_segmentos", polaridad_segmentos()))
    return modelos_base_nb() + extra


# ----------------------------------------------------------------------------------------------
# 8. Modelo final 2: stacking de 15 modelos base + metamodelo HistGradientBoosting
# ----------------------------------------------------------------------------------------------
def construir_modelo(con_sr=True, con_polaridad=True):
    """Estimador SIN entrenar. El StackingClassifier entrena el metamodelo con predicciones fuera de pliegue
    (CV interno de 5 pliegues) de los modelos base, así que todo lo aprendido ocurre dentro de fit().
    Por defecto, las 15 bases; con_sr / con_polaridad = False reproduce las alternativas medidas (ver INFORME)."""
    return StackingClassifier(
        modelos_base_final2(con_sr, con_polaridad),
        final_estimator=HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, early_stopping=False,
                                                       random_state=SEED),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    )


# ----------------------------------------------------------------------------------------------
# INFORME (ronda 2, combinación). Cifras medidas con la caché exacta de predicciones de las bases (validada:
# reproduce evaluar() del StackingClassifier; en la semilla 42 las predicciones OOF son idénticas).
# Errores (semillas externas 42 / 2026 / 7) frente al final de la ronda 1 en los mismos pliegues:
#   CV interno 42 (protocolo)            final r1 871/866/834 = 2571
#     A  + tras_conector_sr              837/845/855 = 2537 (-34, gana 2/3)
#     F  + polaridad_segmentos           835/836/842 = 2513 (-58, gana 2/3)
#     B  + las dos (ESTE MODELO)         842/847/841 = 2530 (-41, gana 2/3)
#     + ultima_opinion_fix (r2_faciles) en reemplazo o añadida: no aporta (C -2 1/3, D -30, E -66, G -39).
#   CV interno 1 (chequeo)               final 880/910/888 = 2678
#     A -47 (3/3)   F -58 (3/3)   B 845/878/860 = 2585 (-93, 3/3)
#   Errores fuera del "pool" de mixtas aleatorias (categorías dura + indec_otro de analisis_final.pkl, ~14% de
#   X_ent, donde la etiqueta es ~moneda al aire): criterio de menor varianza.
#     CV interno 42: final 235/242/218 = 695 | A 667 | B 203/214/204 = 621 (-74, 3/3)
#     CV interno 1:  final 249/252/233 = 734 | A 673 | B 209/216/194 = 619 (-115, 3/3)
#   La ganancia está en reseñas de una sola opinión (una_pol): 107/109/98 -> 86/82/79 errores.
# Las diferencias entre A, F y B en el total (+-20-50 errores) están dentro del ruido; B se elige porque es la
# mejor en el total de las 6 CV (-134, gana 5/6) y mejora fuera del pool en 6/6 (también frente a A: 6/6).
# Alternativas: construir_modelo(con_polaridad=False) = A; construir_modelo(con_sr=False) = F.
