"""SOLO PARA MEDIR. NUNCA importar desde un modelo: la lista de frases de abajo es un léxico de sentimiento
escrito a mano, que la Parte 1 no permite dentro del modelo (sí para analizar).

Hallazgo (sesión local, 2026-10-09): las reseñas expresan la opinión de dos maneras.
  Familia A: adjetivo tras verbo copulativo ("resultó frágil", "luce impecable", "terminó siendo pésima").
             Con dos opiniones opuestas gana SIEMPRE la última, con cualquier conector: el stack acierta ~99,8 %.
  Familia B: unas 40 frases hechas (mitad positivas, mitad negativas: "me encantó ... de principio a fin", "no vale
             lo que cuesta", "cada peso que pagué ... valió la pena"). Con UNA sola frase la etiqueta es su polaridad.
             Con una positiva y una negativa la etiqueta es moneda al aire (positivo 50,4 %; gana la última 51,5 %),
             lleve o no frase de cierre indeciso y con cualquier conector. Ver analisis_zona.py.
La "zona" = reseñas con al menos una frase hecha positiva y una negativa, o con cierre indeciso (el cierre solo
aparece en reseñas mixtas de la familia B). En X_ent son 1.708 de 9.600 (17,8 %); en eval.csv, 564 de 3.000 (18,8 %).
"""
import re

import numpy as np

import modelo_final2 as m

POS = ["encanto", "supero todas mis expectativas", "funciona de maravilla", "todo un acierto",
       "justo lo que estaba buscando", "vale totalmente la pena", "valio la pena", "la vida mas facil",
       "sorprendio para bien", "tal como esperaba", "cumple de sobra", "por encima de l", "quede encantado",
       "termine feliz", "estoy feliz", "alegra haber", "con los ojos cerrados", "sin pensarlo", "cinco estrellas",
       "ni una sola queja"]
NEG = ["decepciono", "defraudo todas mis expectativas", "funciona bastante mal", "error de compra",
       "para nada lo que esperaba", "no vale lo que cuesta", "tire el dinero", "dolor de cabeza", "de tener problemas",
       "queda muy corto", "por debajo de l", "quede muy molesto", "termine harto", "arrepentido de haber",
       "arrepiento de haber", "recomiendo .{0,60}?a nadie", "ni regalado", "una estrella", "antes de lo que esperaba"]
PAT_POS = re.compile("|".join(POS))
PAT_NEG = re.compile("|".join(NEG))
# Familia A: adjetivo tras verbo copulativo (sirve para describir, no para decidir)
COP = (r"(?:me resulto ser|resulto ser|me resulto|resulto|me salio|salio|quedo|luce|se sintio|se siente|se ve|"
       r"se mantuvo|termino siendo|me parecio|parecio)")
PAT_A = re.compile(r"\b" + COP + r"\s+(?:(?:muy|bastante|poco|algo|medio|bien|mal|mas|menos|demasiado|super|un poco|nada)"
                   r"\s+)*\w{4,}")


def describir(textos):
    """Por reseña: [nº de frases hechas positivas, nº de negativas, tiene cierre indeciso, nº de adjetivos copulativos]."""
    filas = []
    for t in textos:
        tc = m.corregir_texto(t)
        filas.append((len(PAT_POS.findall(tc)), len(PAT_NEG.findall(tc)), int(bool(m.CIERRE.search(tc))),
                      len(PAT_A.findall(tc))))
    return np.array(filas)


def zona(textos):
    """True si la reseña está en la zona de moneda al aire (frase hecha positiva + negativa, o cierre indeciso)."""
    d = describir(textos)
    return ((d[:, 0] > 0) & (d[:, 1] > 0)) | (d[:, 2] > 0)
