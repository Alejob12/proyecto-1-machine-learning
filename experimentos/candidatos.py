"""Candidatos probados en la sesión local (módulo importable para que joblib los envíe a los procesos de trabajo).

SinIndecisas: entrena un modelo base sin las reseñas con cierre indeciso (etiqueta positivo/negativo ~aleatoria).
  Resultado (semilla 42, errores del stack; "fuera de la zona" es la métrica que importa, ver zona.py):
      iter13 (15 bases)                       835 = 788 en la zona + 47 fuera
      A: 14 bases sin indecisas + cierre      870 = 820 en la zona + 50 fuera   -> no ayuda
      B: las 15 originales + las 14 nuevas    818 = 777 en la zona + 41 fuera   -> -6 fuera (falta el protocolo)
  Cada base por separado sí mejora fuera de las indecisas (tras_conector -23, ultima_opinion -65,
  tras_conector_sr -60), pero el stack ya corregía esos errores combinando vistas.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import modelo_final2 as m


def sin_cierre(textos):
    """True para las reseñas SIN frase de cierre indeciso (lista estructural CIERRE del modelo)."""
    return np.array([not m.CIERRE.search(m.corregir_texto(t)) for t in textos])


class SinIndecisas(ClassifierMixin, BaseEstimator):
    """Entrena el estimador sin las reseñas con cierre indeciso; para predecir usa todas las reseñas tal cual."""

    def __init__(self, estimador):
        self.estimador = estimador

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        ok = sin_cierre(X)
        self.estimador_ = clone(self.estimador).fit(X[ok], y[ok])
        self.classes_ = self.estimador_.classes_
        return self

    def predict_proba(self, X):
        return self.estimador_.predict_proba(np.asarray(X, dtype=object))

    def predict(self, X):
        return self.estimador_.predict(np.asarray(X, dtype=object))
