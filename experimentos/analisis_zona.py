"""ANÁLISIS (no es parte de ningún modelo): ¿hay alguna regla en las reseñas con una frase hecha positiva y una
negativa? Respuesta: no. Ninguna variable pasa de 0,52 (azar 0,500 ± 0,012) en validación cruzada 10x5.
Usa la lista escrita a mano de zona.py SOLO para medir. Solo X_ent (nunca X_prueba ni eval.csv).

Ejecutar desde la raíz del repo:  python experimentos/analisis_zona.py     (~1 min)

Resultado medido el 2026-10-09 (1.608 reseñas polares con exactamente una frase positiva y una negativa):
  etiqueta positiva 0,504 | gana la última 0,515 (con cierre 0,506, n=1.100; sin cierre 0,533, n=508)
  frase positiva 0,518 | frase negativa 0,492 | ambas 0,508 | con interacción 0,492
  frases + orden + último conector + frase de cierre 0,504 | HGB (frases, orden, nº oraciones, distancia) 0,493
  palabras marcadas con la polaridad de su oración (capta el aspecto opinado) 0,506-0,517
  bolsa de palabras normal 0,503 | n-gramas de caracteres 0,511
  'gana la última' según la frase final: chi2 = 46,6 con 39 grupos (sin efecto se espera ~39): nada.
"""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *  # noqa: E402,F401
import modelo_final2 as m  # noqa: E402
import zona as z  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

PATS = ([("P", i, re.compile(p)) for i, p in enumerate(z.POS)]
        + [("N", i, re.compile(p)) for i, p in enumerate(z.NEG)])

XA = X_ent.reset_index(drop=True)
YA = y_ent.reset_index(drop=True).values
TC = [corregir_texto(t) for t in XA]

# Frases hechas de cada reseña, en orden de aparición: [(posición, polaridad, índice de frase)]
filas = []
for t in TC:
    hallazgos = sorted((mm.start(), pol, k) for pol, k, pat in PATS for mm in pat.finditer(t))
    vistos, limpio = set(), []
    for pos_, pol, k in hallazgos:
        if (pol, k) not in vistos:
            vistos.add((pol, k))
            limpio.append((pos_, pol, k))
    filas.append(limpio)
sig = ["".join(p for _, p, _ in f) for f in filas]
print("firmas más comunes (frases hechas por reseña, en orden):")
print(pd.Series(sig).value_counts().head(8).to_string())

idx = np.array([i for i in range(len(TC)) if sig[i] in ("PN", "NP") and YA[i] != "neutral"])
y = (YA[idx] == "positivo").astype(int)
orden_pn = np.array([sig[i] == "PN" for i in idx])
ult_gana = np.where(orden_pn, y == 0, y == 1)
ind = INDECISA_ENT[idx]
print(f"\nreseñas polares con exactamente una frase positiva y una negativa: {len(idx)}")
print(f"  etiqueta positiva {y.mean():.3f} | gana la última {ult_gana.mean():.3f} | con cierre {ult_gana[ind].mean():.3f} "
      f"(n={ind.sum()}) | sin cierre {ult_gana[~ind].mean():.3f} (n={(~ind).sum()})")

kp = np.array([[k for _, p, k in filas[i] if p == "P"][0] for i in idx])
kn = np.array([[k for _, p, k in filas[i] if p == "N"][0] for i in idx])


def ultimo_conector(t):
    cs = list(m.PATRON_CONECTOR.finditer(t))
    return cs[-1].group() if cs else "(ninguno)"


con = np.array([ultimo_conector(TC[i]) for i in idx])
cierre = np.array([(m.CIERRE.search(TC[i]).group() if m.CIERRE.search(TC[i]) else "-") for i in idx])
n_or = np.array([len(oraciones(TC[i])) for i in idx])
dist = np.array([abs(filas[i][0][0] - filas[i][1][0]) for i in idx])
rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
ee = np.sqrt(0.25 / len(y))


def cv(X, nombre, modelo=None):
    modelo = modelo or LogisticRegression(C=1.0, max_iter=3000)
    a = cross_val_score(modelo, X, y, cv=rcv, scoring="accuracy").mean()
    print(f"  {nombre:<70} acc {a:.3f}   (azar 0.500 ± {ee:.3f})")


oh = OneHotEncoder(handle_unknown="ignore")
print("\nvalidación cruzada 10x5 SOLO con estas reseñas: ¿se puede predecir positivo/negativo?")
cv(oh.fit_transform(np.c_[kp]), "frase positiva (20 valores)")
cv(oh.fit_transform(np.c_[kn]), "frase negativa (20 valores)")
cv(oh.fit_transform(np.c_[kp, kn]), "frase positiva + frase negativa (aditivo)")
cv(oh.fit_transform(np.c_[kp, kn, np.array([f"{a}_{b}" for a, b in zip(kp, kn)])]), "frase positiva x negativa (interacción)")
cv(oh.fit_transform(np.c_[kp, kn, orden_pn, con, cierre]), "frases + orden + último conector + frase de cierre")
cv(np.c_[kp, kn, orden_pn, n_or, dist], "HGB: frases, orden, nº de oraciones, distancia entre frases",
   HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=0))

# Bolsa de palabras con cada palabra marcada por la polaridad de la frase hecha de SU oración (capta el aspecto)
docs = []
for i in idx:
    toks, pos_ = [], 0
    for o in oraciones(TC[i]):
        ini = TC[i].find(o, pos_)
        pos_ = ini + len(o)
        pols = {p for q, p, _ in filas[i] if ini <= q < pos_}
        tag = "PN" if len(pols) == 2 else (pols.pop() if pols else "O")
        toks += [f"{tag}_{w}" for w in re.findall(r"\w\w+", o)]
    docs.append(" ".join(toks))
Xb = TfidfVectorizer(token_pattern=r"\S+", min_df=3, ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
for C in [0.1, 1.0]:
    cv(Xb, f"palabras marcadas con la polaridad de su oración (uni+bigramas), C={C}", LogisticRegression(C=C, max_iter=3000))
textos = [TC[i] for i in idx]
cv(TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True).fit_transform(textos), "bolsa de palabras normal (uni+bigramas)")
cv(TfidfVectorizer(min_df=3, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit_transform(textos),
   "n-gramas de caracteres")

t = pd.DataFrame({"pol": np.where(orden_pn, "N", "P"), "k": np.where(orden_pn, kn, kp), "gana": ult_gana})
g = t.groupby(["pol", "k"])["gana"].agg(["size", "mean"])
g["z"] = (g["mean"] - 0.5) / np.sqrt(0.25 / g["size"])
print(f"\n'gana la última' según la frase que va al final: z entre {g['z'].min():.2f} y {g['z'].max():.2f}; "
      f"chi2 = {(g['z'] ** 2).sum():.1f} con {len(g)} grupos (sin efecto se espera ~{len(g)})")
