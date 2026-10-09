"""Línea base local y candidatos, separando los errores EN la zona de moneda al aire y FUERA de ella (zona.py).

Dentro de la zona (~17,8 % de X_ent) cualquier modelo acierta ~50 %: lo que un modelo gana o pierde ahí es suerte.
La métrica para decidir es el número de errores FUERA de la zona.

Uso, desde la raíz del repo (la primera vez calcula la caché de cada semilla, ~3-4 min por variante con 10 núcleos):
    python experimentos/evaluar_por_zona.py 42 2026 7            # solo la iteración 13
    python experimentos/evaluar_por_zona.py 42 --sinind          # + bases entrenadas sin reseñas indecisas
"""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cachef import *  # noqa: E402,F401
import modelo_final2 as m  # noqa: E402
from candidatos import SinIndecisas  # noqa: E402
from zona import zona  # noqa: E402

N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 4))
BASES = [n for n, _ in m.modelos_base_final2()]
ZONA = zona(XA)


def meta():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, early_stopping=False, random_state=SEED)


def fila(nombre, r, ref=None):
    e = r["pred"] != Y
    txt = (f"{nombre:<42} acc {r['acc']:.4f} ± {r['std']:.4f} | errores {int(e.sum()):>4} = {int(e[ZONA].sum()):>4} en la zona "
           f"(acierto {1 - e[ZONA].mean():.3f}) + {int(e[~ZONA].sum()):>3} fuera")
    if ref is not None:
        er = ref["pred"] != Y
        arregla, rompe = int((er & ~e & ~ZONA).sum()), int((~er & e & ~ZONA).sum())
        txt += f" | fuera de la zona: arregla {arregla}, rompe {rompe}"
    print(txt, flush=True)
    return int(e[~ZONA].sum())


if __name__ == "__main__":
    args = sys.argv[1:]
    con_sinind = "--sinind" in args
    semillas = [int(a) for a in args if not a.startswith("--")] or [SEED]
    print(f"zona de moneda al aire: {int(ZONA.sum())} de {len(ZONA)} reseñas de X_ent ({ZONA.mean():.3f})")
    total = {}
    for semilla in semillas:
        for nombre, est in m.modelos_base_final2():
            cachear(f"i13__{nombre}", est, n_jobs=N_JOBS, semilla=semilla, usar_ens=False)
            if con_sinind and nombre != "cierre":
                cachear(f"sinind__{nombre}", SinIndecisas(est), n_jobs=N_JOBS, semilla=semilla, usar_ens=False)
        i13 = [f"i13__{b}" for b in BASES]
        sinind = [f"sinind__{b}" for b in BASES if b != "cierre"]
        c = cargar(i13 + (sinind if con_sinind else []), semilla=semilla)
        print(f"\n===== CV externo con semilla {semilla} =====")
        base = evaluar_meta(c, i13, meta=meta(), verbose=False, semilla=semilla)
        total.setdefault("iter13", []).append(fila("iter13 (15 bases)", base))
        if con_sinind:
            ra = evaluar_meta(c, sinind + ["i13__cierre"], meta=meta(), verbose=False, semilla=semilla)
            total.setdefault("A", []).append(fila("A: 14 bases sin indecisas + cierre", ra, base))
            rb = evaluar_meta(c, i13 + sinind, meta=meta(), verbose=False, semilla=semilla)
            total.setdefault("B", []).append(fila("B: 15 originales + 14 sin indecisas", rb, base))
    if len(semillas) > 1:
        print("\nerrores FUERA de la zona por semilla y total:", {k: (v, sum(v)) for k, v in total.items()})
