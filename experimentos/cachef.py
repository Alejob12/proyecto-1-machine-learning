"""Caché exacta de predicciones de modelos base (misma lógica que ensamble/cache_lib.py, validada allí
contra evaluar(StackingClassifier)). Por cada pliegue externo k del CV del harness guarda:
  inner[k]: OOF del CV interno (StratifiedKFold 5, SEED) sobre el train externo -> lo que ve el metamodelo al entrenar
  test[k]:  predicción del modelo base entrenado en todo el train externo k sobre el test externo k
Permite evaluar subconjuntos de bases y metamodelos en segundos. Lee (sin modificar) la caché del explorador
'ensamble' para las bases de iter10, y escribe las nuevas en final/cache.
Permite un CV externo alternativo (semilla) para la prueba de robustez.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness import *  # noqa
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

H2 = Path(__file__).resolve().parent
CACHE = H2 / 'cache'                  # se crea sola; NO está en git (≈300 MB). Se regenera con cachear().
CACHE_ENS = H2 / 'cache_base'          # opcional (solo lectura); si no existe, se ignora
CACHE.mkdir(exist_ok=True)
ENC = LabelEncoder().fit(y_ent)
Y = ENC.transform(y_ent)
XA = X_ent.reset_index(drop=True)


def _splits(semilla):
    cv = StratifiedKFold(5, shuffle=True, random_state=semilla)
    outer = list(cv.split(X_ent, y_ent))
    inner = []
    for tr, te in outer:
        cvi = StratifiedKFold(5, shuffle=True, random_state=SEED)
        inner.append(list(cvi.split(XA.iloc[tr], Y[tr])))
    return outer, inner


SPLITS = {SEED: _splits(SEED)}


def splits(semilla=SEED):
    if semilla not in SPLITS:
        SPLITS[semilla] = _splits(semilla)
    return SPLITS[semilla]


def _salida(est, X):
    if hasattr(est, 'predict_proba'):
        return est.predict_proba(X)
    d = est.decision_function(X)
    return d if d.ndim == 2 else d.reshape(-1, 1)


def _tarea(factory, k, j, semilla):
    outer, inner = splits(semilla)
    tr, te = outer[k]
    Xtr, ytr = XA.iloc[tr], Y[tr]
    est = clone(factory)
    if j is None:
        est.fit(Xtr, ytr)
        return k, j, _salida(est, XA.iloc[te])
    itr, ite = inner[k][j]
    est.fit(Xtr.iloc[itr], ytr[itr])
    return k, j, _salida(est, Xtr.iloc[ite])


def _archivo(nombre, semilla):
    suf = '' if semilla == SEED else f'__s{semilla}'
    return CACHE / f'{nombre}{suf}.joblib'


def cachear(nombre, estimador, n_jobs=4, forzar=False, semilla=SEED, usar_ens=True):
    f = _archivo(nombre, semilla)
    if f.exists() and not forzar:
        return joblib.load(f)
    if usar_ens and semilla == SEED and (CACHE_ENS / f'{nombre}.joblib').exists() and not forzar:
        return joblib.load(CACHE_ENS / f'{nombre}.joblib')
    t0 = time.perf_counter()
    outer, inner_s = splits(semilla)
    tareas = [(k, j) for k in range(5) for j in list(range(5)) + [None]]
    res = Parallel(n_jobs=n_jobs)(delayed(_tarea)(estimador, k, j, semilla) for k, j in tareas)
    inner, test = [None] * 5, [None] * 5
    for k, j, p in res:
        tr, te = outer[k]
        if j is None:
            test[k] = p
        else:
            if inner[k] is None:
                inner[k] = np.zeros((len(tr), p.shape[1]))
            inner[k][inner_s[k][j][1]] = p
    d = {'inner': inner, 'test': test, 'seg': time.perf_counter() - t0}
    joblib.dump(d, f)
    print(f'cache {nombre} (semilla {semilla}): {d["seg"]:.0f}s', flush=True)
    return d


def cargar(nombres, semilla=SEED):
    out = {}
    for n in nombres:
        f = _archivo(n, semilla)
        if not f.exists() and semilla == SEED:
            f = CACHE_ENS / f'{n}.joblib'
        out[n] = joblib.load(f)
    return out


INDEC = INDECISA_ENT
POLAR = (y_ent.values != 'neutral')


def hgb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED)


def evaluar_meta(cache, nombres, meta=None, verbose=True, etiqueta='', semilla=SEED):
    meta = meta if meta is not None else hgb()
    outer, _ = splits(semilla)
    accs, pred = [], np.empty(len(Y), dtype=int)
    for k, (tr, te) in enumerate(outer):
        A = np.hstack([cache[n]['inner'][k] for n in nombres])
        B = np.hstack([cache[n]['test'][k] for n in nombres])
        m = clone(meta).fit(A, Y[tr])
        p = m.predict(B)
        pred[te] = p
        accs.append((p == Y[te]).mean())
    accs = np.array(accs)
    ok = pred == Y
    out = dict(acc=accs.mean(), std=accs.std(), folds=np.round(accs, 4).tolist(),
               ind=ok[INDEC & POLAR].mean(), noind=ok[~INDEC & POLAR].mean(), neu=ok[~POLAR].mean(), pred=pred,
               err=int((~ok).sum()))
    if verbose:
        print(f'{etiqueta:<55} acc {out["acc"]:.4f} ± {out["std"]:.4f} | err {out["err"]} | ind {out["ind"]:.3f} | '
              f'noind {out["noind"]:.3f} | neu {out["neu"]:.3f} | folds {out["folds"]}', flush=True)
    return out


def pareado(a, b):
    """Compara dos resultados de evaluar_meta: arregla / rompe y McNemar (chi2 con corrección)."""
    oka, okb = a['pred'] == Y, b['pred'] == Y
    n01, n10 = int((okb & ~oka).sum()), int((oka & ~okb).sum())
    chi = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    from scipy.stats import chi2 as _c
    return n01, n10, chi, 1 - _c.cdf(chi, 1)
