import ctypes
import os
import sys
import numpy as np

# ---------------- locate C library ----------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_C_DIR = os.path.join(_HERE, "_c")

if sys.platform == "darwin":
    _LIBNAME = "libtree.dylib"
elif sys.platform == "win32":
    _LIBNAME = "tree.dll"
else:
    _LIBNAME = "libtree.so"

_LIB_PATH = os.path.join(_C_DIR, _LIBNAME)

if not os.path.exists(_LIB_PATH):
    raise ImportError(
        f"SmallGBM C library not found at {_LIB_PATH}.\n"
        f"Build it manually:\n"
        f"  cc -O3 -fPIC -shared smallgbm/_c/tree.c -o {_LIB_PATH} -lm\n"
        f"or reinstall: pip install --force-reinstall smallgbm"
    )

_lib = ctypes.CDLL(_LIB_PATH)


# ---------------- ctypes signatures ----------------

class _Predictions(ctypes.Structure):
    _fields_ = [
        ("preds",         ctypes.POINTER(ctypes.c_double)),
        ("uncertainties", ctypes.POINTER(ctypes.c_double)),
        ("n",             ctypes.c_int),
    ]


_lib.tree_create.restype  = ctypes.c_void_p
_lib.tree_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
]

_lib.tree_free.restype  = None
_lib.tree_free.argtypes = [ctypes.c_void_p]

_lib.tree_fit.restype  = None
_lib.tree_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int,
]

_lib.tree_predict.restype  = ctypes.POINTER(_Predictions)
_lib.tree_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int,
]

_lib.predictions_free.restype  = None
_lib.predictions_free.argtypes = [ctypes.POINTER(_Predictions)]


def _dptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# ---------------- wrapper ----------------

class RobustDecisionTree:
    def __init__(self, max_depth=2, min_samples_leaf=5, sigma_prior=1.0,
                 n_splits=10, colsample_bytree=1.0):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sigma_prior      = sigma_prior
        self.n_splits         = n_splits
        self.colsample_bytree = colsample_bytree
        self._handle          = None
        self._n_features      = None

    def fit(self, X, residuals):
        X = np.ascontiguousarray(X, dtype=np.float64)
        residuals = np.ascontiguousarray(residuals, dtype=np.float64).ravel()

        n, m = X.shape
        self._n_features = m

        if self._handle is not None:
            _lib.tree_free(self._handle)
            self._handle = None

        self._handle = _lib.tree_create(
            self.max_depth,
            self.min_samples_leaf,
            self.sigma_prior,
            self.n_splits,
            self.colsample_bytree,
        )
        if not self._handle:
            raise MemoryError("tree_create returned NULL")

        _lib.tree_fit(self._handle, _dptr(X), _dptr(residuals), n, m)
        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        n, m = X.shape

        p = _lib.tree_predict(self._handle, _dptr(X), n, m)
        if not p:
            raise RuntimeError("tree_predict returned NULL")

        # zero-copy view + copy, чтобы после free данные остались
        out = np.ctypeslib.as_array(p.contents.preds, shape=(n,)).copy()
        _lib.predictions_free(p)
        return out

    def predict_with_uncertainty(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        n, m = X.shape

        p = _lib.tree_predict(self._handle, _dptr(X), n, m)
        if not p:
            raise RuntimeError("tree_predict returned NULL")

        preds = np.ctypeslib.as_array(p.contents.preds, shape=(n,)).copy()
        uncs  = np.ctypeslib.as_array(p.contents.uncertainties, shape=(n,)).copy()
        _lib.predictions_free(p)
        return preds, uncs

    def __del__(self):
        try:
            if self._handle is not None:
                _lib.tree_free(self._handle)
                self._handle = None
        except Exception:
            pass