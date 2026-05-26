"""
Runtime compatibility shims for third-party packages used by VBench.

This module is auto-imported by Python (via site.py) when present on PYTHONPATH.
"""

import numpy as np


def _ensure_numpy_sctypes() -> None:
    # imgaug < 0.4 expects np.sctypes, removed in NumPy 2.0.
    if hasattr(np, "sctypes"):
        return

    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.str_, np.bytes_],
    }


def _ensure_pkg_resources_packaging_version() -> None:
    # openai/CLIP imports `packaging` through pkg_resources and expects the
    # `version` submodule to be attached as an attribute.
    try:
        from pkg_resources import packaging as pkg_resources_packaging
        import packaging.version as packaging_version
    except Exception:
        return

    if not hasattr(pkg_resources_packaging, "version"):
        pkg_resources_packaging.version = packaging_version


_ensure_numpy_sctypes()
_ensure_pkg_resources_packaging_version()
