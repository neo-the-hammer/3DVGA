"""The three PointNet++ implementations, and the chain that picks one.

Same maths, three trade-offs. They are kept side by side because the CUDA
extension cannot be built everywhere and the model must still run:

    pointnet2_ops_lib/    C++/CUDA extension. Fastest, needs a matching toolchain
                          and a GPU whose compute capability the build targets.
    pointnet2_py_adv/     pure PyTorch, optimized (top-k ball query instead of a
                          full sort, channel-first gathers, no redundant
                          transposes). Runs on CPU or GPU. The default fallback.
    pointnet2_python/     the original pure-Python reference. Slowest; kept
                          because it is the definition the other two match.

Resolution order, each tier tried only when the one above it cannot be imported:

    1. pointnet2_ops.pointnet2_modules
    2. pointnet.pointnet2_py_adv.pn_py_torch
    3. pointnet.pointnet2_python.pointnet2_modules

``resolve()`` walks that chain and returns the module that answered, so a
missing CUDA extension degrades to optimized torch, and a broken optimized
implementation degrades to the reference rather than taking the model down.

``ABLATION.POINTNET_IMPL`` pins one tier by name for A/B comparison. Pinning a
tier that cannot be imported is an error, not a silent downgrade -- an ablation
that quietly measured a different implementation than the one it names would be
worse than a crash.

Outputs are equivalent across all three: ball_query / gather / FPS indices are
bitwise identical for the same seed, features are allclose. The only intentional
non-guarantee is neighbour ordering among exactly-equidistant points.
"""

import importlib
import os
import sys

#: Tier name -> (module path, note). Order is the resolution order.
IMPLEMENTATIONS = (
    ("pointnet2_ops", "pointnet2_ops.pointnet2_modules",
     "CUDA extension"),
    ("pointnet2_py_adv", "pointnet.pointnet2_py_adv.pn_py_torch",
     "optimized pure PyTorch"),
    ("pointnet2_python", "pointnet.pointnet2_python.pointnet2_modules",
     "original pure-Python reference"),
)

#: Back-compatible aliases for ABLATION.POINTNET_IMPL values written before the
#: implementations moved under pointnet/.
ALIASES = {
    "pointnet_py_adv": "pointnet2_py_adv",
    "pn": "pointnet2_py_adv",
    "pn_py_torch": "pointnet2_py_adv",
    "cuda": "pointnet2_ops",
    "ops": "pointnet2_ops",
    "python": "pointnet2_python",
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_path():
    """pointnet2_ops is installed as a top-level package by pip; the two Python
    implementations are plain directories, so the repo root has to be importable
    for ``pointnet.*`` to resolve when the cwd is elsewhere."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def resolve(preferred=None, names=None):
    """Return (module, tier_name) for the first implementation that imports.

    `preferred` pins one tier by name (or alias). A pinned tier that fails to
    import raises ImportError -- silently running a different implementation
    than the one an experiment names would invalidate the experiment.
    """
    _ensure_path()

    if preferred:
        wanted = ALIASES.get(preferred, preferred)
        for tier, path, note in IMPLEMENTATIONS:
            if tier != wanted:
                continue
            try:
                return importlib.import_module(path), tier
            except ImportError as error:
                raise ImportError(
                    f"ABLATION.POINTNET_IMPL pins {preferred!r} ({note}), which "
                    f"cannot be imported: {error}. Either install it or choose "
                    f"another of {[t for t, _, _ in IMPLEMENTATIONS]}."
                ) from error
        raise ImportError(
            f"unknown PointNet++ implementation {preferred!r}. "
            f"Choose one of {[t for t, _, _ in IMPLEMENTATIONS]} "
            f"(aliases: {sorted(ALIASES)})."
        )

    failures = []
    for tier, path, note in IMPLEMENTATIONS:
        try:
            return importlib.import_module(path), tier
        except ImportError as error:
            failures.append(f"  {tier:18s} ({note}): {error}")

    raise ImportError(
        "no PointNet++ implementation could be imported:\n" + "\n".join(failures))


def load(*names, preferred=None):
    """Import `names` from the first working implementation.

    Returns (objects_tuple, tier_name). Used by models/backbone_module.py and
    models/proposal_module.py so both resolve through one code path.
    """
    module, tier = resolve(preferred)
    missing = [n for n in names if not hasattr(module, n)]
    if missing:
        raise ImportError(
            f"PointNet++ implementation {tier!r} does not provide {missing}.")
    return tuple(getattr(module, n) for n in names), tier


def available():
    """[(tier, note, importable, detail)] for every tier. Imports nothing that
    is already imported; used by diagnostics and the environment report."""
    _ensure_path()
    rows = []
    for tier, path, note in IMPLEMENTATIONS:
        try:
            importlib.import_module(path)
            rows.append((tier, note, True, "ok"))
        except ImportError as error:
            rows.append((tier, note, False, str(error).splitlines()[0][:110]))
    return rows
