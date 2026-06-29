import os
from torch.utils.cpp_extension import load

_DIR = os.path.dirname(__file__)
_ALL = [
    ("add", "add.cu"),
]

def _load(op):
    load(
        name=f"super_ops_{op[0]}",
        sources=[os.path.join(_DIR, op[1])],
        extra_cuda_cflags=["--use_fast_math"],
        verbose=False,
    )

def load_all():
    for op in _ALL:
        _load(op)