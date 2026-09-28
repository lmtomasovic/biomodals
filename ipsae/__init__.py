"""ipSAE scoring for predicted complexes.

Vendored from https://github.com/DunbrackLab/IPSAE (MIT, commit 6174cf9,
2026-01-03). `ipsae.py` and `af2rechain.py` are unmodified upstream scripts;
`run.py` is the batch wrapper. See README.md.
"""

__all__ = ["find_pairs", "parse_table", "run_one", "summarise"]


def __getattr__(name):
    if name in __all__:
        from . import run

        return getattr(run, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
