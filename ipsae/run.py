"""Run ipSAE over predicted complexes and collect the scores into one table.

ipSAE (Dunbrack 2025) ranks predicted protein-protein interactions from the PAE
matrix. It separates true from false complexes more reliably than ipTM, which
dilutes interface confidence across disordered or accessory regions.

    python -m ipsae.run out/boltz/ -o ipsae.csv

The upstream `ipsae.py` is a script, not a library: it reads sys.argv and writes
`<stem>_<pae>_<dist>.txt` next to its input. This wraps it, pairs up the PAE and
structure files, parses the tables back and adds `ipsae_min` -- the minimum over
the two asymmetric directions, which is the variant binder benchmarks report.

IMPORTANT: needs a FULL PAE matrix.
  - Boltz: run `modal_boltz.py` with `--write_full_pae` (not in DEFAULT_PARAMS),
    which produces `pae_<name>_model_N.npz` beside `<name>_model_N.cif`.
  - BoltzGen output will NOT work: its npz files carry only reduced scalars
    (`interaction_pae`, `min_interaction_pae`), never the N x N matrix.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).with_name("ipsae.py")

def _pae_candidates(stem: str) -> list[str]:
    """Filenames a predictor might use for the PAE that goes with `stem`.

    AF3 pairs `fold_x_model_0.cif` with `fold_x_full_data_0.json` -- the
    `_model_` segment is replaced, not suffixed, so it needs its own rule.
    """
    cands = [
        f"pae_{stem}.npz",  # Boltz-1 / Boltz-2
        f"{stem}.npz",
        f"{stem}.json",
    ]
    if "_model_" in stem:
        cands.append(stem.replace("_model_", "_full_data_") + ".json")  # AlphaFold3
    cands += [
        f"{stem}_full_data_0.json",
        f"{stem}.json.gz",  # ColabFold AF2
        f"{stem}_scores.json",
    ]
    return cands


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Pair each structure with its PAE file.

    Looks for `.cif`/`.pdb` under `root` and the sibling PAE file each
    predictor writes. Structures with no PAE file are skipped.
    """
    pairs = []
    for struct in sorted([*root.rglob("*.cif"), *root.rglob("*.pdb")]):
        for name in _pae_candidates(struct.stem):
            cand = struct.parent / name
            if cand.exists():
                pairs.append((cand, struct))
                break
    return pairs


def parse_table(txt: Path) -> pd.DataFrame:
    """Read an ipsae.py `.txt` output into a frame."""
    # The file opens with a blank line, so seek the header rather than
    # assuming it is first.
    lines = [ln for ln in txt.read_text().splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    header = lines[0].split()
    rows = [
        dict(zip(header, parts))
        for ln in lines[1:]
        if len(parts := ln.split()) == len(header)
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in df.columns:
        if c not in ("Chn1", "Chn2", "Type", "Model"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def run_one(pae: Path, struct: Path, pae_cutoff: float, dist_cutoff: float) -> pd.DataFrame:
    """Run ipsae.py on one pair and return its parsed table."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(pae), str(struct), str(pae_cutoff), str(dist_cutoff)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ipsae.py failed on {struct.name}: {proc.stderr.strip()[-400:]}")

    cut = f"{pae_cutoff:g}_{dist_cutoff:g}"
    txt = struct.with_name(f"{struct.stem}_{cut}.txt")
    if not txt.exists():
        raise RuntimeError(f"ipsae.py wrote no table for {struct.name} (expected {txt.name})")

    df = parse_table(txt)
    df.insert(0, "structure", str(struct))
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per structure per chain pair, with ipsae_min/max.

    `ipsae_min` is the minimum over the two asymmetric directions -- the
    variant reported as `min ipSAE` in binder benchmarks, and the best single
    Boltz-2 metric (AUROC 0.638) in the ProteinBase calibration.
    """
    asym = df[df.Type == "asym"].copy()
    if asym.empty:
        return pd.DataFrame()
    asym["pair"] = [
        "".join(sorted((a, b))) for a, b in zip(asym.Chn1, asym.Chn2)
    ]
    g = asym.groupby(["structure", "pair"], as_index=False).agg(
        ipsae_min=("ipSAE", "min"),
        ipsae_max=("ipSAE", "max"),
        iptm_af=("ipTM_af", "first"),
        pdockq=("pDockQ", "first"),
        pdockq2=("pDockQ2", "max"),
        lis=("LIS", "max"),
    )
    return g.sort_values("ipsae_min", ascending=False).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ipsae.run",
        description="Run ipSAE over predicted complexes and collect scores.",
    )
    ap.add_argument("root", type=Path, help="directory to search for structure + PAE pairs")
    ap.add_argument("-o", "--out", type=Path, help="write the per-chain-pair summary here")
    ap.add_argument("--full-out", type=Path, help="also write the raw per-direction rows")
    ap.add_argument("--pae-cutoff", type=float, default=10.0)
    ap.add_argument("--dist-cutoff", type=float, default=10.0)
    args = ap.parse_args(argv)

    pairs = find_pairs(args.root)
    if not pairs:
        print(
            f"no structure+PAE pairs under {args.root}.\n"
            "Boltz needs `--write_full_pae`; BoltzGen npz files do not contain a "
            "full PAE matrix and cannot be scored.",
            file=sys.stderr,
        )
        return 1

    frames, failures = [], []
    for pae, struct in pairs:
        try:
            frames.append(run_one(pae, struct, args.pae_cutoff, args.dist_cutoff))
        except Exception as exc:  # noqa: BLE001 - reported below
            failures.append((struct, exc))

    for struct, exc in failures:
        print(f"skipped {struct}: {exc}", file=sys.stderr)
    if not frames:
        return 1

    full = pd.concat(frames, ignore_index=True)
    summary = summarise(full)

    if args.full_out:
        full.to_csv(args.full_out, index=False)
    if args.out:
        summary.to_csv(args.out, index=False)
        print(f"wrote {args.out}  ({len(summary)} rows from {len(frames)} structures)")

    print(f"\n{'ipsae_min':>9}  {'ipsae_max':>9}  {'ipTM':>6}  {'pDockQ':>7}  pair  structure")
    for r in summary.head(25).itertuples():
        print(
            f"{r.ipsae_min:>9.3f}  {r.ipsae_max:>9.3f}  {r.iptm_af:>6.3f}  "
            f"{r.pdockq:>7.3f}  {r.pair:>4}  {Path(r.structure).name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
