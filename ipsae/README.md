# ipSAE

Ranks predicted protein–protein complexes from the PAE matrix. Separates true from false
complexes more reliably than ipTM, which dilutes interface confidence across disordered or
accessory regions.

Vendored from [DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE) — MIT, commit
`6174cf9`, 2026-01-03. `ipsae.py` and `af2rechain.py` are unmodified upstream; `run.py` is
the batch wrapper added here.

Paper: Dunbrack, *Rēs ipSAE loquuntur*, [bioRxiv 2025.02.10.637595](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2)

## Usage

```bash
python -m ipsae.run out/boltz/ -o ipsae.csv
```

```python
from ipsae import find_pairs, run_one, summarise
```

Options: `--pae-cutoff` / `--dist-cutoff` (both default 10), `--full-out` for the raw
per-direction rows.

`ipsae_min` is the minimum over the two asymmetric directions — the variant binder
benchmarks report, and the best single Boltz-2 metric (AUROC 0.638) in the ProteinBase
calibration.

## Input requirements — read this first

ipSAE needs a **full N×N PAE matrix**. Two gotchas that will otherwise waste your time:

**BoltzGen output does not work.** Its `.npz` files carry only reduced scalars —
`interaction_pae`, `min_interaction_pae`, `min_design_to_target_pae` — never the matrix.
There is no way to recover ipSAE from them.

**Boltz needs an extra flag.** `modal_boltz.py`'s `DEFAULT_PARAMS` is
`--use_msa_server --seed 42`, which does *not* write the PAE. Pass `--write_full_pae`:

```bash
modal run modal_boltz.py --input-faa design.faa \
  --params-str '--use_msa_server --seed 42 --write_full_pae'
```

That produces `pae_<name>_model_N.npz` beside `<name>_model_N.cif`, which `run.py` pairs
automatically. AlphaFold3 (`*_full_data_0.json`) and ColabFold AF2 outputs are also paired.

So the order is: design → co-fold with Boltz → ipSAE. It is a validation step on a
predicted complex, not something you run on design output directly.

## Where it fits

Complementary to `proteinbase/`, which scores interface *physics* from coordinates with no
network confidence terms. ipSAE scores the predictor's *confidence*. Different signals, so
worth carrying both.

Tempering expectation: in the ProteinBase model selection, "physics + min ipSAE" was
statistically indistinguishable from physics alone and the parsimony rule dropped ipSAE.
On that one calibration target it added nothing measurable over the composite. It is cheap,
it is what benchmarks report, and it may behave differently on your target — but do not
expect it to move your ranking much.

## Verification

Run at integration time against upstream's committed example
(`Example/fold_aurka_0_tpx2_0_model_0.cif`): reproduces the published values exactly,
ipSAE 0.448952 / 0.866498 for the A→B and B→A directions.

One difference from upstream's committed table: `ipTM_af` comes back 0.000 rather than
0.870, because that field is read from AF3's `summary_confidences` JSON, which is not in
the example directory. It does not affect ipSAE itself.
