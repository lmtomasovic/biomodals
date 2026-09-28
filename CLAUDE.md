# biomodals — local additions

This is a clone of [hgbrian/biomodals](https://github.com/hgbrian/biomodals) (Modal runners
for bioinformatics tools). `origin` points at that upstream repo, **not** a personal fork —
do not push there.

Everything below was added locally and is not upstream.

## Current campaign

De novo binder against **human IL-2** (PDB 2ERJ chain D), blocking the **CD25 / IL-2Rα**
epitope. Config: `2erj_D_binder.yaml`.

Epitope verified against the quaternary complex — the hotspot list is exactly the chain-A
contact set, with zero overlap into β or γ:

| receptor subunit | IL-2 contact residues | in our hotspot list |
|---|---|---|
| chain A — CD25 / IL-2Rα | 35,37,38,41,42,43,44,45,61,62,64,65,68,69,72,105,107 | all 17 |
| chain B — CD122 / IL-2Rβ | 12,13,15,16,19,20,23,84,87,88,91,92,95 | none |
| chain C — CD132 / γc | 11,15,18,22,48,119,120,123,126,127,129,130,133 | none |

So the design intent is CD25-blocking and β/γ-sparing — biasing IL-2 toward effector T/NK
over Tregs. Any counter-screen should check that binders do **not** contact the β/γ sets.

(2ERJ has two copies in the asymmetric unit, A–D and E–H. Chain F touches D at residues
76/79/81 — crystal packing, ignore it.)

## Local tools

| path | what | compute |
|---|---|---|
| `proteinbase/` | physics-only interface scoring + calibrated composite ranking | local, CPU |
| `ipsae/` | ipSAE confidence ranking from PAE matrices | local, CPU |
| `modal_foldseek.py` | structural homology search / clustering | Modal, CPU |
| `modal_mosaic.py` | multi-objective gradient binder design | Modal, GPU |

Each has its own README with provenance and caveats. Read those before trusting output.

## Gotchas that will silently produce garbage

**BoltzGen writes placeholder side chains in pre-refolding files.** Files under
`**/before_refolding/` and `intermediate_designs_inverse_folded/binder_*.cif` have 42–52% of
binder-chain atoms at *exactly coincident* positions. Backbone Rg looks normal, so this
passes casual inspection; the tell is nearest-neighbour distance within the binder chain
(0.00 Å vs ~1.37 Å for real structures). SASA inflates several-fold — 30,992 Å² vs 5,657 Å²
for the same 88-residue chain — and any interface score saturates to a confident,
meaningless 0 or 1.

Score only: `final_ranked_designs/**/rank*.cif`,
`intermediate_designs_inverse_folded/refold_cif/*.cif`, `intermediate_designs/*.cif`.

`proteinbase/` rejects the bad ones automatically. **Rosetta, PyRosetta, freesasa and the
adaptyv `protein-qc` thresholds are all fooled identically** — apply the same caution there.

**Boltz does not write PAE by default.** `modal_boltz.py`'s `DEFAULT_PARAMS` is
`--use_msa_server --seed 42`. ipSAE needs `--write_full_pae` added. BoltzGen `.npz` files
never contain a full PAE matrix regardless.

**The ProteinBase composite is a ranking score, not a probability.** Fit with
`class_weight='balanced'`, so mean output is 0.37 against a true base rate of 0.098. It is
also calibrated on a single target (Nipah glycoprotein-G); transfer is untested.

## SolubleMPNN

No separate tool needed. `modal_ligandmpnn.py`'s image already runs
`get_model_params.sh`, which downloads all MPNN weights including
`solublempnn_v_48_020.pt`. Swap the checkpoint:

```bash
modal run modal_ligandmpnn.py --input-pdb design.pdb \
  --params-str '--seed 1 --model_type "protein_mpnn" \
  --checkpoint_protein_mpnn "/LigandMPNN/model_params/solublempnn_v_48_020.pt" \
  --save_stats 1'
```

Same runner also covers plain ProteinMPNN via `proteinmpnn_v_48_020.pt` (the default).
Relevant because Adaptyv scores expression, and the benchmark tables track
`twist_expression_mg_per_mL` and `twist_sec_pct_main_peak`.

## Already covered — don't rebuild

BoltzGen's `metrics_tmp/metrics_*.npz` already computes sequence liabilities (MetOx, TrpOx,
AspCleave, AspBridge, HydroPatch, DPP4, NTCycl, ProtTryp), hydrophobic patch size, chain
hydrophobicity, ΔSASA, and PLIP hydrogen bonds / salt bridges. Several `protein-qc` checks
duplicate these.

## Related

The adaptyv protein-design-skills plugin is enabled globally. Note it references
`rfdiffusion` and `mosaic` skills whose tools may not all exist here — `proteinmpnn` and
`solublempnn` are covered via `modal_ligandmpnn.py` as above.
