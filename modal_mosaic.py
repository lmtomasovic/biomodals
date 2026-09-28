# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Mosaic https://github.com/escalante-bio/mosaic (MIT)

STATUS: UNTESTED against Modal as of 2026-09-28. This file compiles and its use
of Mosaic's API was read from the pinned source, but no run has happened. Known
unknowns, roughly in order of how likely they are to bite:

  - The image build is unverified. Mosaic has no tags, no releases and no
    uv.lock, and its own dependencies are unpinned git repos (joltz, protenij,
    joltzgen, jopenfold3, esmjfold2), so `uv sync` can resolve differently than
    when this was written. Build before assuming anything else works.
  - The epitope index mapping is the dangerous one. Mosaic takes raw 0-based
    indices into the target sequence; the author-numbering -> index conversion
    here was written but never checked against the real 2ERJ chain D residue
    list. If it is off, the optimizer silently targets the wrong residues and
    every downstream metric still looks healthy. Verify it by hand first.
  - Featurization needs network at runtime for any use_msa=True chain (it calls
    the public ColabFold MSA server). Use --no-msa to avoid that.
  - Templates default OFF; Mosaic's own source comment is "boltz wants perfect
    .cifs". Whether 2ERJ chain D passes template featurization is unconfirmed.
  - Loss weights are starting points adapted from a Protenix example, not tuned
    for this target. Mosaic's README is emphatic that they need per-target work.
  - The design code runs via subprocess under Mosaic's own uv venv rather than
    being imported in-process, because `uv pip install` does not honour Mosaic's
    [tool.uv.sources]. That venv path is unverified.

Multi-objective, gradient-based protein binder design. Mosaic is a JAX library
that reimplements a zoo of protein models (Boltz-1/2, AF2, Protenix, OpenFold3,
ESMFold2, ProteinMPNN, ESM-2/ESM-C, a megascale stability head, ...) behind one
differentiable interface, so several of them can be composed into a single loss
and optimized over a continuously relaxed sequence (a PSSM of shape [L, 20]).
It is the same family of methods as ColabDesign / RSO / BindCraft, but the
objective is yours to write.

This runner uses the Boltz-2 backend and optimizes with `simplex_APGM`
(accelerated projected gradient on the probability simplex), then re-predicts
the winning PSSM with extra recycling and writes the complex out.

## Worked example: an IL-2 binder that blocks CD25 but not CD122/CD132

Target: human IL-2, PDB 2ERJ chain D. The therapeutic goal is a "non-alpha"
blocker: occupy the CD25 (IL-2R-alpha) epitope while leaving the IL-2R-beta
(CD122) and gamma-c (CD132) faces free.

Author residue numbers on IL-2 (2ERJ chain D):
  CD25  contacts: 35,37,38,41,42,43,44,45,61,62,64,65,68,69,72,105,107
  CD122 contacts: 12,13,15,16,19,20,23,84,87,88,91,92,95
  CD132 contacts: 11,15,18,22,48,119,120,123,126,127,129,130,133

Mosaic expresses this directly: `sp.BinderTargetContact(epitope_idx=...)`
restricts the binder-target contact loss to a subset of target residues, and
`LossTerm` supports `+`, `-` and scalar `*`, so a *negative* design term is just
a subtracted contact loss. `--avoid` is subtracted (clipped, so the reward for
avoiding is bounded and cannot run away), `--epitope` is added.

```bash
wget https://files.rcsb.org/download/2ERJ.cif -O 2erj.cif

# Positive-only: bind the CD25 epitope
modal run modal_mosaic.py --input-cif 2erj.cif --target-chain D \\
    --epitope 35,37,38,41,42,43,44,45,61,62,64,65,68,69,72,105,107 \\
    --binder-length 70 --num-designs 4

# Positive + negative: bind CD25 epitope, avoid the CD122/CD132 faces
modal run modal_mosaic.py --input-cif 2erj.cif --target-chain D \\
    --epitope 35,37,38,41,42,43,44,45,61,62,64,65,68,69,72,105,107 \\
    --avoid 12,13,15,16,19,20,23,84,87,88,91,92,95,11,18,22,48,119,120,123,126,127,129,130,133 \\
    --avoid-weight 1.0 --binder-length 70 --num-designs 8 --run-name il2_nonalpha

# No structure file: sequence-only target, optimizer picks its own epitope
modal run modal_mosaic.py --target-sequence APTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSETTFMCEYADETATIVEFLNRWITFCQSIISTLT \\
    --binder-length 70 --num-designs 2
```

Environment overrides: `GPU=A100 TIMEOUT=240 modal run modal_mosaic.py ...`

## Notes

- Target residues are given as *author* residue numbers from the input file;
  they are mapped to 0-based indices into the modeled chain here, so gaps in the
  deposited structure do not shift the epitope.
- Boltz-2 featurization calls the public ColabFold MSA server
  (https://api.colabfold.com) for any chain with MSAs enabled. Pass
  `--no-msa` for a single-sequence target if you would rather not.
- Mosaic's own README is blunt that this is a toolkit, not a turnkey pipeline:
  weights need tuning per target and designs should still be filtered
  (ipSAE / AF2 / BindCraft-style metrics) before ordering anything.
"""

import json
import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "H100")
TIMEOUT = int(os.environ.get("TIMEOUT", 180))

# Mosaic publishes no tags or releases and no uv.lock, so the only reproducible
# handle is an explicit commit SHA. This is HEAD of main at the time of writing
# (2026-09-24, "Update ESMFold2 design (#88)"). Bump deliberately: mosaic's own
# dependencies are unpinned git repos, so `uv sync` still resolves those fresh.
MOSAIC_COMMIT = "b94b9d4eb9907a700a6d78ed2d29d3704c5df46c"

MOSAIC_DIR = "/root/mosaic"
# uv sync builds the project venv here; mosaic pins requires-python == 3.12.*
# and installs a CPU torch plus jax[cuda12] (JAX does all the GPU work).
MOSAIC_PYTHON = f"{MOSAIC_DIR}/.venv/bin/python"

# Boltz-2 checkpoints, the CCD, and ColabFold MSAs are large and slow to fetch.
# Mosaic routes all of them through MOSAIC_CACHE_DIR, so one Volume covers them.
CACHE_VOLUME = Volume.from_name("mosaic-cache", create_if_missing=True)
CACHE_DIR = "/cache"


def _download_weights():
    """Pre-fetch the Boltz-2 checkpoint and CCD into the cache Volume.

    Mosaic downloads these lazily on first `Boltz2()` construction, which would
    otherwise burn GPU-minutes (and risk a timeout) on the first run.
    """
    from boltz.main import download_boltz2

    cache = Path(CACHE_DIR) / "boltz"
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)
    CACHE_VOLUME.commit()


image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone https://github.com/escalante-bio/mosaic {MOSAIC_DIR}",
        f"cd {MOSAIC_DIR} && git checkout {MOSAIC_COMMIT}",
        # jax-cuda is mosaic's own dependency group: jax[cuda12] + the cuBLAS /
        # cuSOLVER pins it needs. The CUDA runtime comes from the pip wheels, so
        # no CUDA base image is required (same approach as modal_rso.py's
        # jax[cuda] on debian_slim, just pinned by mosaic instead of by us).
        f"cd {MOSAIC_DIR} && uv sync --group jax-cuda",
        # Modal's agent runs in the image's system python; the design script
        # runs under the uv venv, which therefore does not need modal itself.
    )
    .env({"MOSAIC_CACHE_DIR": CACHE_DIR, "JAX_PLATFORMS": "cuda"})
    .run_function(_download_weights, volumes={CACHE_DIR: CACHE_VOLUME})
)

app = App("mosaic", image=image)


# The design script runs under mosaic's uv venv via subprocess rather than being
# imported here, because mosaic's dependency graph (joltz, protenij, joltzgen,
# esmjfold2, a CPU-only torch index, ...) is resolved by `uv sync` from
# [tool.uv.sources] and cannot be reproduced with a flat pip install.
DESIGN_SCRIPT = r'''
"""Mosaic binder design driven by a JSON config. Runs inside mosaic's uv venv."""

import json
import sys
from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np

import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import ClippedLoss
from mosaic.models.boltz2 import Boltz2
from mosaic.optimizers import simplex_APGM
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain

cfg = json.loads(Path(sys.argv[1]).read_text())
out_dir = Path(cfg["out_dir"])
out_dir.mkdir(parents=True, exist_ok=True)


def target_from_structure(path, chain_id):
    """Read one chain and return (sequence, gemmi.Chain, {author resnum: index}).

    The index map is built from the *modeled* residues, so epitope residue
    numbers stay correct even when the deposited structure has gaps.
    """
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_ligands_and_waters()
    chain = st[0][chain_id]
    residues = [r for r in chain]
    sequence = gemmi.one_letter_code([r.name for r in residues]).upper()
    numbering = {r.seqid.num: i for i, r in enumerate(residues)}
    return sequence, chain, numbering


def resolve(residue_numbers, numbering, label):
    """Map author residue numbers to 0-based indices into the target sequence."""
    missing = [n for n in residue_numbers if n not in numbering]
    if missing:
        raise SystemExit(
            f"{label} residues {missing} are not modeled in the target chain"
        )
    return [numbering[n] for n in residue_numbers]


if cfg["input_path"]:
    target_sequence, template_chain, numbering = target_from_structure(
        cfg["input_path"], cfg["target_chain"]
    )
else:
    target_sequence = cfg["target_sequence"]
    template_chain, numbering = None, {i + 1: i for i in range(len(target_sequence))}

# Author numbering -> 0-based columns of the binder/target distogram block.
epitope_idx = resolve(cfg["epitope"], numbering, "epitope") or None
avoid_idx = resolve(cfg["avoid"], numbering, "avoid") or None
print(f"target length {len(target_sequence)}")
print(f"epitope_idx {epitope_idx}")
print(f"avoid_idx {avoid_idx}")

binder_length = cfg["binder_length"]

boltz2 = Boltz2()
mpnn = load_mpnn_sol(0.0)

target_chain = TargetChain(
    sequence=target_sequence,
    use_msa=cfg["use_msa"],
    # Boltz is fussy about template CIFs; templating is opt-in.
    template_chain=template_chain if cfg["use_template"] else None,
)

# Complex features (binder + target) and monomer features (binder alone).
complex_features, complex_writer = boltz2.binder_features(
    binder_length=binder_length, chains=[target_chain]
)
monomer_features, _ = boltz2.binder_features(binder_length=binder_length, chains=[])

# --- The objective -----------------------------------------------------------
#
# BinderTargetContact returns -mean log P(contact), so ADDING it rewards contact
# and SUBTRACTING it rewards the absence of contact. The negative term is
# clipped to [0, avoid_cap] so "push this face further away" saturates instead
# of dominating the gradient once the off-target contact probability is low.
positive_contact = sp.BinderTargetContact(epitope_idx=epitope_idx)

complex_loss = (
    cfg["epitope_weight"] * positive_contact
    + 1.0 * sp.WithinBinderContact()
    + 0.05 * sp.BinderTargetPAE()
    + 0.05 * sp.TargetBinderPAE()
    + 0.4 * sp.WithinBinderPAE()
    + 0.025 * sp.IPTMLoss()
    + 0.025 * sp.pTMEnergy()
    + 0.1 * sp.PLDDTLoss()
    + cfg["mpnn_weight"]
    * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(cfg["mpnn_temp"]))
)

if avoid_idx is not None and cfg["avoid_weight"] > 0:
    complex_loss = complex_loss - cfg["avoid_weight"] * ClippedLoss(
        sp.BinderTargetContact(epitope_idx=avoid_idx),
        0.0,
        cfg["avoid_cap"],
        name="avoid_contact",
    )

monomer_loss = (
    0.2 * sp.PLDDTLoss()
    + sp.DistogramRadiusOfGyration(target_radius=cfg["target_radius"])
    + cfg["helix_weight"] * sp.HelixLoss()
)

loss = boltz2.build_loss(
    loss=complex_loss,
    features=complex_features,
    recycling_steps=cfg["recycling_steps"],
    sampling_steps=cfg["sampling_steps"],
) + 0.5 * boltz2.build_loss(
    loss=monomer_loss,
    features=monomer_features,
    recycling_steps=cfg["recycling_steps"],
    sampling_steps=cfg["sampling_steps"],
)

# --- Optimize ----------------------------------------------------------------

summary = []
for design_i in range(cfg["num_designs"]):
    seed = cfg["seed"] + design_i
    print(f"\n=== design {design_i} (seed {seed}) ===", flush=True)

    PSSM = jax.nn.softmax(
        0.5
        * jax.random.gumbel(key=jax.random.key(seed), shape=(binder_length, 20))
    )

    # Stage 1: soft, exploratory. Mosaic's README suggests 0.1*sqrt(L).
    _, PSSM = simplex_APGM(
        loss_function=loss,
        x=PSSM,
        n_steps=cfg["soft_steps"],
        stepsize=cfg["stepsize"] * np.sqrt(binder_length),
        momentum=0.3,
        scale=1.0,
        max_gradient_norm=1.0,
    )

    # Stage 2: sharpen toward a one-hot sequence (scale > 1 pushes to vertices).
    PSSM, _ = simplex_APGM(
        loss_function=loss,
        x=PSSM,
        n_steps=cfg["sharp_steps"],
        stepsize=2.0 * cfg["stepsize"] * np.sqrt(binder_length),
        momentum=0.0,
        scale=1.3,
        max_gradient_norm=1.0,
    )

    sequence = "".join(TOKENS[int(i)] for i in PSSM.argmax(-1))
    print(f"designed: {sequence}", flush=True)

    # Re-predict with more recycling than design used, for honest metrics.
    prediction = boltz2.predict(
        PSSM=PSSM,
        features=complex_features,
        writer=complex_writer,
        recycling_steps=cfg["predict_recycling_steps"],
        sampling_steps=cfg["sampling_steps"],
        key=jax.random.key(seed),
    )

    name = f"design_{design_i}"
    (out_dir / f"{name}.pdb").write_text(prediction.st.make_minimal_pdb())
    (out_dir / f"{name}.fasta").write_text(
        f">{name}_binder\n{sequence}\n>{name}_target\n{target_sequence}\n"
    )
    np.save(out_dir / f"{name}_pssm.npy", np.asarray(PSSM))

    metrics = {
        "design": name,
        "seed": seed,
        "sequence": sequence,
        "binder_length": binder_length,
        "iptm": float(prediction.iptm),
        "binder_plddt": float(np.asarray(prediction.plddt)[:binder_length].mean()),
        "complex_plddt": float(np.asarray(prediction.plddt).mean()),
        "interface_pae": float(
            np.asarray(prediction.pae)[:binder_length, binder_length:].mean()
        ),
    }
    if epitope_idx is not None:
        # Mean PAE from binder to the residues we asked it to bind, and to the
        # residues we asked it to avoid: the specificity readout.
        pae = np.asarray(prediction.pae)
        metrics["epitope_pae"] = float(
            pae[:binder_length, binder_length:][:, epitope_idx].mean()
        )
    if avoid_idx is not None:
        pae = np.asarray(prediction.pae)
        metrics["avoid_pae"] = float(
            pae[:binder_length, binder_length:][:, avoid_idx].mean()
        )
    print(json.dumps(metrics, indent=2), flush=True)
    summary.append(metrics)

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
'''


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={CACHE_DIR: CACHE_VOLUME},
)
def mosaic_design(config: dict, input_bytes: bytes | None, input_name: str | None):
    """Run Mosaic gradient-based binder design on Modal.

    Args:
        config: Design configuration passed through to the in-container script.
            Keys mirror the local entrypoint arguments (binder_length, epitope,
            avoid, weights, optimizer steps, ...).
        input_bytes: Raw bytes of the target structure file (CIF or PDB), or
            None when designing against a bare sequence.
        input_name: Filename for input_bytes, used to pick the parser.

    Returns:
        list[tuple[Path, bytes]]: (relative output path, file content) for every
            file written by the design run -- per-design .pdb complexes, .fasta
            sequences, .npy PSSMs, and summary.json.
    """
    from subprocess import run
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        out_dir = work / "out"

        if input_bytes is not None:
            input_path = work / input_name
            input_path.write_bytes(input_bytes)
            config["input_path"] = str(input_path)
        else:
            config["input_path"] = None
        config["out_dir"] = str(out_dir)

        script_path = work / "mosaic_design.py"
        script_path.write_text(DESIGN_SCRIPT)
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config))

        cmd = [MOSAIC_PYTHON, str(script_path), str(config_path)]
        print(f"Running: {' '.join(cmd)}")
        run(cmd, check=True, cwd=MOSAIC_DIR)

        return [
            (out_file.relative_to(out_dir), out_file.read_bytes())
            for out_file in out_dir.rglob("*")
            if out_file.is_file()
        ]


def _parse_residues(residues: str | None) -> list[int]:
    """Parse "35,37,41-45" into a sorted list of unique residue numbers."""
    if not residues:
        return []
    out: set[int] = set()
    for part in residues.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


@app.local_entrypoint()
def main(
    input_cif: str | None = None,
    target_sequence: str | None = None,
    target_chain: str = "A",
    epitope: str | None = None,
    avoid: str | None = None,
    binder_length: int = 70,
    num_designs: int = 1,
    seed: int = 0,
    epitope_weight: float = 4.0,
    avoid_weight: float = 1.0,
    avoid_cap: float = 3.0,
    mpnn_weight: float = 10.0,
    mpnn_temp: float = 0.001,
    helix_weight: float = 0.3,
    target_radius: float = 15.0,
    soft_steps: int = 100,
    sharp_steps: int = 20,
    stepsize: float = 0.1,
    recycling_steps: int = 1,
    predict_recycling_steps: int = 4,
    sampling_steps: int = 25,
    no_msa: bool = False,
    use_template: bool = False,
    out_dir: str = "./out/mosaic",
    run_name: str | None = None,
):
    """Run Mosaic binder design and save the results locally.

    Args:
        input_cif: Path to the target structure (CIF or PDB). Mutually exclusive
            with target_sequence.
        target_sequence: Target amino-acid sequence, if you have no structure.
        target_chain: Chain to extract from input_cif (e.g. "D" for IL-2 in 2ERJ).
        epitope: Target author residue numbers the binder should contact, as
            "35,37,41-45". Rewarded.
        avoid: Target author residue numbers the binder should NOT contact.
            Penalized (clipped), giving a positive/negative specificity objective.
        binder_length: Length of the de novo binder to design.
        num_designs: Number of independent trajectories (seeds) to run.
        seed: Base random seed; design i uses seed + i.
        epitope_weight: Weight on the rewarded BinderTargetContact term.
        avoid_weight: Weight on the subtracted (penalized) contact term.
        avoid_cap: Upper clip on the penalized term, bounding the avoidance reward.
        mpnn_weight: Weight on ProteinMPNN InverseFoldingSequenceRecovery.
        mpnn_temp: ProteinMPNN sampling temperature for that term.
        helix_weight: Weight on the monomer HelixLoss (raise for helical bundles).
        target_radius: Target radius of gyration for the binder monomer, in A.
        soft_steps: simplex_APGM steps in the soft/exploratory stage.
        sharp_steps: simplex_APGM steps in the sharpening stage.
        stepsize: Base step size; scaled by sqrt(binder_length) as Mosaic suggests.
        recycling_steps: Boltz-2 recycling steps during optimization.
        predict_recycling_steps: Boltz-2 recycling steps for the final prediction.
        sampling_steps: Boltz-2 diffusion sampling steps.
        no_msa: Disable MSAs for the target (skips the ColabFold MSA server).
        use_template: Pass the input chain to Boltz-2 as a structural template.
            Off by default: Boltz template featurization needs a clean CIF.
        out_dir: Local directory to write results into.
        run_name: Optional run name; defaults to a timestamp.
    """
    from datetime import datetime

    if bool(input_cif) == bool(target_sequence):
        raise ValueError("Provide exactly one of --input-cif or --target-sequence")
    if use_template and not input_cif:
        raise ValueError("--use-template requires --input-cif")

    epitope_residues = _parse_residues(epitope)
    avoid_residues = _parse_residues(avoid)
    overlap = sorted(set(epitope_residues) & set(avoid_residues))
    if overlap:
        # Published contact lists overlap (IL-2 residue 15 sits in both the
        # CD122 and CD132 footprints), so a residue can easily end up on both
        # sides of the objective, where the two terms would cancel.
        raise ValueError(f"Residues in both --epitope and --avoid: {overlap}")

    config = {
        "target_sequence": target_sequence,
        "target_chain": target_chain,
        "epitope": epitope_residues,
        "avoid": avoid_residues,
        "binder_length": binder_length,
        "num_designs": num_designs,
        "seed": seed,
        "epitope_weight": epitope_weight,
        "avoid_weight": avoid_weight,
        "avoid_cap": avoid_cap,
        "mpnn_weight": mpnn_weight,
        "mpnn_temp": mpnn_temp,
        "helix_weight": helix_weight,
        "target_radius": target_radius,
        "soft_steps": soft_steps,
        "sharp_steps": sharp_steps,
        "stepsize": stepsize,
        "recycling_steps": recycling_steps,
        "predict_recycling_steps": predict_recycling_steps,
        "sampling_steps": sampling_steps,
        "use_msa": not no_msa,
        "use_template": use_template,
    }

    input_bytes = Path(input_cif).read_bytes() if input_cif else None
    input_name = Path(input_cif).name if input_cif else None

    outputs = mosaic_design.remote(config, input_bytes, input_name)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        output_path = out_dir_full / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(out_content)

    print(f"\nResults saved to: {out_dir_full}")
