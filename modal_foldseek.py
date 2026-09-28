# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Foldseek — fast structural search and clustering.

https://github.com/steineggerlab/foldseek

STATUS: UNTESTED against Modal as of 2026-09-28. Compiles, but no run has
happened. The release pin (10-941cd33) and its foldseek-linux-avx2.tar.gz asset
were verified to exist; the AVX2 build assumes Modal's x86 workers support AVX2
(foldseek-linux-sse2.tar.gz is the fallback). Run `foldseek version` in the
built image before committing to a multi-GB database download.

Searches query structures (.pdb/.cif) against a structural database with
`easy-search`, or clusters a set of structures with `easy-cluster`.

NOTE ON LICENSING: Foldseek is GPL-3.0 — the only copyleft dependency in this
repo. It runs here as an unmodified upstream binary inside a Modal container and
is not linked into or distributed with this code, but keep that in mind before
vendoring or redistributing anything built from it.

Databases are multi-GB, so they are downloaded once with `foldseek databases`
into a Modal Volume and reused by every later run. Common names (see
`foldseek databases` for the full list): PDB, Alphafold/UniProt50, Alphafold/Proteome,
Alphafold/Swiss-Prot, ESMAtlas30, CATH50, BFVD.

```
# Search one structure against the PDB
modal run modal_foldseek.py --input-pdbs 2erj.cif

# Search several designs against AFDB50, keep the top 10 hits each
modal run modal_foldseek.py --input-pdbs a.pdb,b.pdb --database Alphafold/UniProt50 --max-seqs 10

# Cluster a set of designs by structural similarity (80% TM-score, 80% coverage)
modal run modal_foldseek.py --input-pdbs a.pdb,b.pdb,c.pdb --mode cluster --min-seq-id 0.0 --coverage 0.8

# Download/refresh a database into the cache volume without searching
modal run modal_foldseek.py --mode download --database PDB
```
"""

import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", None)  # Foldseek is CPU-only here; no GPU needed
TIMEOUT = int(os.environ.get("TIMEOUT", 60))
CPU = float(os.environ.get("CPU", 8))

FOLDSEEK_VOLUME_NAME = "foldseek-databases"
FOLDSEEK_DB_VOLUME = Volume.from_name(FOLDSEEK_VOLUME_NAME, create_if_missing=True)
DB_DIR = f"/{FOLDSEEK_VOLUME_NAME}"

FOLDSEEK_EXE = "/usr/local/bin/foldseek"

# fident/alnlen/evalue/bits/prob plus TM-scores and lDDT, which Foldseek computes
# for structural alignments.
DEFAULT_FORMAT_OUTPUT = (
    "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,"
    "evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,qcov,tcov"
)

image = (
    Image.debian_slim()
    .apt_install("wget", "tar", "gzip", "aria2", "curl")
    .run_commands(
        # Official static release build (AVX2), not conda, for reproducibility.
        "wget -q https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz",
        "tar xzf foldseek-linux-avx2.tar.gz",
        "mv /foldseek/bin/foldseek /usr/local/bin/foldseek",
        "chmod +x /usr/local/bin/foldseek",
        "rm -rf foldseek-linux-avx2.tar.gz /foldseek",
    )
)

app = App("foldseek", image=image)


def _db_prefix(database: str) -> str:
    """Path prefix inside the cache volume for a named Foldseek database.

    Args:
        database (str): Foldseek database name, e.g. "PDB" or "Alphafold/UniProt50".

    Returns:
        str: Absolute path prefix that `foldseek databases` writes to / reads from.
    """
    return f"{DB_DIR}/{database.replace('/', '_')}/db"


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    cpu=CPU,
    volumes={DB_DIR: FOLDSEEK_DB_VOLUME},
)
def download_database(database: str = "PDB", force: bool = False) -> str:
    """Downloads a Foldseek database into the cache Volume if it is not already there.

    Args:
        database (str): Foldseek database name, e.g. "PDB", "Alphafold/UniProt50".
                        Defaults to "PDB".
        force (bool): Re-download even if the database is already cached.
                      Defaults to False.

    Returns:
        str: Absolute path prefix of the cached database inside the container.
    """
    from subprocess import run
    from tempfile import TemporaryDirectory

    db_prefix = _db_prefix(database)
    Path(db_prefix).parent.mkdir(parents=True, exist_ok=True)

    if Path(db_prefix).exists() and not force:
        print(f"Using cached database: {db_prefix}")
        return db_prefix

    with TemporaryDirectory() as td:
        run(
            [FOLDSEEK_EXE, "databases", database, db_prefix, td],
            check=True,
        )

    FOLDSEEK_DB_VOLUME.commit()
    print(f"Downloaded database to: {db_prefix}")
    return db_prefix


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    cpu=CPU,
    volumes={DB_DIR: FOLDSEEK_DB_VOLUME},
)
def foldseek_easy_search(
    pdb_strs: str | list[str],
    pdb_names: str | list[str] | None = None,
    database: str = "PDB",
    max_seqs: int = 1000,
    evalue: float = 0.001,
    alignment_type: int = 2,
    format_output: str | None = None,
    params_str: str | None = None,
) -> list[tuple[Path, bytes]]:
    """Runs `foldseek easy-search` of one or more structures against a database.

    Args:
        pdb_strs (str | list[str]): Contents of the query structure files (.pdb or .cif).
        pdb_names (str | list[str] | None): Original filenames of the queries, used to
                                            name them in the output. Defaults to
                                            query_0.pdb, query_1.pdb, ...
        database (str): Foldseek database name, e.g. "PDB". Defaults to "PDB".
        max_seqs (int): Maximum number of hits to keep per query. Defaults to 1000.
        evalue (float): E-value cutoff for reported hits. Defaults to 0.001.
        alignment_type (int): 0=3Di+AA Gotoh-Smith-Waterman, 1=TMalign, 2=3Di+AA
                              with TM-score reported. Defaults to 2.
        format_output (str | None): Comma-separated alignment output columns.
                                    Defaults to DEFAULT_FORMAT_OUTPUT.
        params_str (str | None): Extra flags appended to the foldseek command line,
                                 e.g. "--exhaustive-search 1".

    Returns:
        list[tuple[Path, bytes]]: Relative output file paths and their byte contents
                                  (the hits TSV plus a header file describing columns).
    """
    import shlex
    from subprocess import run
    from tempfile import TemporaryDirectory

    if isinstance(pdb_strs, str):
        pdb_strs = [pdb_strs]
    if isinstance(pdb_names, str):
        pdb_names = [pdb_names]
    if pdb_names is None:
        pdb_names = [f"query_{i}.pdb" for i in range(len(pdb_strs))]
    assert len(pdb_strs) == len(pdb_names), "pdb_strs and pdb_names length mismatch"

    format_output = format_output or DEFAULT_FORMAT_OUTPUT

    db_prefix = download_database.local(database)

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out, TemporaryDirectory() as td_tmp:
        in_path, out_path = Path(td_in), Path(td_out)
        for pdb_str, pdb_name in zip(pdb_strs, pdb_names):
            (in_path / Path(pdb_name).name).write_text(pdb_str)

        hits_tsv = out_path / "hits.tsv"
        cmd = [
            FOLDSEEK_EXE,
            "easy-search",
            *[str(in_path / Path(n).name) for n in pdb_names],
            db_prefix,
            str(hits_tsv),
            td_tmp,
            "--format-mode", "4",  # TSV with a header line
            "--format-output", format_output,
            "--alignment-type", str(alignment_type),
            "--max-seqs", str(max_seqs),
            "-e", str(evalue),
        ] + shlex.split(params_str or "")

        run(cmd, check=True)

        (out_path / "columns.txt").write_text(format_output.replace(",", "\n") + "\n")

        return [
            (f.relative_to(out_path), f.read_bytes())
            for f in out_path.glob("**/*")
            if f.is_file()
        ]


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    cpu=CPU,
    volumes={DB_DIR: FOLDSEEK_DB_VOLUME},
)
def foldseek_easy_cluster(
    pdb_strs: str | list[str],
    pdb_names: str | list[str] | None = None,
    min_seq_id: float = 0.0,
    coverage: float = 0.8,
    evalue: float = 0.001,
    alignment_type: int = 2,
    params_str: str | None = None,
) -> list[tuple[Path, bytes]]:
    """Runs `foldseek easy-cluster` to group structures by structural similarity.

    Useful for de-duplicating a set of designed backbones before ordering them.

    Args:
        pdb_strs (str | list[str]): Contents of the structure files to cluster.
        pdb_names (str | list[str] | None): Original filenames, used as cluster member
                                            names. Defaults to design_0.pdb, ...
        min_seq_id (float): Minimum sequence identity within a cluster. 0.0 clusters
                            purely on structure. Defaults to 0.0.
        coverage (float): Minimum alignment coverage (-c). Defaults to 0.8.
        evalue (float): E-value cutoff. Defaults to 0.001.
        alignment_type (int): 0=3Di+AA, 1=TMalign, 2=3Di+AA with TM-score.
                              Defaults to 2.
        params_str (str | None): Extra flags appended to the foldseek command line.

    Returns:
        list[tuple[Path, bytes]]: Relative output file paths and their byte contents
                                  (_cluster.tsv, _rep_seq.fasta, _all_seqs.fasta).
    """
    import shlex
    from subprocess import run
    from tempfile import TemporaryDirectory

    if isinstance(pdb_strs, str):
        pdb_strs = [pdb_strs]
    if isinstance(pdb_names, str):
        pdb_names = [pdb_names]
    if pdb_names is None:
        pdb_names = [f"design_{i}.pdb" for i in range(len(pdb_strs))]
    assert len(pdb_strs) == len(pdb_names), "pdb_strs and pdb_names length mismatch"

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out, TemporaryDirectory() as td_tmp:
        in_path, out_path = Path(td_in), Path(td_out)
        for pdb_str, pdb_name in zip(pdb_strs, pdb_names):
            (in_path / Path(pdb_name).name).write_text(pdb_str)

        cmd = [
            FOLDSEEK_EXE,
            "easy-cluster",
            str(in_path),
            str(out_path / "clu"),
            td_tmp,
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "-e", str(evalue),
            "--alignment-type", str(alignment_type),
        ] + shlex.split(params_str or "")

        run(cmd, check=True)

        return [
            (f.relative_to(out_path), f.read_bytes())
            for f in out_path.glob("**/*")
            if f.is_file()
        ]


@app.local_entrypoint()
def main(
    input_pdbs: str | None = None,
    mode: str = "search",
    database: str = "PDB",
    max_seqs: int = 1000,
    evalue: float = 0.001,
    alignment_type: int = 2,
    min_seq_id: float = 0.0,
    coverage: float = 0.8,
    format_output: str | None = None,
    params_str: str | None = None,
    out_dir: str = "./out/foldseek",
    run_name: str | None = None,
):
    """Local entrypoint for Foldseek structural search / clustering on Modal.

    Args:
        input_pdbs (str | None): Comma-separated paths to query structures (.pdb/.cif).
                                 Not needed for mode="download".
        mode (str): "search" (easy-search vs a database), "cluster" (easy-cluster of
                    the inputs), or "download" (cache a database and exit).
                    Defaults to "search".
        database (str): Foldseek database name for search/download. Defaults to "PDB".
        max_seqs (int): Maximum hits per query in search mode. Defaults to 1000.
        evalue (float): E-value cutoff. Defaults to 0.001.
        alignment_type (int): Foldseek --alignment-type. Defaults to 2.
        min_seq_id (float): Cluster mode minimum sequence identity. Defaults to 0.0.
        coverage (float): Cluster mode minimum coverage. Defaults to 0.8.
        format_output (str | None): Search mode output columns (comma-separated).
        params_str (str | None): Extra flags passed straight to foldseek.
        out_dir (str): Local directory for results. Defaults to "./out/foldseek".
        run_name (str | None): Optional subdirectory name; defaults to a timestamp.

    Returns:
        None
    """
    from datetime import datetime

    if mode == "download":
        db_prefix = download_database.remote(database)
        print(f"Database cached at: {db_prefix} (volume {FOLDSEEK_VOLUME_NAME})")
        return

    assert input_pdbs, "--input-pdbs is required for search and cluster modes"
    paths = [Path(p) for p in input_pdbs.split(",")]
    for p in paths:
        assert p.suffix in (".pdb", ".cif"), f"Unsupported input: {p} (need .pdb or .cif)"
    pdb_strs = [p.read_text() for p in paths]
    pdb_names = [p.name for p in paths]

    if mode == "search":
        outputs = foldseek_easy_search.remote(
            pdb_strs=pdb_strs,
            pdb_names=pdb_names,
            database=database,
            max_seqs=max_seqs,
            evalue=evalue,
            alignment_type=alignment_type,
            format_output=format_output,
            params_str=params_str,
        )
    elif mode == "cluster":
        outputs = foldseek_easy_cluster.remote(
            pdb_strs=pdb_strs,
            pdb_names=pdb_names,
            min_seq_id=min_seq_id,
            coverage=coverage,
            evalue=evalue,
            alignment_type=alignment_type,
            params_str=params_str,
        )
    else:
        raise ValueError(f"Unknown mode: {mode} (need search, cluster or download)")

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    for out_file, out_content in outputs:
        target = out_dir_full / out_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out_content)

    print(f"Results saved to: {out_dir_full}")
