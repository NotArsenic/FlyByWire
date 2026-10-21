"""FlyWire central-complex loader (Codex snapshot 783, manual download).

Reads the connections table and the cell-type annotation table from ``data/flywire/raw/``,
keeps neurons whose type matches the configured group patterns, sums synapse counts per
(pre, post) pair across neuropils, drops pairs below ``min_synapses`` and self-connections, and
writes ``data/flywire/processed/cx_graph.npz`` (see :class:`~flybywire.connectome.celltypes.Connectome`).

File and column names come from ``connectome.flywire`` in the config. On a schema mismatch the
loader prints the columns it found and exits.

Usage: ``python -m flybywire.connectome.flywire [override=value ...]``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from flybywire.connectome.celltypes import Connectome, match_groups


class SchemaError(SystemExit):
    pass


def _read_table(path: Path, needed: dict[str, str], what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} table not found: {path}\n"
            "Download it from FlyWire Codex (snapshot 783) into data/flywire/raw/; see README 'Data'."
        )
    suffixes = "".join(path.suffixes).lower()
    if ".parquet" in suffixes:
        import pyarrow.parquet as pq

        header = pq.read_schema(path).names
    elif ".feather" in suffixes:
        import pyarrow.feather as pf

        header = pf.read_table(path, memory_map=True).column_names
    else:
        header = pd.read_csv(path, nrows=0).columns
    missing = {k: v for k, v in needed.items() if v not in header}
    if missing:
        print(f"\nSchema mismatch in {what} table {path}:", file=sys.stderr)
        print(f"  expected columns (config key -> name): {missing}", file=sys.stderr)
        print(f"  columns actually found: {list(header)}", file=sys.stderr)
        print("  Fix connectome.flywire.columns in the config and re-run.", file=sys.stderr)
        raise SchemaError(2)
    usecols = sorted(set(needed.values()))
    if ".parquet" in suffixes:
        return pd.read_parquet(path, columns=usecols)
    if ".feather" in suffixes:
        return pd.read_feather(path, columns=usecols)
    return pd.read_csv(path, usecols=usecols, engine="pyarrow")


def load_flywire_cx(cfg, raw_dir: Path, verbose: bool = True) -> Connectome:
    fw = cfg.connectome.flywire
    cols = fw.columns
    patterns = {g: list(p) for g, p in cfg.connectome.groups.items()}

    ann = _read_table(raw_dir / fw.annotations_file, {"id": cols.id, "type": cols.type}, "annotation")
    ann = ann.dropna(subset=[cols.type]).drop_duplicates(subset=[cols.id])
    ann[cols.type] = ann[cols.type].astype(str)
    groups_all = match_groups(ann[cols.type].tolist(), patterns)
    keep_rows = np.concatenate([v for v in groups_all.values()]) if groups_all else np.array([], int)
    if len(keep_rows) == 0:
        print("No neurons matched the configured group patterns. Types present (sample):", file=sys.stderr)
        print(sorted(ann[cols.type].unique())[:200], file=sys.stderr)
        raise SchemaError(2)

    # deterministic order: config group order, then type, then id
    group_of_row = {}
    for g, rows in groups_all.items():
        for r in rows:
            group_of_row[int(r)] = g
    sel = ann.iloc[sorted(group_of_row)].copy()
    sel["_group"] = [group_of_row[int(r)] for r in sorted(group_of_row)]
    sel["_gorder"] = sel["_group"].map({g: i for i, g in enumerate(patterns)})
    sel = sel.sort_values(["_gorder", cols.type, cols.id]).reset_index(drop=True)
    ids = sel[cols.id].to_numpy(np.int64)
    types = sel[cols.type].to_numpy(str)
    groups = {g: np.flatnonzero(sel["_group"].to_numpy() == g) for g in patterns}

    instances = types.copy()
    if fw.get("names_file"):
        names_path = raw_dir / fw.names_file
        if names_path.exists():
            names = _read_table(names_path, {"id": cols.name_id, "name": cols.name}, "names")
            name_map = dict(zip(names[cols.name_id].to_numpy(np.int64), names[cols.name].astype(str), strict=True))
            instances = np.asarray([name_map.get(int(i), t) for i, t in zip(ids, types, strict=True)])
        elif verbose:
            print(f"(names table {names_path} not found; ring layout will fall back to 'tuning')")

    # sum over neuropils before thresholding, so min_synapses applies to the whole pair
    needed = {"pre": cols.pre, "post": cols.post, "syn_count": cols.syn_count}
    con = _read_table(raw_dir / fw.connections_file, needed, "connections")
    idset = pd.Index(ids)
    con = con[con[cols.pre].isin(idset) & con[cols.post].isin(idset)]
    con = con.groupby([cols.pre, cols.post], as_index=False)[cols.syn_count].sum()
    con = con[(con[cols.syn_count] >= int(cfg.connectome.min_synapses)) & (con[cols.pre] != con[cols.post])]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    i = pos.loc[con[cols.pre].to_numpy(np.int64)].to_numpy()
    j = pos.loc[con[cols.post].to_numpy(np.int64)].to_numpy()
    M = np.zeros((len(ids), len(ids)), dtype=bool)
    M[i, j] = True

    conn = Connectome(
        M=M,
        neuron_ids=ids,
        types=types,
        instances=instances,
        groups=groups,
        source="flywire",
        meta={"snapshot": int(fw.snapshot), "min_synapses": int(cfg.connectome.min_synapses)},
    )
    if verbose:
        print("\nMatched cell types:")
        for g, idx in groups.items():
            sub = pd.Series(types[idx]).value_counts().sort_index()
            print(f"  [{g}] {len(idx)} neurons")
            for t, n in sub.items():
                print(f"      {t:<24} {n:>5}")
        print(f"\nN = {conn.n}   edges = {conn.n_edges}   density = {conn.density:.4f}")
        empty = [g for g, idx in groups.items() if len(idx) == 0]
        if empty:
            print(f"WARNING: groups with no neurons: {empty}; check the patterns against the annotation table")
    return conn


def main(argv: list[str] | None = None) -> int:
    from flybywire.config import data_dir, load_config

    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    base = data_dir(cfg) / "flywire"
    conn = load_flywire_cx(cfg, base / "raw")
    out = base / "processed" / "cx_graph.npz"
    conn.save(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
