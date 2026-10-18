"""FlyWire graph, rewire statistics, ER null, groups, npz I/O, FlyWire loader schema handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flybywire.connectome import Connectome, directed_rewire, erdos_renyi_like, match_groups


def _random_digraph(n=200, p=0.05, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.random((n, n)) < p
    np.fill_diagonal(M, False)
    return M


def test_flywire_graph_shape_and_groups(conn):
    assert conn.n == 457 and conn.source == "flywire"
    sizes = {g: len(v) for g, v in conn.groups.items()}
    assert sizes == {"ER": 278, "EPG": 51, "PEN": 42, "PEG": 20, "D7": 42, "PFL3": 24}
    assert not conn.M.diagonal().any()
    assert sum(sizes.values()) == conn.n  # groups are disjoint and cover every neuron


@pytest.mark.parametrize("seed", [0, 1])
def test_rewire_statistics_random_graph(seed):
    M = _random_digraph(seed=seed)
    M2, st = directed_rewire(M, swaps_per_edge=10, seed=seed)
    assert np.array_equal(M.sum(0), M2.sum(0)) and np.array_equal(M.sum(1), M2.sum(1))
    assert M2.sum() == M.sum() and not M2.diagonal().any()
    assert st.swaps_done == st.swaps_target == 10 * M.sum()
    assert 0.0 <= st.surviving_fraction < 0.2  # sparse graph: few original edges survive 10|E| swaps
    expected = (M & M2).sum() / M.sum()
    assert st.surviving_fraction == pytest.approx(expected)


def test_rewire_is_seeded():
    M = _random_digraph()
    a, _ = directed_rewire(M, seed=5)
    b, _ = directed_rewire(M, seed=5)
    c, _ = directed_rewire(M, seed=6)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_rewire_rejects_self_loops():
    M = _random_digraph()
    M[0, 0] = True
    with pytest.raises(ValueError):
        directed_rewire(M)


def test_er_like():
    M = _random_digraph()
    E = erdos_renyi_like(M, seed=0)
    assert E.shape == M.shape and E.sum() == M.sum() and not E.diagonal().any()


def test_match_groups_disjoint():
    types = ["EPG", "EPGt", "PEN_a(PEN1)", "Delta7", "ER4d", "PFL3", "hDeltaB"]
    g = match_groups(types, {"EPG": ["EPG*"], "PEN": ["PEN*"], "D7": ["Delta7*"], "ER": ["ER*"]})
    assert g["EPG"].tolist() == [0, 1] and g["PEN"].tolist() == [2] and g["ER"].tolist() == [4]
    with pytest.raises(ValueError):
        match_groups(types, {"A": ["EPG*"], "B": ["EP*"]})


def test_connectome_npz_roundtrip(conn, tmp_path):
    p = tmp_path / "g.npz"
    conn.save(p)
    back = Connectome.load(p)
    assert np.array_equal(back.M, conn.M) and back.source == conn.source
    assert all(np.array_equal(back.groups[g], conn.groups[g]) for g in conn.groups)
    assert np.array_equal(back.instances, conn.instances)


def _write_fake_flywire(raw, cfg):
    fw = cfg.connectome.flywire
    raw.mkdir(parents=True)
    ann = pd.DataFrame({"root_id": [1, 2, 3, 4, 5, 6, 7], "primary_type": ["ER1", "EPG", "EPG", "PEN_a", "PEG",
                                                                             "Delta7", "PFL3"]})
    ann.to_csv(raw / fw.annotations_file, index=False)
    con = pd.DataFrame({
        "pre_root_id": [1, 1, 2, 2, 3, 4, 2, 9],
        "post_root_id": [2, 2, 4, 2, 7, 3, 5, 2],
        "neuropil": ["EB", "EB_L", "EB", "EB", "LAL", "PB", "EB", "EB"],
        "syn_count": [3, 4, 10, 50, 6, 2, 5, 99],
    })
    con.to_csv(raw / fw.connections_file, index=False)
    pd.DataFrame({"root_id": [2, 3], "name": ["EPG_R1", "EPG_L1"]}).to_csv(raw / fw.names_file, index=False)


def test_flywire_loader_on_fake_tables(cfg, tmp_path):
    from flybywire.connectome.flywire import load_flywire_cx

    _write_fake_flywire(tmp_path / "raw", cfg)
    conn = load_flywire_cx(cfg, tmp_path / "raw", verbose=False)
    ids = conn.neuron_ids.tolist()
    assert sorted(ids) == [1, 2, 3, 4, 5, 6, 7]
    pos = {i: k for k, i in enumerate(ids)}
    assert conn.M[pos[1], pos[2]]  # 3 + 4 summed across neuropils = 7 >= 5
    assert conn.M[pos[2], pos[4]] and conn.M[pos[3], pos[7]] and conn.M[pos[2], pos[5]]
    assert not conn.M[pos[4], pos[3]]  # 2 < 5 dropped
    assert not conn.M.diagonal().any()  # 2 -> 2 self-connection dropped
    assert conn.n_edges == 4
    assert conn.instances[pos[2]] == "EPG_R1"


def test_flywire_loader_schema_mismatch(cfg, tmp_path, capsys):
    from omegaconf import OmegaConf

    from flybywire.connectome.flywire import SchemaError, load_flywire_cx

    _write_fake_flywire(tmp_path / "raw", cfg)
    bad = OmegaConf.merge(cfg, OmegaConf.create({"connectome": {"flywire": {"columns": {"type": "cell_type"}}}}))
    with pytest.raises(SchemaError):
        load_flywire_cx(bad, tmp_path / "raw", verbose=False)
    err = capsys.readouterr().err
    assert "columns actually found" in err and "primary_type" in err
