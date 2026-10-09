"""Tests du split train/validation. Données synthétiques au format réel de GTSRB."""
import numpy as np
import pytest

from trafficsigns.data import split as sp


def make_dataset(tracks_per_class=(4, 6, 10, 7, 20), seed=0):
    """
    Une classe par entrée de tracks_per_class ; chaque piste a 20 à 40 vues
    (comme les séquences de GTSRB). Renvoie "y", paths au format 'c/ccccc_sssss_iiiii.png'.
    """
    rng = np.random.default_rng(seed)
    y, paths = [], []
    for c, n_tracks in enumerate(tracks_per_class):
        for t in range(n_tracks):
            for i in range(int(rng.integers(20, 41))):
                y.append(c)
                paths.append(f"{c}/{c:05d}_{t:05d}_{i:05d}.png")
    return np.array(y), np.array(paths)


@pytest.fixture
def data():
    y, paths = make_dataset()
    return y, paths, sp.extract_group_keys(paths)


# ---------- clés de groupe ----------
def test_extract_group_keys_uses_class_and_sequence():
    keys = sp.extract_group_keys(["2/00002_00001_00001.png", "2/00002_00001_00002.png",
                                  "0/00000_00000_00001.png"])
    assert keys.tolist() == ["00002_00001", "00002_00001", "00000_00000"]


def test_extract_group_keys_rejects_unexpected_names():
    with pytest.raises(ValueError):
        sp.extract_group_keys(["0/image.png"])


def test_same_sequence_number_in_two_classes_gives_two_groups():
    keys = sp.extract_group_keys(["0/00000_00003_00000.png", "1/00001_00003_00000.png"])
    assert keys[0] != keys[1]


# ---------- propriétés du split ----------
def test_indices_are_disjoint_and_cover_everything(data):
    y, _, groups = data
    tr, va = sp.grouped_stratified_split(y, groups, 0.2, seed=42)
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr)) == len(tr) and len(set(va)) == len(va)
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(y)))


def test_no_physical_sign_in_both_sets(data):
    """Test anti-fuite principal : aucun (classe, séquence) des deux côtés."""
    y, _, groups = data
    tr, va = sp.grouped_stratified_split(y, groups, 0.2, seed=42)
    assert set(groups[tr]).isdisjoint(set(groups[va]))
    sp.assert_no_leakage(groups, tr, va)  # ne doit pas lever


def test_every_class_in_both_sets(data):
    y, _, groups = data
    tr, va = sp.grouped_stratified_split(y, groups, 0.2, seed=42)
    assert set(y[tr]) == set(y[va]) == set(np.unique(y))


def test_at_least_one_group_per_class_in_validation_even_for_tiny_ratio(data):
    y, _, groups = data
    tr, va = sp.grouped_stratified_split(y, groups, 0.01, seed=42)
    assert set(y[va]) == set(np.unique(y))
    assert set(y[tr]) == set(np.unique(y))


def test_validation_proportion_is_close_to_requested(data):
    y, _, groups = data
    tr, va = sp.grouped_stratified_split(y, groups, 0.2, seed=42)
    assert abs(len(va) / len(y) - 0.2) < 0.05
    frac = sp.val_fraction_per_class(y, va, num_classes=5)
    assert np.all(np.abs(frac - 0.2) < 0.15)


def test_same_seed_same_split_and_different_seed_different_split(data):
    y, _, groups = data
    a = sp.grouped_stratified_split(y, groups, 0.2, seed=42)
    b = sp.grouped_stratified_split(y, groups, 0.2, seed=42)
    c = sp.grouped_stratified_split(y, groups, 0.2, seed=7)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[0], b[0])
    assert not np.array_equal(a[1], c[1])


def test_split_does_not_depend_on_row_order():
    """Mêmes données mélangées -> mêmes panneaux en validation."""
    y, paths = make_dataset()
    groups = sp.extract_group_keys(paths)
    _, va = sp.grouped_stratified_split(y, groups, 0.2, seed=42)
    perm = np.random.default_rng(1).permutation(len(y))
    _, va_perm = sp.grouped_stratified_split(y[perm], groups[perm], 0.2, seed=42)
    assert set(groups[va]) == set(groups[perm][va_perm])


def test_class_with_a_single_group_is_rejected():
    y = np.array([0] * 30 + [1] * 30 + [1] * 30)
    groups = np.array(["a"] * 30 + ["b"] * 30 + ["c"] * 30)  # classe 0 : un seul panneau
    with pytest.raises(ValueError, match="Classe 0"):
        sp.grouped_stratified_split(y, groups, 0.2, seed=42)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
def test_invalid_ratio_is_rejected(data, ratio):
    y, _, groups = data
    with pytest.raises(ValueError):
        sp.grouped_stratified_split(y, groups, ratio, seed=42)


# ---------- le détecteur de fuite détecte vraiment les fuites ----------
def test_leakage_detector_catches_naive_random_split(data):
    """Un split image par image au hasard DOIT être signalé comme fuite."""
    y, _, groups = data
    idx = np.random.default_rng(0).permutation(len(y))
    cut = int(0.8 * len(y))
    with pytest.raises(AssertionError, match="Fuite"):
        sp.assert_no_leakage(groups, np.sort(idx[:cut]), np.sort(idx[cut:]))


def test_leakage_detector_catches_shared_indices(data):
    y, _, groups = data
    with pytest.raises(AssertionError):
        sp.assert_no_leakage(groups, np.array([0, 1, 2]), np.array([2, 3]))


# ---------- sauvegarde ----------
def test_save_then_load_roundtrip(data, tmp_path):
    y, _, groups = data
    tr, va = sp.grouped_stratified_split(y, groups, 0.2, seed=42)
    sp.save_split(tr, va, tmp_path / "out", seed=42, val_ratio=0.2)
    tr2, va2 = sp.load_split(tmp_path / "out")
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)