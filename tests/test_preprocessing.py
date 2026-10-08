"""Tests du prétraitement partagé. Images synthétiques uniquement : aucune dépendance à data/."""
import cv2
import numpy as np
import pytest

from trafficsigns.data import preprocessing as pp

NUM_CLASSES = pp.NUM_CLASSES


# ---------- utilitaires ----------
def _write_png(path, img_rgb):
    """Écrit une image RGB (uint8) en PNG (OpenCV attend du BGR)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def _make_raw_tree(root, classes=range(NUM_CLASSES)):
    """
    Arbo type GTSRB : root/<classe>/<classe>_<piste>_<image>.png, 2 pistes par classe.
    Chaque image est uniforme avec la valeur classe*5 : on peut vérifier que X et y
    restent alignés. Les tailles varient pour tester le resize.
    """
    for c in classes:
        for track, (h, w) in enumerate([(30, 30), (60, 40)]):
            img = np.full((h, w, 3), c * 5, dtype=np.uint8)
            _write_png(root / str(c) / f"{c:05d}_{track:05d}_00000.png", img)
    return root


@pytest.fixture
def raw_dir(tmp_path):
    root = _make_raw_tree(tmp_path / "raw")
    (root / "0" / "GT-00000.csv").write_text("fichier parasite")  # doit être ignoré
    return root


# ---------- resize ----------
@pytest.mark.parametrize("shape", [(25, 25), (43, 43), (225, 243), (30, 80), (80, 30)])
def test_resize_gives_target_shape_and_keeps_uint8(shape):
    img = np.random.default_rng(0).integers(0, 256, (*shape, 3), dtype=np.uint8)
    out = pp.resize_img(img)
    assert out.shape == (pp.IMG_SIZE[1], pp.IMG_SIZE[0], 3)
    assert out.dtype == np.uint8


def test_resize_is_deterministic():
    img = np.random.default_rng(1).integers(0, 256, (43, 61, 3), dtype=np.uint8)
    assert np.array_equal(pp.resize_img(img), pp.resize_img(img))


# ---------- normalisation ----------
def test_normalize_white_and_black():
    white = np.full((48, 48, 3), 255, dtype=np.uint8)
    black = np.zeros((48, 48, 3), dtype=np.uint8)
    assert np.all(pp.normalize(white) == 1.0)
    assert np.all(pp.normalize(black) == 0.0)


def test_normalize_dtype_range_and_input_untouched():
    img = np.random.default_rng(2).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    copy = img.copy()
    out = pp.normalize(img)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert np.array_equal(img, copy)


def test_normalize_batch_keeps_shape():
    batch = np.zeros((7, 48, 48, 3), dtype=np.uint8)
    assert pp.normalize(batch).shape == (7, 48, 48, 3)


def test_normalize_without_stats_is_plain_division():
    img = np.random.default_rng(3).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    assert np.allclose(pp.normalize(img), img / 255.0)


def test_standardize_centers_and_reduces_per_channel():
    img = np.full((48, 48, 3), 128, dtype=np.uint8)
    mean = [128 / 255, 0.0, 0.0]
    std = [1.0, 0.5, 0.25]
    out = pp.normalize(img, mean, std)
    assert np.allclose(out[..., 0], 0.0, atol=1e-6)
    assert np.allclose(out[..., 1], (128 / 255) / 0.5)
    assert np.allclose(out[..., 2], (128 / 255) / 0.25)


# ---------- lecture disque ----------
def test_load_image_returns_rgb(tmp_path):
    red = np.zeros((20, 20, 3), dtype=np.uint8)
    red[..., 0] = 255  # rouge pur en RGB
    _write_png(tmp_path / "red.png", red)
    out = pp.load_image(tmp_path / "red.png")
    assert out.shape == (20, 20, 3) and out.dtype == np.uint8
    assert np.all(out[..., 0] == 255) and np.all(out[..., 1:] == 0)  # détecte un BGR/RGB inversé


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        pp.load_image(tmp_path / "absent.png")


def test_preprocess_single_image_end_to_end(tmp_path):
    img = np.random.default_rng(4).integers(0, 256, (43, 61, 3), dtype=np.uint8)
    _write_png(tmp_path / "a.png", img)
    out = pp.preprocess_single_image(tmp_path / "a.png")
    assert out.shape == (pp.IMG_SIZE[1], pp.IMG_SIZE[0], 3)
    assert out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0


# ---------- construction du dataset ----------
def test_build_dataset_shapes_and_dtypes(raw_dir):
    X, y, paths = pp.build_dataset(raw_dir)
    assert X.shape == (2 * NUM_CLASSES, pp.IMG_SIZE[1], pp.IMG_SIZE[0], 3)  # le CSV est ignoré
    assert X.dtype == np.uint8
    assert y.shape == (len(X),) and len(paths) == len(X)


def test_build_dataset_labels_match_class_ids_and_numeric_order(raw_dir):
    _, y, _ = pp.build_dataset(raw_dir)
    assert sorted(set(y.tolist())) == list(range(NUM_CLASSES))
    assert np.all(np.diff(y) >= 0)  # "10" vient après "9", pas après "1"


def test_build_dataset_keeps_X_y_aligned(raw_dir):
    X, y, _ = pp.build_dataset(raw_dir)
    means = X.reshape(len(X), -1).mean(axis=1)
    assert np.allclose(means, y * 5)


def test_build_dataset_paths_are_posix_and_consistent_with_labels(raw_dir):
    _, y, paths = pp.build_dataset(raw_dir)
    assert all("\\" not in p for p in paths)
    assert all(p.startswith(f"{label}/") for p, label in zip(paths, y))
    assert all(p.split("/")[1].startswith(f"{label:05d}_") for p, label in zip(paths, y))


def test_build_dataset_is_deterministic(raw_dir):
    X1, y1, p1 = pp.build_dataset(raw_dir)
    X2, y2, p2 = pp.build_dataset(raw_dir)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2) and np.array_equal(p1, p2)


def test_build_dataset_rejects_file_in_wrong_class_folder(raw_dir):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    _write_png(raw_dir / "3" / "00007_00000_00000.png", img)
    with pytest.raises(ValueError, match="mauvaise classe"):
        pp.build_dataset(raw_dir)


def test_build_dataset_rejects_missing_classes(tmp_path):
    root = _make_raw_tree(tmp_path / "raw_partial", classes=[0, 1])
    with pytest.raises(ValueError, match="inattendues"):
        pp.build_dataset(root)


# ---------- sauvegarde / rechargement ----------
@pytest.mark.parametrize("mmap", [False, True])
def test_save_then_load_roundtrip(raw_dir, tmp_path, mmap):
    X, y, paths = pp.build_dataset(raw_dir)
    pp.save_dataset(X, y, paths, tmp_path / "cache")
    X2, y2, p2 = pp.load_dataset(tmp_path / "cache", mmap=mmap)
    assert np.array_equal(X, X2) and np.array_equal(y, y2) and np.array_equal(paths, p2)


# ---------- encodage des labels ----------
def test_one_hot_roundtrip_for_all_classes():
    y = np.arange(NUM_CLASSES)
    oh = pp.to_one_hot(y)
    assert oh.shape == (NUM_CLASSES, NUM_CLASSES) and oh.dtype == np.float32
    assert np.all(oh.sum(axis=1) == 1)
    assert np.array_equal(pp.from_one_hot(oh), y)


def test_from_one_hot_accepts_softmax_probabilities():
    probs = np.full((2, NUM_CLASSES), 0.01, dtype=np.float32)
    probs[0, 5] = 0.6
    probs[1, 42] = 0.7
    assert pp.from_one_hot(probs).tolist() == [5, 42]