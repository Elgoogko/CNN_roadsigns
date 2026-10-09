"""
Découpage train / validation stratifié par classe ET groupé par panneau physique.
"""
from pathlib import Path

import numpy as np

def extract_group_keys(img_paths) -> np.ndarray:
    """
    Chemins (ex. '2/00002_00001_00001.png') → clé de groupe '00002_00001'
    = (classe, séquence), c'est-à-dire un panneau physique.
    Extrait le couple classe / séquence de panneaux d'une série d'images (noms de fichiers)
    :param img_paths: Chemin vers le répertoire d'images
    :return: Liste numpy contenant tous les couples uniques de groupe d'images
    """
    keys = []
    for p in img_paths:
        parts = Path(str(p)).stem.split("_")
        if len(parts) != 3:
            raise ValueError(f"Nom de fichier inattendu (attendu classe_sequence_image) : {p}")
        keys.append(f"{parts[0]}_{parts[1]}")
    return np.array(keys)


def grouped_stratified_split(
    y: np.ndarray, groups: np.ndarray, val_ratio: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pour chaque classe : mélange ses groupes (graine fixée), puis envoie en validation
    les groupes qui rapprochent le plus le nombre d'images de val_ratio * effectif de
    la classe. Au moins un groupe en validation et un en train par classe.
    :param y: Labels des images sous format (N, 43) → One hot encoding
    :param groups:
    :param val_ratio : quantité visée en % du dataset pour la validation (20% du dataset visé)
    :param seed: graine du générateur de nombre
    :return: (train_idx, val_idx), deux tableaux d'indices triés
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio doit être strictement entre 0 et 1")
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("y et groups doivent avoir la même longueur")

    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(y), dtype=bool)

    for label in np.unique(y):
        cls_groups = np.unique(groups[y == label])  # triés : ordre déterministe
        if len(cls_groups) < 2:
            raise ValueError(
                f"Classe {label} : {len(cls_groups)} groupe(s), impossible d'en mettre "
                "au moins un en train et un en validation"
            )
        sizes = {g: int(np.sum(groups == g)) for g in cls_groups}
        target = val_ratio * sum(sizes.values())

        n_val, chosen = 0, []
        for g in rng.permutation(cls_groups):
            if len(chosen) == len(cls_groups) - 1:  # garder un groupe pour le train
                break
            if n_val == 0 or abs(n_val + sizes[g] - target) < abs(n_val - target):
                chosen.append(g)
                n_val += sizes[g]
        val_mask |= np.isin(groups, chosen) & (y == label)

    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def assert_no_leakage(groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """
    Fonction de vérification anti-leak : la fonction vérifie que les ensemble train et validation n'ont aucun panneaux en commun.
    Lève une erreur s'il y a des panneaux en communs.
    :param groups: Tous les groupes de panneaux répertoriés (séquences de panneaux sous formes [classe]_[séquence])
    :param train_idx : tableau des images d'entrainement
    :param val_idx : tableau des images de validation
    """
    if np.intersect1d(train_idx, val_idx).size:
        raise AssertionError("Des indices sont à la fois en train et en validation")
    shared = np.intersect1d(groups[train_idx], groups[val_idx])
    if shared.size:
        raise AssertionError(f"Fuite : {len(shared)} panneau(x) en train ET en validation, ex. {shared[:3].__repr__()}")


def val_fraction_per_class(y: np.ndarray, val_idx: np.ndarray, num_classes: int = 43) -> np.ndarray:
    """
    Part d'images en validation pour chaque classe (contrôle de la stratification).
    Mesure l'écart entre la quantité recherché (20% du dataset) et la quantité réelle crée
    :param y : Labels des images
    :param val_idx : Tableau des images du split "validation"
    :param num_classes Nombre de classes du dataset
    """
    total = np.bincount(y, minlength=num_classes)
    in_val = np.bincount(y[val_idx], minlength=num_classes)
    return in_val / np.maximum(total, 1)


def save_split(train_idx: np.ndarray, val_idx: np.ndarray, output_dir: str | Path, seed: int, val_ratio: float) -> None:
    """
    Sauvegarde le split ; à recharger plutôt que recalculer.
    :param train_idx : Tableau des images d'entrainement
    :param val_idx : tableau des images de validation
    :param output_dir : Chemin du dossier ou enregistré
    :param seed : Graine du générateur de nombre aléatoire
    :param val_ratio : ratio de validation (quantité du dataset utilisé en % pour créer la validation)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "split.npz", train_idx=train_idx, val_idx=val_idx,
             seed=seed, val_ratio=val_ratio)


def load_split(input_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge le split créer précédement pour l'utiliser dans le réseau
    :param input_dir: dossier où se trouvent les fichiers séparés
    :return: le tableau des images de train et validation
    """
    data = np.load(Path(input_dir) / "split.npz")
    return data["train_idx"], data["val_idx"]


if __name__ == "__main__":
    from trafficsigns.config import load_config
    from trafficsigns.data.preprocessing import load_dataset

    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root / "configs" / "base.yaml")
    loaded_seed = cfg["seed"]
    loaded_val_ratio = cfg["split"]["val_ratio"]

    processed = root / "data" / "processed"
    _, new_y, paths = load_dataset(processed / "train", mmap=True)
    new_groups = extract_group_keys(paths)

    new_train_idx, new_val_idx = grouped_stratified_split(new_y, new_groups, loaded_val_ratio, loaded_seed)
    assert_no_leakage(new_groups, new_train_idx, new_val_idx)
    save_split(new_train_idx, new_val_idx, processed / "train", loaded_seed, loaded_val_ratio)

    print(f"train : {len(new_train_idx)} images, {len(np.unique(new_groups[new_train_idx]))} panneaux")
    print(f"val   : {len(new_val_idx)} images, {len(np.unique(new_groups[new_val_idx]))} panneaux "
          f"({len(new_val_idx) / len(new_y):.1%})")
    frac = val_fraction_per_class(new_y, new_val_idx)
    print(f"part de validation par classe : min {frac.min():.1%}, max {frac.max():.1%}")