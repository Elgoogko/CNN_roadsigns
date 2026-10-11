import numpy as np
from pathlib import Path

def load_split(input_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge le split créer précédement pour l'utiliser dans le réseau
    :param input_dir: dossier où se trouvent les fichiers séparés
    :return: le tableau des images de train et validation
    """
    data = np.load(Path(input_dir) / "split.npz")
    return data["train_idx"], data["val_idx"]


def load_dataset(input_dir: str | Path, mmap: bool = False):
    """
    Recharge (x, y, paths). mmap=True évite de charger x entièrement en RAM.
    :param input_dir: Répertoire ou sont stockées les tableaux
    :param mmap: divise le tableau numpy pour éviter de surchargé la mémoire
    """
    input_dir = Path(input_dir)
    x = np.load(input_dir / "img_mats.npy", mmap_mode="r" if mmap else None)
    return x, np.load(input_dir / "labels.npy"), np.load(input_dir / "paths.npy")
