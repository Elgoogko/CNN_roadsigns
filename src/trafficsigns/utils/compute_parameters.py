"""
Script de calcul dynamique des paramètres par rapport au dataset.
"""
from pathlib import Path
import numpy as np
from trafficsigns.data.preprocessing import normalize
from trafficsigns.data.dataset import load_split, load_dataset

def get_mean_std(data_dir: Path | str, batch_size: int = 2048):
    """
    Renvoie la moyenne et la variance sur le dataset d'entrainement seulement.
    :param data_dir: Chemin vers le dataset d'entrainement
    :param batch_size: Nombre d'images sur lequel effectué le calcul
    :return: moyenne et variance sur le dataset d'entrainement
    """
    x, _, _ = load_dataset(data_dir, mmap=True)
    train_idx, _ = load_split(data_dir)
    train_idx = np.sort(train_idx)  # accès mmap plus efficace

    n_pixels = 0
    sum_ = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)

    for start in range(0, len(train_idx), batch_size):
        batch = normalize(x[train_idx[start:start + batch_size]]).astype(np.float64)
        n_pixels += batch.shape[0] * batch.shape[1] * batch.shape[2]
        sum_ += batch.sum(axis=(0, 1, 2))
        sum_sq += (batch ** 2).sum(axis=(0, 1, 2))

    mean = sum_ / n_pixels
    std = np.sqrt(sum_sq / n_pixels - mean ** 2)
    return mean, std

def save_mean_std(output_dir: Path|str, mean, std) -> None:
    """
    Sauvegarde la moyenne et la variance sur le dataset d'entrainement
    :param output_dir: fichier de destination
    :param mean moyenne des canaux
    :param std variance des canaux
    """
    assert Path(output_dir).exists(), "Le chemin spécifié est introuvable"
    np.savez(Path.joinpath(Path(output_dir), "norm_stats"), mean=mean, std=std)

def load_mean_std(data_dir : Path | str):
    """
    Charge les moyennes et variances calculés sur le dataset d'entrainement
    Les moyennes et variances sont chargés sous format [Rmoy,Gmoy,Bmoy]; [Rvar, Gvar, Bvar]
    :param data_dir: Chemin vers le fichier .npz qui stocke les varaibles
    :return: moyenne et variance sur le dataset d'entrainement
    """
    assert Path(data_dir).name.endswith(".npz"), f"Le fichier spécifié n'est pas au format .npz"
    assert Path(data_dir).exists(), f"Le fichier spécifié est introuvable {data_dir}"

    data = np.load(data_dir)
    return data["mean"], data["std"]

if __name__ == "__main__":
    get_mean_std(str(Path(__file__).resolve().parents[3] / "data" / "processed" / "train"))
