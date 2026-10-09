"""
Prétraitement partagé des images : resize, normalisation, labels, encodage.

Deux étapes distinctes :
  1. build_dataset / save_dataset : lecture + resize -> tableaux uint8 mis en cache
  2. normalize : uint8 -> float32 [0, 1], appliquée à chaque lot au chargement
"""
from pathlib import Path

import cv2
import numpy as np
from trafficsigns.config import load_config

ROOT = Path(__file__).resolve().parents[3]

cfg = load_config(ROOT / "configs" / "base.yaml")

# Valeurs de Fallback
IMG_SIZE = cfg["image"]["size"]
NUM_CLASSES = int(cfg["dataset"]["num_classes"])
IMG_EXTENSIONS = cfg["dataset"]["extensions"]


def load_image(img_path: str | Path) -> np.ndarray:
    """
    Lit une image et la renvoie en RGB, uint8, de forme (H, L, 3).
    :param img_path: Chemin vers une image
    :return: tableau de taille (N, H, L, 3).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Image illisible : {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_img(img_mat: np.ndarray, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Redimensionne une image (H, L, 3) vers size = (largeur, hauteur).
    INTER_AREA pour réduire, INTER_LINEAR pour agrandir.
    :param img_mat: Chemin vers une image
    :param size: largeur, hauteur attendue de l'image
    :return: tableau de taille (N, H, L, 3).
    """
    h, w = img_mat.shape[:2]
    shrinking = w >= size[0] and h >= size[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(img_mat, size, interpolation=interpolation)


def rescale_img(img_mat: np.ndarray, downscaling_value: float = 255.0) -> np.ndarray:
    """
    Convertit en float32 et divise : [0, 255] -> [0.0, 1.0].
    :param img_mat: Chemin vers une image
    :param downscaling_value: valeur par laquelle divisé les valeurs de l'image
    :return: tableau de taille (N, H, L, 3).
    """
    return img_mat.astype(np.float32) / downscaling_value


def standardize_img(img_mat: np.ndarray, mean, std) -> np.ndarray:
    """
    Centre et réduit par canal : (x - moyenne) / écart-type.
    :param img_mat: Chemin vers une image
    :param mean: moyenne
    :param std: variance
    :return Image centrée réduite
    """
    return (img_mat - mean) / std


def normalize(x_uint8: np.ndarray, mean=None, std=None) -> np.ndarray:
    """
    uint8 -> float32 dans [0, 1] ; standardisation seulement si mean et std
    sont fournis (à calculer sur le train uniquement). Accepte une image (H,L,3)
    ou un lot (N,H,L,3).
    :param x_uint8 Tableau numpy en uint8 représentant les images
    :param mean: moyenne
    :param std: variance
    :return tableau de taille (N, H, L, 3) des images centrés réduites
    """
    x = rescale_img(x_uint8)
    if mean is not None and std is not None:
        x = standardize_img(
            x, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
        )
    return x


def load_and_resize(img_path: str | Path, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Chemin -> tableau uint8 (H, L, 3) redimensionné. Sert à construire le cache.
    :param img_path: Chemin vers une image
    :param size: largeur, hauteur attendue de l'image
    :return image chargée en mémoire et re dimensioné à la taille size
    """
    return resize_img(load_image(img_path), size)


def preprocess_single_image(img_path, size=IMG_SIZE, mean=None, std=None) -> np.ndarray:
    """
    Chemin -> entrée prête pour le modèle (float32). Pour l'inférence / la démo.
    Applique la pipeline complète à l'image.
    :param img_path: Chemin vers une image
    :param size: largeur, hauteur attendue de l'image
    :param mean: moyenne
    :param std: variance
    :return image convertie en tableau, re dimensionné et normalisée
    """
    return normalize(load_and_resize(img_path, size), mean, std)


def build_dataset(raw_dir: str | Path, size: tuple[int, int] = IMG_SIZE):
    """
    Parcourt raw_dir/<classe>/<fichier> dans un ordre déterministe.
    :param raw_dir: Chemin vers une base de l'ordre d'un ordre
    :param size: Taille attendu de l'image
    :return: X (N, H, L, 3) uint8, y (N,) int64, paths (N,) chemins relatifs à raw_dir
    """
    raw_dir = Path(raw_dir)
    class_dirs = sorted(
        (d for d in raw_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    images, labels, paths = [], [], []
    for cls_dir in class_dirs:
        label = int(cls_dir.name)
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS)
        print(f"Classe {label} : {len(files)} images")

        for f in files:
            if int(f.name.split("_")[0]) != label:
                raise ValueError(f"Fichier {f} rangé dans la mauvaise classe")
            images.append(load_and_resize(f, size))
            labels.append(label)
            paths.append(f.relative_to(raw_dir).as_posix())

    y = np.array(labels, dtype=np.int64)
    if set(y.tolist()) != set(range(NUM_CLASSES)):
        raise ValueError(f"Classes trouvées inattendues : {sorted(set(y.tolist()))}")

    return np.stack(images), y, np.array(paths)


def save_dataset(x: np.ndarray, y:np.ndarray, paths:np.ndarray, output_dir: str | Path) -> None:
    """
    Sauvegarde le cache en .npy (non compressé, relisible avec mmap_mode).
    :param x: Images du dataset transformé en tableau
    :param y: Label des images transformé en One-hot-encoding (tableau de taille (43, N))
    :param paths: chemins uniques de toutes les images associés à leur index dans X
    :param output_dir: répertoire de sauvegarde des fichiers
    :return:
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "img_mats.npy", x)
    np.save(output_dir / "labels.npy", y)
    np.save(output_dir / "paths.npy", paths)
    print(f"Sauvegardé dans '{output_dir}' : X {x.shape} {x.dtype}, y {y.shape}")


def load_dataset(input_dir: str | Path, mmap: bool = False):
    """
    Recharge (X, y, paths). mmap=True évite de charger X entièrement en RAM.
    :param input_dir: Répertoire ou sont stockées les tableaux
    :param mmap: divise le tableau numpy pour éviter de surchargé la mémoire
    """
    input_dir = Path(input_dir)
    X = np.load(input_dir / "img_mats.npy", mmap_mode="r" if mmap else None)
    return X, np.load(input_dir / "labels.npy"), np.load(input_dir / "paths.npy")


def to_one_hot(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """

    Transforme les entiers spécifiant les classes en one hot encoding. Entiers (N,) -> one-hot (N, num_classes).
    :param y: Vecteurs des classes (N,1)
    :param num_classes: nombre de classes du dataset
    :return: matrice one hot encoding de taille (N, num_classes)
    """
    return np.eye(num_classes, dtype=np.float32)[y]


def from_one_hot(y_one_hot: np.ndarray) -> np.ndarray:
    """
    One-hot ou probabilités softmax (N, num_classes) -> entiers (N,).
    :param y_one_hot: matrice de taille (N, num_classes)
    :return: matrice de taille (N, num_classes) représentant les classes prédites
    """
    return np.argmax(y_one_hot, axis=-1)


if __name__ == "__main__":

    X, y, paths = build_dataset(ROOT / "data" / "raw" / "gtsrb" / "Train")
    save_dataset(X, y, paths, ROOT / "data" / "processed" / "train")