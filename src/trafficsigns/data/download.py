"""Téléchargement idempotent du dataset GTSRB depuis Kaggle."""
from __future__ import annotations #Annotations compatibles pour collab

import argparse             # Lis les arguments en lignes de commandes (--config, --force)
import logging              # Logger, print
import shutil               # Copie de dossiers entiers
from pathlib import Path    # Manipule les chemins comme des objets, indépendament du système

import kagglehub            # La BDD

# Init logger
logger = logging.getLogger(__name__)

# Fichier marqueur
MARKER = ".download_complete"

def download_dataset(handle: str, raw_dir: str | Path, force: bool = False) -> Path:
    """Télécharge le dataset dans raw_dir et renvoie ce chemin.
    Ne fait rien si le téléchargement a déjà été complété (sauf force=True).

    :param handle Identifiant Kaggle du dataset
    :type handle: str
    :param raw_dir dossier de destination
    :type raw_dir: str | Path
    :param force si True, on re-télécharge
    :type force: bool
    :return: Chemin ou se trouve les données téléchargées
    :rtype: Path
    """
    raw_dir = Path(raw_dir)
    marker = raw_dir / MARKER

    # le fichier .download_complete permet de vérifié si le dataset à déjà été télécharger
    if marker.exists() and not force:
        logger.info("Dataset déjà présent dans %s, rien à faire.", raw_dir)
        return raw_dir

    logger.info("Téléchargement de %s ...", handle)
    cache_path = Path(kagglehub.dataset_download(handle, force_download=force))

    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_path, raw_dir, dirs_exist_ok=True)
    marker.write_text(handle)  # écrit en dernier : prouve que la copie est complète

    logger.info("Dataset prêt dans %s", raw_dir)
    return raw_dir


def main() -> None:
    from trafficsigns.config import load_config  # à venir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config(args.config)
    download_dataset(cfg["data"]["kaggle_handle"], cfg["data"]["raw_dir"], args.force)


if __name__ == "__main__":
    main()