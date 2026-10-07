"""
Observe le dataset, créer des aperçus du dataset
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


def get_quantity_of_class(df: pd.DataFrame, class_column_name: str) -> pd.DataFrame:
    """
    On suppose que la dataframe donnée possèdent la colonne ClassID
    :param class_column_name: nom de la colonne avec les classID
    :param df: dataframe d'entrée
    :return: dataframe des quantités de chaque classe.
    """

    assert class_column_name in df.columns.tolist(), f"La colonne {class_column_name} n'existe pas sur le dataframe"

    data = []
    all_classes = df[class_column_name].unique() #prend toutes les classes uniques

    for elt in all_classes:
        data.append({"ClassID": elt, "Quantité" : len(df[df[class_column_name] == elt]) })

    return pd.DataFrame(data)


def get_size_plot(df: pd.DataFrame) -> None:
    """
    créer un plot (barchart) de la quantité d'images par classes
    :param df: dataframe qui contient les colonnes "ClassID" et "Quantité"
    """
    # 1. triée les classes par quantité d'images croissantes
    df_sorted = df.sort_values(by='Quantité', ascending=True).copy()

    # 2. Conversion de ClassID en string pour éviter que Matplotlib n'en fasse un axe numérique
    df_sorted['ClassID'] = df_sorted['ClassID'].astype(str)

    # 3. Augmentation de la taille de la figure (hauteur de 12 pour espacer les 43 barres)
    fig, ax = plt.subplots(figsize=(10, 12))

    # 4. Dégradé de couleur bleu basé sur la quantité
    norm = plt.Normalize(df_sorted['Quantité'].min(), df_sorted['Quantité'].max())
    colors = plt.cm.Blues(norm(df_sorted['Quantité']))

    # 5. Rendu du barchart
    bars = ax.barh(df_sorted['ClassID'], df_sorted['Quantité'], color=colors, height=0.7)

    # 6. Ajout de la valeur exacte au bout de chaque barre
    ax.bar_label(bars, padding=3, fontsize=8, color="#333333")

    # 7. Habillage du graphique
    ax.set_title("Distribution par classe (triée par quantité)", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Quantité", fontsize=11)
    ax.set_ylabel("ClassID", fontsize=11)

    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    
def get_size_of_images_plot(df: pd.DataFrame) -> None:
    """
    Renvoie le plot (nuage de points) des tailles des images. (Toutes les classes sont prises en compte)
    :param df: Le dataframe descriptif du dataset
    """
    # 1. Taille de figure équilibrée et carrée pour observer le ratio Height/Width
    fig, ax = plt.subplots(figsize=(8, 8))

    # 2. Nuage de points optimisé
    # – alpha=0.3 : gère la superposition (plus c'est foncé, plus il y a de points superposés)
    # - c=df['ClassId'] : colore par classe pour repérer d'éventuels groupes
    # – s=25 : taille des points modérée
    # – cmap='tab20' : palette contrastée pour différencier les classes
    scatter = ax.scatter(
        df['Height'],
        df['Width'],
        c=df['ClassId'],
        cmap='tab20',
        alpha=0.3,
        s=25,
        edgecolors='none'
    )

    # 3. Ligne d'égal ratio 1:1 (diagonale)
    # Permet de voir instantanément si une image est plus haute que large ou carrée
    max_dim = max(df['Height'].max(), df['Width'].max())
    min_dim = min(df['Height'].min(), df['Width'].min())
    ax.plot([min_dim, max_dim], [min_dim, max_dim], color='gray', linestyle='--', alpha=0.7, label='Ratio 1:1 (Carré)')

    # 4. Habillage et lisibilité
    ax.set_title("Répartition des dimensions des images (Height vs Width)", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Hauteur (px)", fontsize=11)
    ax.set_ylabel("Largeur (px)", fontsize=11)

    ax.grid(True, linestyle='--', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()

def analyze_single_class_dataset(dossier_path: Path, verbose_mode : bool = False) -> pd.DataFrame | None:
    """
    Parcourt toutes les images d'un dossier pour extraire leurs caractéristiques
    et vérifier l'homogénéité du dataset.
    """
    dossier = Path(dossier_path)
    # Extensions supportées (PNG, PPM, JPG, JPEG)
    extensions = ('*.png', '*.png', '*.ppm', '*.jpg', '*.jpeg', '*.PPM', '*.JPG', '*.PNG')

    fichiers_images = []
    for ext in extensions:
        fichiers_images.extend(dossier.glob(ext))

    if not fichiers_images:
        print(f"Aucune image trouvée dans {dossier_path}")
        return None

    resultats = []

    if verbose_mode:
        print(f"Analyse de {len(fichiers_images)} images en cours...")

    for img_path in fichiers_images:
        with Image.open(img_path) as img:
            # 1. Format & Mode Couleur (Pillow)
            fmt = img.format  # ex: 'PNG', 'PPM', 'JPEG'
            mode = img.mode  # ex: 'RGB', 'L' (gris), 'RGBA'

            # 2. Conversion en tableau Numpy pour analyser les pixels
            arr = np.array(img)

            # 3. Nombre de canaux
            if arr.ndim == 2:
                nb_canaux = 1  # Image en niveaux de gris
            else:
                nb_canaux = arr.shape[2]  # ex: 3 pour RGB, 4 pour RGBA

            # 4. Type de données ET Plage des valeurs (Min / Max)
            dtype_pixels = str(arr.dtype)  # ex: 'uint8'
            val_min = arr.min()
            val_max = arr.max()

            resultats.append({
                'Fichier': img_path.name,
                'Format': fmt,
                'Mode': mode,
                'Canaux': nb_canaux,
                'Type_Dtype': dtype_pixels,
                'Plage_Min': val_min,
                'Plage_Max': val_max,
                'Dimensions': img.size  # (largeur, hauteur)
            })

    # Conversion en DataFrame Pandas
    df_info = pd.DataFrame(resultats)
    df_info_compressed = pd.DataFrame()

    df_info_compressed['Formats'] = df_info['Format'].unique()
    df_info_compressed['Modes'] = df_info['Mode'].unique()
    df_info_compressed['Canaux'] =  df_info['Canaux'].unique()
    df_info_compressed['Type_Dtype'] = df_info['Type_Dtype'].unique()

    if verbose_mode:
        # -------------------------------------------------------------
        # VÉRIFICATION D'HOMOGÉNÉITÉ SUR L'ENSEMBLE DU DATASET
        # -------------------------------------------------------------
        print("\n" + "=" * 50)
        print(f"RÉSUMÉ DU DATASET {dossier_path.parent.name}")
        print("=" * 50)

        print(f"Formats trouvés        : {list(df_info_compressed['Formats'])}")
        print(f"Modes couleur trouvés  : {list(df_info_compressed['Modes'])}")
        print(f"Nombre de canaux       : {list(df_info_compressed['Canaux'])}")
        print(f"Types de pixels (dtype): {list(df_info_compressed['Type_Dtype'])}")
        print(f"Plage globale observée : [{df_info['Plage_Min'].min()} - {df_info['Plage_Max'].max()}]")

        # Diagnostic d'homogénéité
        est_homogene = (
                len(list(df_info_compressed['Formats'])) == 1 and
                len(list(df_info_compressed['Modes'])) == 1 and
                len(list(df_info_compressed['Canaux'])) == 1 and
                len(list(df_info_compressed['Type_Dtype'])) == 1
        )

        if est_homogene:
            print(
                "\n✅ VÉRIFICATION RÉUSSIE : Toutes les images possèdent exactement le même format, mode couleur et structure de pixels !")
        else:
            print("\n⚠️ AVERTISSEMENT : Le dataset est HÉTÉROGÈNE !")
            if len(list(df_info_compressed['Modes'])) > 1:
                print(f" -> Conflit de modes couleur détecté : {df_info['Mode'].value_counts().to_dict()}")
            if len(list(df_info_compressed['Formats'])) > 1:
                print(f" -> Conflit de formats détecté : {df_info['Format'].value_counts().to_dict()}")

    return df_info_compressed


def analyze_dataset(dossier_path: Path) -> pd.DataFrame:
    """
    Analyse du dataset complet et de ses images. (Format, canaux ect.)
    :param dossier_path: Chemin du dossier où se trouve le dataset
    """

    dossiers = [p.name for p in dossier_path.iterdir()]

    resultats = []
    for dossier in dossiers:
        print("Traitement du dossier : ", dossier)
        resultats.append(analyze_single_class_dataset(dossier_path / dossier))

    dict_total = pd.concat(resultats, ignore_index=True)

    formats_uniques = dict_total['Formats'].unique()
    modes_uniques = dict_total['Modes'].unique()
    canaux_uniques = dict_total['Canaux'].unique()
    dtypes_uniques = dict_total['Type_Dtype'].unique()

    est_homogene = (
            len(formats_uniques) == 1 and
            len(modes_uniques) == 1 and
            len(canaux_uniques) == 1 and
            len(dtypes_uniques) == 1
    )

    if est_homogene:
        print(
            "\nVÉRIFICATION RÉUSSIE : Toutes les images possèdent exactement le même format, mode couleur et structure de pixels !")
    else:
        print("\nAVERTISSEMENT : Le dataset est HÉTÉROGÈNE !")

    return dict_total



def create_frame(
    raw_dir: str | Path,
    correspondance: dict[int, str],
    cols: int = 6,
    img_size: tuple[int, int] = (64, 64),
    seed: int = 42,
    save_path: str | Path | None = "planche_classes.png",
) -> None:
    """
    raw_dir : chemin vers le dossier contenant les sous-dossiers (0, 1, 2, ...)
    Sélectionne une image aléatoire par sous-dossier et affiche la grille de toutes les images (une par classe).

    :param raw_dir: Chemin vers le dossier
    :param correspondance: entre les classes (int) et leurs noms (str)
    :param cols: nombre de colonnes sur la planche
    :param img_size: nombre des images sur la planche
    :param seed: seed pour la sélection aléatoire des images dans les dossiers
    :param save_path: chemin vers le dossier
    """
    raw_dir = Path(raw_dir)
    random.seed(seed)

    sous_dossiers = sorted(
        (d for d in raw_dir.iterdir() if d.is_dir()),
        key=lambda d: int(d.name),  # trie numériquement : 0, 1, ..., 42
    )

    rows = (len(sous_dossiers) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = axes.flatten()

    for i, dossier in enumerate(sous_dossiers):
        class_id = int(dossier.name)
        images = list(dossier.iterdir())
        img_path = random.choice(images)

        img = Image.open(img_path).convert("RGB").resize(img_size)
        axes[i].imshow(img)
        axes[i].set_title(f"{class_id}\n{correspondance.get(class_id, '?')}", fontsize=7)
        axes[i].axis("off")

    # Masquer les cases vides
    for ax in axes[len(sous_dossiers):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def create_class_frame(
    dossier: str | Path,
    titre: str = "",
    n: int = 10,
    cols: int = 5,
    img_size: tuple[int, int] = (96, 96),
    seed: int = 42,
    save_path: str | Path | None = None,
) -> None:
    """
    Sélectionne n images aléatoires d'un dossier et les affiche sur une planche
    avec un titre global.
    :param save_path:
    :param seed: Graine pour le générateur de nombres aléatoires
    :param dossier: Chemin vers le dossier
    :param titre: titre de la planche (str)
    :param n: nombre de colonnes sur la planche
    :param cols: nombre de colonnes sur la planche
    :param img_size: nombre des images sur la planche
    """
    dossier = Path(dossier)
    random.seed(seed)

    images = random.sample(list(dossier.iterdir()), n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.4))
    axes = axes.flatten()

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB").resize(img_size)
        axes[i].imshow(img)
        axes[i].axis("off")

    # Masquer les cases vides
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(titre, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")


def create_every_class_frame(
    raw_dir: str | Path,
    correspondance: dict[int, str],
    n: int = 10,
    cols: int = 5,
    save_dir: str | Path | None = None,
) -> None:
    """
    Boucle sur tous les sous-dossiers de raw_dir et génère une planche
    par classe.
    :param correspondance:
    :param raw_dir: Chemin vers le dossier
    : param correspondance dictionnaire de correspondance entre les classID et le nom réel
    :param n: nombre de colonnes sur la planche
    :param cols: nombre de colonnes sur la planche
    :param save_dir: chemin vers le dossier

    """
    raw_dir = Path(raw_dir)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for dossier in sorted(raw_dir.iterdir(), key=lambda d: int(d.name)):
        class_id = int(dossier.name)
        titre = f"{class_id} — {correspondance.get(class_id, '?')}"
        save_path = save_dir / f"planche_{class_id:02d}.png" if save_dir else None
        create_class_frame(dossier, titre=titre, n=n, cols=cols, save_path=save_path)


def get_mean_std(data_dir, batch_size=64, img_size=(64, 64)):
    """
    Calcule la moyenne (mean) et l'écart-type (std) exacts par canal (RGB)
    sur l'ensemble des images d'un dossier.
    :param data_dir: Chemin vers le dossier
    :param batch_size: nombre de batches
    :param img_size: nombre des images sur la planche
    """

    # 1. Pipeline minimal : redimensionnement et conversion en Tensor [0.0, 1.0]
    transform_calc = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # 2. Chargement du dataset (sans normalisation)
    dataset = ImageFolder(root=data_dir, transform=transform_calc)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0

    print(f"Calcul en cours sur {len(dataset)} images...")

    # 3. Accumulation par batch
    for images, _ in loader:
        # images shape: [batch_size, 3, height, width]
        # On calcule la moyenne par canal pour chaque image du batch
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    # 4. Calcul de la moyenne globale et de l'écart-type global (E[X^2] - (E[X])^2)
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean.tolist(), std.tolist()
