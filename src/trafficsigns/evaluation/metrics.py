import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from trafficsigns.config import ROOT, load_config
from trafficsigns.data.preprocessing import preprocess_single_image, load_and_resize
from trafficsigns.utils.compute_parameters import get_mean_std
import matplotlib.pyplot as plt
import seaborn as sns

def save_confusion_matrix_plot(
        cm: np.ndarray,
        out_path: Path,
        normalize: bool = True
) -> None:
    """
    Génère et sauvegarde une figure de la matrice de confusion.
    :param cm: Matrice de confusion (N, N)
    :param out_path: Chemin de sauvegarde (ex: .png)
    :param normalize: Si True, affiche des pourcentages (0.0 à 1.0) au lieu des effectifs bruts.
    """
    if normalize:
        # Évite la division par zéro pour les classes sans échantillons
        cm_display = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        title = "Matrice de confusion (Normalisée)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Matrice de confusion (Nombre d'images)"

    # Taille adaptée pour 43 classes
    plt.figure(figsize=(14, 12))

    # Heatmap Seaborn
    sns.heatmap(
        cm_display,
        annot=False,  # Set to True si tu veux afficher les valeurs dans chaque case (très chargé à 43 classes)
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=1,
        yticklabels=1
    )

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Classe Prédite", fontsize=11)
    plt.ylabel("Classe Réelle (Vérité Terrain)", fontsize=11)

    # Ajustement des étiquettes des axes
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graphique de la matrice de confusion sauvegardé dans {out_path}")

def save_misclassified_chart(
    misclassified_df: pd.DataFrame,
    x_raw: list[np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    out_path: Path,
    num_samples: int = 5
) -> None:
    """
    Génère et sauvegarde une figure Matplotlib montrant des erreurs de prédiction.
    :param misclassified_df: tableau des images mal classées
    :param x_raw: liste des images brutes
    :param y_true: tableau des vraies classes
    :param y_pred: tableau des classes prédites
    :param probs: tableau des probabilités prédites
    :param out_path: chemin de sauvegarde (ex: .png)
    :param num_samples: nombre d'images à afficher
    """
    n_errors = len(misclassified_df)
    if n_errors == 0:
        print("Aucune erreur de prédiction à afficher !")
        return

    # Prendre au maximum `num_samples` erreurs
    indices = misclassified_df.index[:num_samples]
    k = len(indices)

    fig, axes = plt.subplots(k, 2, figsize=(10, 3 * k))
    if k == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        img_rgb = x_raw[idx]
        true_cls = y_true[idx]
        pred_cls = y_pred[idx]
        pred_prob = probs[idx]

        # 1. Image
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(
            f"Réel: {true_cls} | Prédit: {pred_cls} ({pred_prob[pred_cls]*100:.1f}%)",
            color="darkred",
            fontweight="bold",
            fontsize=10
        )
        axes[row, 0].axis("off")

        # 2. Vecteur entier des probabilités (Barplot)
        num_classes = len(pred_prob)
        colors = ["lightgray"] * num_classes
        colors[true_cls] = "green"   # Classe réelle en vert
        colors[pred_cls] = "red"     # Classe prédite en rouge

        axes[row, 1].bar(range(num_classes), pred_prob, color=colors)
        axes[row, 1].set_xlim(-1, num_classes)
        axes[row, 1].set_ylim(0, 1.0)
        axes[row, 1].set_xlabel("Classes", fontsize=8)
        axes[row, 1].set_ylabel("Probabilité", fontsize=8)
        axes[row, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graphique des erreurs sauvegardé dans {out_path}")

def evaluate_model(model_path: Path, test_csv: Path, img_dir: Path,
                   train_dir: Path, out_dir: Path) -> None:
    """Évalue un modèle Keras sur le jeu de test et sauvegarde la matrice de confusion."""
    print(f"--- Modèle : {model_path}")
    model = tf.keras.models.load_model(model_path)

    df = pd.read_csv(test_csv)
    exists = df["Path"].map(lambda p: (img_dir / p).exists())
    if not exists.all():
        print(f"{(~exists).sum()} image(s) introuvable(s), ignorée(s)")
    df = df[exists]  # garde y_true aligné avec X_test

    mean, std = get_mean_std(train_dir)
    X_test = np.array([preprocess_single_image(img_dir / p, mean=mean, std=std) for p in df["Path"]])
    X_raw = [load_and_resize(img_dir / p) for p in df["Path"]]
    y_true = df["ClassId"].values
    print(f"Tenseur de test : {X_test.shape}")

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(f"\nAccuracy globale : {accuracy_score(y_true, y_pred) * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    np.save(out_dir / "confusion_matrix.npy", cm)

    # Sauvegarde du graphique Plot (Matplotlib/Seaborn)
    save_confusion_matrix_plot(
        cm=cm,
        out_path=out_dir / "confusion_matrix.png",
        normalize=True  # Normaliser en % est très utile quand le dataset de test est déséquilibré
    )

    # Extraction et affichage des 5 premières erreurs
    errors_mask = y_true != y_pred
    misclassified_df = df[errors_mask]

    save_misclassified_chart(
        misclassified_df=misclassified_df,
        x_raw=X_raw,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        out_path=out_dir / "misclassified_samples.png",
        num_samples=5
    )

if __name__ == "__main__":
    cfg = load_config(Path(ROOT) / "configs" / "base.yaml")
    root = Path(ROOT)

    parser = argparse.ArgumentParser(description="Évaluation du modèle")
    parser.add_argument("--model", default=cfg["model"]["type"],
                        help="Nom du modèle (dossier dans models/), ex: baseline, cnn")
    args = parser.parse_args()

    raw_dir = root / cfg["data"]["raw_dir"]
    evaluate_model(
        model_path=root / cfg["model"]["dir"] / args.model / "model.keras",
        test_csv=raw_dir / cfg["data"]["test_csv"],
        img_dir=raw_dir,
        train_dir=root / cfg["data"]["processed_dir"],
        out_dir=root / cfg["reports"]["result_dir"] / args.model,
    )