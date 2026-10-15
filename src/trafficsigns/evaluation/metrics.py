import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from trafficsigns.config import ROOT, load_config
from trafficsigns.data.preprocessing import preprocess_single_image
from trafficsigns.utils.compute_parameters import get_mean_std


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
    y_true = df["ClassId"].values
    print(f"Tenseur de test : {X_test.shape}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(f"\nAccuracy globale : {accuracy_score(y_true, y_pred) * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "confusion_matrix.npy", cm)
    print(f"Matrice de confusion sauvegardée dans {out_dir / 'confusion_matrix.npy'}")


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