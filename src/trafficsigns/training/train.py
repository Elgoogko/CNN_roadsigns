import argparse
from trafficsigns.data.preprocessing import normalize, NUM_CLASSES
from trafficsigns.config import load_config, ROOT, MODEL_REGISTRY
from trafficsigns.utils.compute_parameters import get_mean_std
from trafficsigns.data.dataset import load_dataset, load_split
from pathlib import Path
import numpy as np
import pandas as pd

def train_model(data_dir: str, model_save_path: str, epochs: int, batch_size: int, model_to_train: str):
    """
    Charge les données, normalise, entraîne le modèle et sauvegarde les poids.
    """
    assert model_to_train in MODEL_REGISTRY.keys()

    print(f"--- 1. Chargement des données depuis le dossier {data_dir} ---")
    x, y, _ = load_dataset(data_dir, mmap=True)
    train_idx, val_idx = load_split(data_dir)
    
    print("--- 2. Prétraitement (Normalisation) ---")
    mean, std = get_mean_std(data_dir)  # calculées sur le TRAIN uniquement
    x_train, y_train = normalize(x[train_idx], mean, std), y[train_idx]
    x_val, y_val = normalize(x[val_idx], mean, std), y[val_idx]  # mêmes stats

    print(f"Images d'entraînement : {x_train.shape}")
    print(f"Images de validation  : {x_val.shape}")

    print("\n--- 3. Construction du modèle ---")
    input_shape = x_train.shape[1:]
    num_classes = NUM_CLASSES
    
    model = MODEL_REGISTRY[model_to_train](input_shape=input_shape, num_classes=num_classes)
    model.summary()

    print("\n--- 4. Début de l'entraînement ---")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    print(f"\n--- 5. Sauvegarde du modèle ---")

    df_history = pd.DataFrame(history.history)
    df_history.index.name = "epoch"

    df_history.to_csv(ROOT / "failsafe" / "history_epochs.csv")

    model.save(Path.joinpath(ROOT, "failsafe", "model.keras"))
    np.savez(Path.joinpath(ROOT, "failsafe", "norm_stats.npz"), mean=mean, std=std) #Sert lors de l'inférence

    Path.mkdir(Path(model_save_path), exist_ok=True)

    df_history.to_csv(model_save_path / "history_epochs.csv")
    print(f"Historique par époque sauvegardé dans {model_save_path / 'history_epochs.csv'}")

    model.save(Path.joinpath(Path(model_save_path), "model.keras"))
    np.savez(Path.joinpath(Path(model_save_path), "norm_stats.npz"), mean=mean, std=std) #Sert lors de l'inférence

    print(f"Modèle sauvegardé avec succès sous : {model_save_path}")

    return history

if __name__ == "__main__":

    cfg_path = Path.joinpath(ROOT, "configs", "base.yaml")
    cfg = load_config(cfg_path) if cfg_path.exists() else {}

    # Extraire les sous-sections du YAML avec valeurs de secours (fallbacks)
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    # Configurer le parser avec les valeurs du YAML comme 'default'
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle")
    parser.add_argument("--model", type=str, default=model_cfg.get("type", "baseline"), help="Modèle à entraîner")
    parser.add_argument("--data_dir", type=str, default=train_cfg.get("data_dir", "data/processed/train"),
                        help="Dossier des données")
    parser.add_argument("--model_out", type=str, default=train_cfg.get("model_dir", "models/new_model"),
                        help="Chemin de sauvegarde")
    parser.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 15), help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=train_cfg.get("batch_size", 32), help="Taille de batch")

    args = parser.parse_args()

    train_model(
        model_to_train=args.model,
        data_dir=args.data_dir,
        model_save_path=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size
    )