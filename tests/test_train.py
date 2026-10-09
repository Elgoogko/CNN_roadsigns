import os
from pathlib import Path
import numpy as np
import pytest
from src.trafficsigns.training.train import train_model

def test_train_model_execution(tmp_path):
    """Teste que le script d'entraînement s'exécute avec la structure de fichiers de Bari."""
    
    dummy_x_full = np.random.randint(0, 256, (4, 48, 48, 3), dtype=np.uint8)
    dummy_y_full = np.array([0, 1, 0, 1])
    
    dummy_train_idx = np.array([0, 1])
    dummy_val_idx = np.array([2, 3])

    data_dir = tmp_path / "train_data"
    data_dir.mkdir()
    
    np.save(data_dir / "img_mats.npy", dummy_x_full)
    np.save(data_dir / "labels.npy", dummy_y_full)
    np.savez(data_dir / "split.npz", train_idx=dummy_train_idx, val_idx=dummy_val_idx)
    
    dummy_model_path = tmp_path / "dummy_model.keras"
    
    history = train_model(
        data_dir=str(data_dir),
        model_save_path=str(dummy_model_path),
        epochs=1,
        batch_size=2
    )
    
    assert history is not None, "L'historique d'entraînement ne doit pas être vide"
    assert "loss" in history.history, "La métrique 'loss' doit être présente"
    assert os.path.exists(dummy_model_path), "Le fichier du modèle .keras n'a pas été sauvegardé"