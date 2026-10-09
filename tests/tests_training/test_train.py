import os
import numpy as np
import pytest
from src.trafficsigns.training.train import train_model

def test_train_model_execution(tmp_path):
    """Teste que le script d'entraînement s'exécute de bout en bout sans erreur."""
    
    dummy_x_train = np.random.rand(2, 32, 32, 3).astype(np.float32)
    dummy_y_train = np.array([0, 1])
    dummy_x_val = np.random.rand(2, 32, 32, 3).astype(np.float32)
    dummy_y_val = np.array([1, 0])
    

    dummy_data_path = tmp_path / "dummy_dataset.npz"
    dummy_model_path = tmp_path / "dummy_model.keras"
    
    np.savez(
        dummy_data_path, 
        X_train=dummy_x_train, y_train=dummy_y_train, 
        X_val=dummy_x_val, y_val=dummy_y_val
    )
    
    # 1 seule epoch et batch_size=2 pour que le test soit rapide
    history = train_model(
        data_path=str(dummy_data_path),
        model_save_path=str(dummy_model_path),
        epochs=1,
        batch_size=2
    )
    
    assert history is not None, "L'historique d'entraînement ne doit pas être vide"
    assert "loss" in history.history, "La métrique 'loss' doit être présente"
    assert os.path.exists(dummy_model_path), "Le fichier du modèle .keras n'a pas été sauvegardé"