import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from trafficsigns.data.preprocessing import preprocess_single_image

def evaluate_model(model_path: str, test_csv_path: str, base_img_dir: str):
    """
    Évalue un modèle Keras sur le jeu de test complet et affiche les métriques.
    :param model_path: Le chemin vers le fichier du modèle entraîné (ex: '.keras').
    :param test_csv_path: Le chemin vers le fichier CSV de test contenant les annotations 
                          (doit inclure les colonnes 'Path' et 'ClassId').
    :param base_img_dir: Le chemin vers le dossier racine contenant les images de test.
    :return: None. Affiche les métriques dans la console et sauvegarde 
             'confusion_matrix.npy' dans le répertoire courant.
    """
    print(f"--- 1. Chargement du modèle depuis {model_path} ---")
    model = tf.keras.models.load_model(model_path)
    
    print(f"--- 2. Chargement des données de test depuis {test_csv_path} ---")
    df = pd.read_csv(test_csv_path)
    y_true = df['ClassId'].values
    image_paths = df['Path'].values
    
    print("--- 3. Prétraitement des images de test ---")
    X_test = []
    
    for img_path in image_paths:
        full_path = Path(base_img_dir) / img_path
        
        if not full_path.exists():
            print(f"Image introuvable ignorée : {full_path}")
            continue
            
        img_processed = preprocess_single_image(full_path, mean=1.0, std=1.0)
        X_test.append(img_processed)
        
    X_test = np.array(X_test)
    print(f"Tenseur de test créé avec succès : {X_test.shape}")
    
    print("\n--- 4. Génération des prédictions ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n--- 5. Résultats et Métriques ---")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy Globale : {acc * 100:.2f}%\n")
    
    print("Rapport de classification détaillé :")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("Matrice de confusion :")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    np.save("confusion_matrix.npy", cm)
    print("Matrice de confusion sauvegardée sous 'confusion_matrix.npy'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'évaluation du modèle")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle entraîné (.keras)")
    parser.add_argument("--csv", type=str, default="data/raw/gtsrb/Test.csv", help="Chemin vers le CSV de test")
    parser.add_argument("--img_dir", type=str, default="data/raw/gtsrb/", help="Dossier racine des images de test")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        test_csv_path=args.csv,
        base_img_dir=args.img_dir
    )