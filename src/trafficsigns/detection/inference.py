from pathlib import Path
import keras
import numpy as np
from trafficsigns.data.preprocessing import preprocess_single_image
from trafficsigns.utils.compute_parameters import load_mean_std

class Inference:
    """
    Classe d'inférence modulaire pour prédire sur une ou plusieurs images.
    """
    def __init__(self, model_path: str | Path, norm_stats_path: str | Path):
        model_path = Path(model_path)
        assert model_path.suffix == ".keras", f"Le fichier doit être un .keras (reçu: {model_path.suffix})"

        self.model = keras.models.load_model(model_path)
        # On utilise le paramètre norm_stats_path passé à l'initialisation
        self.mean, self.std = load_mean_std(norm_stats_path)

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """Effectue une passe avant sur un tensor (N, H, W, C)."""
        return self.model.predict(input_tensor, verbose=0)

    def apply_argmax(self, output_tensor: np.ndarray) -> np.ndarray:
        """Convertit les probabilités softmax en indices de classes."""
        return np.argmax(output_tensor, axis=-1)

    def confidence(self, predictions: np.ndarray, predicted_classes: np.ndarray,
                   img_id: int | None = None) -> float | np.ndarray:
        """
        Calcule la confiance de la classe prédite.
        Si img_id est spécifié, retourne un float. Sinon, retourne un tableau de floats.
        """
        if img_id is not None:
            return float(predictions[img_id, predicted_classes[img_id]])

        # Extraction vectorisée des confiances pour tout le lot
        batch_indices = np.arange(len(predictions))
        return predictions[batch_indices, predicted_classes]

    def predict_single(self, img_path: str | Path) -> tuple[int, float]:
        """
        Prédit la classe et la confiance pour une seule image.
        :return: (classe_prédite, indice_de_confiance)
        """
        img_array = preprocess_single_image(img_path, mean=self.mean, std=self.std)
        input_tensor = np.expand_dims(img_array, axis=0)  # Shape (1, H, W, C)

        predictions = self.predict(input_tensor)
        predicted_class = int(self.apply_argmax(predictions)[0])
        conf = float(predictions[0, predicted_class])

        return predicted_class, conf

    def predict_multiple(self, img_paths: list[str | Path]) :
        """
        Prédit les classes et confiances pour une liste d'images.
        :return: (tableau_des_classes, tableau_des_confiances)
        """
        # Utilise la vectorisation (si preprocess_multiple_images existe dans ton module)
        # Sinon, empile les images prétraitées individuellement
        images = [preprocess_single_image(p, mean=self.mean, std=self.std) for p in img_paths]
        input_tensor = np.array(images)  # Shape (N, H, W, C)

        predictions = self.predict(input_tensor)
        predicted_classes = self.apply_argmax(predictions)
        confidences = self.confidence(predictions, predicted_classes)

        return predicted_classes, confidences


if __name__ == "__main__":
    # Test de démonstration
    from trafficsigns.config import ROOT  # Import de ROOT selon ta structure

    mdl_path = ROOT / "models" / "baseline" / "MLP_v1.keras"
    stats_path = ROOT / "models" / "baseline" / "norm_stats.npz"

    engine = Inference(model_path=mdl_path, norm_stats_path=stats_path)

    # Test sur une liste d'images
    test_dir = ROOT / "data" / "raw" / "gtsrb" / "Test"
    sample_imgs = [test_dir / f"0000{i}.png" for i in range(4)]

    classes, confs = engine.predict_multiple(sample_imgs)

    for path, cls, conf in zip(sample_imgs, classes, confs):
        print(f"Image: {path.name} -> Classe : {cls} (Confiance : {conf * 100:.2f}%)")