import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

from trafficsigns.config import ROOT
from trafficsigns.data.preprocessing import normalize, preprocess_single_image, resize_img, resize_img
from trafficsigns.utils.compute_parameters import load_mean_std

def detect_and_classify(image_path, yolo_model_path, cnn_model_path, stats_path):
    """
    Pipeline complet : Détection (YOLO) -> Recadrage -> Classification (CNN)
    """
    print("1. Chargement des modèles...")
    yolo_detector = YOLO(yolo_model_path) 
    
    cnn_classifier = keras.models.load_model(cnn_model_path)
    mean, std = load_mean_std(stats_path)

    print("2. Chargement de l'image de la scène...")
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Keras travaille en RGB
    
    img_result = img_rgb.copy()

    print("3. YOLO : Détection des panneaux...")
    results = yolo_detector(img_rgb)
    
    boxes = results[0].boxes

    print(f"-> {len(boxes)} objet(s) détecté(s) par YOLO.")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        conf = float(box.conf[0])
        if conf < 0.5:
            continue

        cropped_sign = img_rgb[y1:y2, x1:x2]

        if cropped_sign.size == 0:
            continue

        processed_sign = resize_img(cropped_sign, (48, 48))
        processed_sign = normalize(processed_sign, mean, std)
        
        input_tensor = np.expand_dims(processed_sign, axis=0)

        predictions = cnn_classifier.predict(input_tensor, verbose=0)
        predicted_class = np.argmax(predictions[0])
        cnn_confidence = np.max(predictions[0])

        cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        label = f"Classe {predicted_class} ({cnn_confidence:.2f})"
        
        cv2.putText(img_result, label, (x1, max(20, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_result)
    plt.axis('off')
    plt.title("Résultat du Pipeline (YOLO + CNN)")
    plt.show()

if __name__ == "__main__":
    scene_image = ROOT / "data" / "raw" / "test_scene.jpg"
    yolo_model = "yolov8n.pt"
    cnn_model = ROOT / "models" / "test" / "model.keras"
    norm_stats = ROOT / "models" / "test" / "norm_stats.npz"

    detect_and_classify(scene_image, yolo_model, cnn_model, norm_stats)