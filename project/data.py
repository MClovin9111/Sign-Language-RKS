import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 64

# Mapeo fijo (NO CAMBIAR)
LABEL_MAP = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
    'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
    'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
    'Z': 25
}

def load_images_from_folder(folder_path):
    X = []
    y = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)

        if not os.path.isdir(label_path):
            continue

        if label not in LABEL_MAP:
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            # leer imagen
            img = cv2.imread(img_path)

            if img is None:
                continue

            # convertir a gris
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # redimensionar a 64x64
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # normalizar 0-1
            img = img / 255.0

            # reshape (64,64,1)
            img = np.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, 1))

            X.append(img)
            y.append(LABEL_MAP[label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y


def load_data():
    # CAMBIAr dataset
    dataset_path = "dataset"

    X, y = load_images_from_folder(dataset_path)

    # dividir train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test
