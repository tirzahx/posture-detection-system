import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data(dataset_path, img_size=(64, 64)):
    images, labels = [], []
    for folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = load_img(img_path, target_size=img_size, color_mode="grayscale")
                img = img_to_array(img) / 255.0
                images.append(img)
                labels.append(folder)
            except:
                pass
    X = np.array(images)
    y = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, len(le.classes_))

    return X, y, le

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
