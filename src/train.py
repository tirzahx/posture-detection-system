import numpy as np
from src.data_loader import load_data, split_data
from src.cnn_model import build_cnn
from src.feature_extractor import get_feature_extractor
from src.hybrid_classifier import build_hybrid_classifier
from src.evaluation import evaluate_model

DATASET_PATH = "/content/drive/MyDrive/posture_recognition"

X, y, le = load_data(DATASET_PATH)
X_train, X_test, y_train, y_test = split_data(X, y)

cnn_model = build_cnn(input_shape=(64,64,1), num_classes=len(le.classes_))

cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

loss, acc = cnn_model.evaluate(X_test, y_test)
print(f"\nCNN Test Accuracy: {acc*100:.2f}%")

cnn_preds = np.argmax(cnn_model.predict(X_test), axis=1)

feature_extractor = get_feature_extractor(cnn_model)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

y_train_int = np.argmax(y_train, axis=1)
y_test_int = np.argmax(y_test, axis=1)

hybrid = build_hybrid_classifier()
hybrid.fit(X_train_features, y_train_int)
hybrid_preds = hybrid.predict(X_test_features)

evaluate_model("CNN (Softmax)", y_test_int, cnn_preds, le.classes_)
evaluate_model("Hybrid (CNN + RF + SVM)", y_test_int, hybrid_preds, le.classes_)