from tensorflow.keras.models import Model

def get_feature_extractor(cnn_model):
    return Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("feature_layer").output)
