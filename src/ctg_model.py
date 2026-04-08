from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_ctg_model():

    inp = Input(shape=(21,))

    x = Dense(128, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation="relu")(x)

    out = Dense(3, activation="softmax")(x)

    model = Model(inp, out)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model