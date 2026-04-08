import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def build_image_model():

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # 🔥 Freeze most layers, train top
    for layer in base.layers[:-30]:
        layer.trainable = False

    for layer in base.layers[-30:]:
        layer.trainable = True

    x = base.output
    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)

    outputs = Dense(3, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model