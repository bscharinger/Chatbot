import tensorflow as tf
from tensorflow import keras
from keras import layers

def get_model(input_size, output_size):
    inputs = keras.Input(shape=input_size)
    d1 = layers.Dense(128, activation='relu')(inputs)
    drop1 = layers.Dropout(0.5)(d1)

    d2 = layers.Dense(64, activation='relu')(drop1)
    drop2 = layers.Dropout(0.5)(d2)
    out = layers.Dense(output_size, activation='softmax')(drop2)

    model = keras.Model(inputs, out)
    return model




