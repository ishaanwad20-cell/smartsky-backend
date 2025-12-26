import tensorflow as tf
import os
import shutil

MODEL = "mars_lstm.h5"
OUT = "saved_model"

custom_objects = {
    "mse": tf.keras.losses.MeanSquaredError()
}

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL, custom_objects=custom_objects, compile=False)

if os.path.exists(OUT):
    shutil.rmtree(OUT)

print("Exporting to SavedModel...")
tf.saved_model.save(model, OUT)

print("SavedModel created →", OUT)
