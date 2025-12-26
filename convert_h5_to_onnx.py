import tensorflow as tf
import tf2onnx
import os
import shutil

h5_path = "mars_lstm.h5"
saved_model_dir = "tmp_saved_model"
onnx_path = "mars_lstm.onnx"

# allow old Keras loss to load
custom_objects = {
    "mse": tf.keras.losses.MeanSquaredError()
}

print("Loading model...")
model = tf.keras.models.load_model(h5_path, custom_objects=custom_objects, compile=False)

# Clean previous temp folder
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

print("Exporting to SavedModel...")
tf.saved_model.save(model, saved_model_dir)

print("Converting SavedModel → ONNX...")
tf2onnx.convert.from_saved_model(
    saved_model_dir,
    opset=13,
    output_path=onnx_path
)

print("SUCCESS — created", onnx_path)
