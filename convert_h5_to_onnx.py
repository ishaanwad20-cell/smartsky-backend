import tensorflow as tf
import tf2onnx
import onnx

h5_path = "mars_lstm.h5"        # your trained Keras model file
onnx_path = "mars_lstm.onnx"    # output ONNX file
# Example input shape: (None, WINDOW, 1) where WINDOW=7
# Replace 7 with your WINDOW value if different.
spec = (tf.TensorSpec((None, 7, 1), tf.float32, name="input"),)

model = tf.keras.models.load_model(h5_path)
# convert
model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=onnx_path
)
print("Saved ONNX ->", onnx_path)

# quick sanity check
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")
