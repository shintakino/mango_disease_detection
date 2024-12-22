import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('mango_disease_model.h5')

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model as a .tflite file
with open('mango_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
