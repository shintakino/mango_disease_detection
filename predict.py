import os
# Set the environment variable
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


def predict_image(img_path, interpreter):
    """
    Predict the class of the image using the TensorFlow Lite model.

    Parameters:
    - img_path (str): Path to the image to be analyzed.
    - interpreter (tf.lite.Interpreter): Loaded TensorFlow Lite model interpreter.

    Returns:
    - label (str): Predicted class label ('Healthy' or 'Anthracnose').
    - confidence (float): Confidence score of the prediction.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to (224, 224)
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get input and output details from the TFLite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the prediction result
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Check if the model outputs a single value or an array
    if output_data.shape[-1] > 1:  # Multi-class output
        confidence = np.max(output_data)  # Highest confidence score
        class_index = np.argmax(output_data)  # Index of the predicted class
        label = "Anthracnose" if class_index == 0 else "Healthy"
    else:  # Binary output (e.g., single sigmoid output)
        confidence = float(output_data[0])  # Convert the first element to float
        label = "Anthracnose" if confidence > 0.5 else "Healthy"
        confidence = confidence if confidence > 0.5 else 1 - confidence

    return label, confidence

def main():
    # Load the TensorFlow Lite model
    model_path = "test1.tflite"  # Update with the actual path to your TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Path to the image file
    img_path = "20241201_200059.jpg"  # Update with the path to the test image

    # Call the predict_image function
    label, confidence = predict_image(img_path, interpreter)

    # Print the result
    print(f"Predicted label: {label}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
