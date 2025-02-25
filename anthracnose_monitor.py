import os
# Set the environment variable
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
#Use in the terminal -- set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
from pathlib import Path
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
import requests
import firebase_admin
from firebase_admin import credentials, firestore 
from tensorflow.keras.preprocessing import image 
import numpy as np
import cv2
from datetime import datetime, timedelta
import shutil

# Initialize Firebase
cred = credentials.Certificate("firebaseFirestore.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()

# Initialize Flask app
app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="mango1.tflite")
interpreter.allocate_tensors()
interpreter1 = tf.lite.Interpreter(model_path="bestMangoModel.tflite")
interpreter1.allocate_tensors()

# Path to the uploaded images folder
UPLOAD_FOLDER = 'static/analysis'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# Ensure the 'images' folder exists in the 'static' directory
if not os.path.exists('static/images'):
    os.makedirs('static/images')
    
# Firebase Realtime Database URL
FIREBASE_URL = "https://mango-monitoring-535a1-default-rtdb.asia-southeast1.firebasedatabase.app/EnvironmentData.json"
@app.after_request
def set_headers(response):
    response.headers["Permissions-Policy"] = "interest-cohort=()"
    return response

def fetch_latest_sensor_data():
    """Fetch the latest temperature, humidity, and soil moisture data from Firebase."""
    response = requests.get(FIREBASE_URL, params={"orderBy": '"$key"', "limitToLast": 1})
    if response.status_code == 200:
        data = response.json()
        if data:
            latest_key = list(data.keys())[0]
            latest_entry = data[latest_key]
            return latest_entry
    return None

def fetch_historical_sensor_data(limit=20):
    """Fetch historical data for the graph."""
    response = requests.get(FIREBASE_URL, params={"orderBy": '"$key"', "limitToLast": limit})
    if response.status_code == 200:
        data = response.json()
        if data:
            return data
    return None

@app.route("/sensor_cards")
def sensor_cards():
    # Fetch the latest data for the cards
    latest_data = fetch_latest_sensor_data()
    if latest_data:
        temperature = latest_data.get("Temperature", "N/A")
        humidity = latest_data.get("Humidity", "N/A")
        soil_moisture = latest_data.get("SoilMoisture", "N/A")
    else:
        temperature = humidity = soil_moisture = "N/A"

    # Fetch historical data for the graph
    historical_data = fetch_historical_sensor_data()
    if historical_data:
        labels = []
        temperature_data = []
        humidity_data = []
        soil_moisture_data = []
        for key, entry in historical_data.items():
            labels.append(entry.get("Timestamp", "N/A"))
            temperature_data.append(entry.get("Temperature", 0))
            humidity_data.append(entry.get("Humidity", 0))
            soil_moisture_data.append(entry.get("SoilMoisture", 0))
    else:
        labels = temperature_data = humidity_data = soil_moisture_data = []

    return render_template(
        "sensor_cards.html",
        temperature=temperature,
        humidity=humidity,
        soil_moisture=soil_moisture,
        labels=labels,
        temperature_data=temperature_data,
        humidity_data=humidity_data,
        soil_moisture_data=soil_moisture_data,
    )
    
@app.route("/latest-timestamp")
def latest_timestamp():
    """Return the latest timestamp from Firebase."""
    latest_data = fetch_latest_data() # type: ignore
    if latest_data:
        return jsonify({"timestamp": latest_data.get("Timestamp", "")})
    return jsonify({"timestamp": ""})

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')
@app.route('/sensors_data')
def sensors_data():
    return render_template('sensor_graph.html')
@app.route('/analyze_data_image')
def analysis_data_image():
    return render_template('analyze_data_image.html')

@app.route('/get-analysis-custom', methods=['GET'])
def get_analysis_custom():
    try:
        # Get start and end dates from query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)  # Include the end date

        # Fetch data from Firestore where timestamp is between start_date and end_date
        analysis_docs = db.collection('analyzed_data') \
            .where('timestamp', '>=', start_date) \
            .where('timestamp', '<', end_date) \
            .stream()

        analysis_data = []
        for doc in analysis_docs:
            data = doc.to_dict()
            image_url = url_for('static', filename='analysis/' + os.path.basename(data['image_url']))
            analysis_data.append({
                'confidence': data.get('confidence'),
                'disease': data.get('disease'),
                'category': data.get('category'),
                'image_url': image_url,
                'timestamp': data.get('timestamp'),
            })

        return jsonify({
            'message': 'Analysis data for custom date range fetched successfully',
            'data': analysis_data
        }), 200

    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500
    
@app.route('/get-analysis-day', methods=['GET'])
def get_analysis_day():
    try:
        # Get the 'date_date' query parameter (in YYYY-MM-DD format)
        date_date_str = request.args.get('day_date')

        # If 'date_date' is provided, parse it; otherwise, default to today
        if date_date_str:
            start_of_day = datetime.strptime(date_date_str, '%Y-%m-%d')
        else:
            now = datetime.now()
            start_of_day = datetime(now.year, now.month, now.day)

        # Calculate the end of the day for the selected date
        end_of_day = start_of_day + timedelta(days=1)

        # Fetch data from Firestore where timestamp is between start_of_day and end_of_day
        analysis_docs = db.collection('analyzed_data') \
            .where('timestamp', '>=', start_of_day) \
            .where('timestamp', '<', end_of_day) \
            .stream()

        analysis_data = []
        for doc in analysis_docs:
            data = doc.to_dict()
            image_url = url_for('static', filename='analysis/' + os.path.basename(data['image_url']))
            analysis_data.append({
                'confidence': data.get('confidence'),
                'disease': data.get('disease'),
                'category': data.get('category'),
                'image_url': image_url,
                'timestamp': data.get('timestamp'),
            })

        return jsonify({
            'message': f'Analysis data for {start_of_day.strftime("%Y-%m-%d")} fetched successfully',
            'data': analysis_data
        }), 200

    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500



@app.route('/get-analysis-week', methods=['GET'])
def get_analysis_week():
    try:
        # Get current date and the start of the week (Monday)
        now = datetime.now()
        start_of_week = now - timedelta(days=now.weekday())  # Monday of the current week

        # Fetch data from Firestore where timestamp is after start_of_week
        analysis_docs = db.collection('analyzed_data') \
            .where('timestamp', '>=', start_of_week) \
            .stream()

        analysis_data = []
        for doc in analysis_docs:
            data = doc.to_dict()
            image_url = url_for('static', filename='analysis/' + os.path.basename(data['image_url']))
            analysis_data.append({
                'confidence': data.get('confidence'),
                'disease': data.get('disease'),
                'category': data.get('category'),
                'image_url': image_url,
                'timestamp': data.get('timestamp'),
            })

        return jsonify({
            'message': 'Analysis data for this week fetched successfully',
            'data': analysis_data
        }), 200

    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

@app.route('/get-analysis-month', methods=['GET'])
def get_analysis_month():
    try:
        # Get current date and the start of the month
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1)

        # Fetch data from Firestore where timestamp is after start_of_month
        analysis_docs = db.collection('analyzed_data') \
            .where('timestamp', '>=', start_of_month) \
            .stream()

        analysis_data = []
        for doc in analysis_docs:
            data = doc.to_dict()
            image_url = url_for('static', filename='analysis/' + os.path.basename(data['image_url']))
            analysis_data.append({
                'confidence': data.get('confidence'),
                'disease': data.get('disease'),
                'category': data.get('category'),
                'image_url': image_url,
                'timestamp': data.get('timestamp'),
            })

        return jsonify({
            'message': 'Analysis data for this month fetched successfully',
            'data': analysis_data
        }), 200

    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

@app.route('/get-analysis-year', methods=['GET'])
def get_analysis_year():
    try:
        # Get current date and the start of the year
        now = datetime.now()
        start_of_year = datetime(now.year, 1, 1)

        # Fetch data from Firestore where timestamp is after start_of_year
        analysis_docs = db.collection('analyzed_data') \
            .where('timestamp', '>=', start_of_year) \
            .stream()

        analysis_data = []
        for doc in analysis_docs:
            data = doc.to_dict()
            image_url = url_for('static', filename='analysis/' + os.path.basename(data['image_url']))
            analysis_data.append({
                'confidence': data.get('confidence'),
                'disease': data.get('disease'),
                'category': data.get('category'),
                'image_url': image_url,
                'timestamp': data.get('timestamp'),
            })

        return jsonify({
            'message': 'Analysis data for this year fetched successfully',
            'data': analysis_data
        }), 200

    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500




@app.route('/get-analysis-graph', methods=['GET'])
def get_analysis_graph():
    try:
        # Fetch data from Firestore
        analysis_docs = db.collection('analyzed_data').stream()

        # List to store filtered data
        analysis_data = []

        # Check each document for image availability and collect data
        for doc in analysis_docs:
            data = doc.to_dict()

            # Check if image_url points to an existing file in static/analysis
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(data['image_url']))
            if os.path.exists(image_path):
                # Using url_for to generate the static URL
                image_url = url_for('static', filename='analysis/' + os.path.basename(data['image_url']))
                analysis_data.append({
                    'confidence': data.get('confidence'),
                    'disease': data.get('disease'),
                    'category': data.get('category'),
                    'image_url': image_url,
                    'timestamp': data.get('timestamp'),
                })

        # Sort data by timestamp (most recent first)
        analysis_data.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'message': 'Analysis data fetched successfully',
            'data': analysis_data
        }), 200

    except Exception as e:
        # Log and return error message
        print(f"Error fetching analysis data: {e}")
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500


# Function to send data to Firebase Firestore
def send_analyze_data(db, label, confidence, category, image_url):
    try:
        # Ensure confidence is a native Python float
        confidence_value = float(confidence)

        # Add data to Firestore
        doc_ref = db.collection('analyzed_data').add({
            'disease': label,
            'confidence': confidence_value,
            'category': category,
            'image_url': image_url,
            'timestamp': firestore.SERVER_TIMESTAMP,  # Server-generated timestamp
        })

        # Success response
        return jsonify({
            'message': 'Data submitted successfully',
            'document_id': doc_ref[1].id  # Return the ID of the newly created document
        }), 200

    except Exception as e:
        # Handle and log errors
        print(f"Error submitting data: {e}")  # Use a logger in production
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

# Function to preprocess and predict the image using TensorFlow Lite
def predict_image(img_path):
    # Load the image for the first model (224x224)
    img1 = image.load_img(img_path, target_size=(260, 260))  
    img_array1 = image.img_to_array(img1) / 255.0  # Normalize
    img_array1 = np.expand_dims(img_array1, axis=0)  # Add batch dimension

    # Get input details for the first model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Verify input shape dynamically
    expected_shape = tuple(input_details[0]['shape'][1:3])
    if expected_shape != (260, 260):
        raise ValueError(f"Expected input shape {expected_shape}, but got (260, 260).")

    # Set input tensor for the first model
    interpreter.set_tensor(input_details[0]['index'], img_array1)

    # Run inference
    interpreter.invoke()

    # Get prediction result
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Determine if it's Anthracnose or Healthy
    if output_data.shape[-1] > 1:  # Multi-class output
        confidence = np.max(output_data)
        class_index = np.argmax(output_data)
        label = "Anthracnose" if class_index == 0 else "Healthy"
    else:  # Binary classification output
        confidence = float(output_data[0][0])
        label = "Anthracnose" if confidence > 0.5 else "Healthy"
        confidence = confidence if confidence > 0.5 else 1 - confidence

    if label == "Healthy":
        return label, confidence, "CATEGORY 0"

    # If Anthracnose is detected, use the second model (260x260)
    class_labels = [
        'CATEGORY 0',  # Index 0
        'CATEGORY 1',  # Index 1
        'CATEGORY 2',  # Index 2
        'CATEGORY 3',  # Index 3
        'CATEGORY 4'   # Index 4
    ]

    # Load the image again for the second model (260x260)
    img2 = image.load_img(img_path, target_size=(260, 260))  
    img_array2 = image.img_to_array(img2) / 255.0  # Normalize
    img_array2 = np.expand_dims(img_array2, axis=0)  # Add batch dimension

    # Get input details for the second model
    input_details1 = interpreter1.get_input_details()
    output_details1 = interpreter1.get_output_details()

    # Verify input shape dynamically
    expected_shape1 = tuple(input_details1[0]['shape'][1:3])
    if expected_shape1 != (260, 260):
        raise ValueError(f"Expected input shape {expected_shape1}, but got (260, 260).")

    # Set input tensor for the second model
    interpreter1.set_tensor(input_details1[0]['index'], img_array2)

    # Run inference
    interpreter1.invoke()

    # Get prediction result
    output_data1 = interpreter1.get_tensor(output_details1[0]['index'])

    # Ensure output shape matches the number of classes
    if output_data1.shape[-1] != len(class_labels):
        raise ValueError(f"Expected {len(class_labels)} classes but got {output_data1.shape[-1]}.")

    # Find the class with the highest confidence score
    class_index1 = np.argmax(output_data1)
    category = class_labels[class_index1]
    
    # Modified condition: If Anthracnose but Category 0, change to Healthy
    if label == "Anthracnose" and category == "CATEGORY 0":
        label = "Healthy"
        confidence = 1 - confidence  # Invert confidence measure
        category = "CATEGORY 0"

    return label, confidence, category


# Route for the main page
@app.route('/')
def home():
    return render_template('sensor_cards.html')

# Route for the upload image page
@app.route('/upload')
def upload_image():
    return render_template('upload.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the file to the upload folder
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    print(img_path)
    img_path1 = 'analysis/' + file.filename
    # Make prediction using TensorFlow Lite
    label, confidence, category = predict_image(img_path)

    # Set severity category
    severity_category = category if label == "Anthracnose" else "CATEGORY 0"
    
    send_analyze_data(db, label, confidence, severity_category, img_path1)
    
    # Return the prediction result as a JSON response
    return jsonify({
        'label': label,
        'confidence': f'{confidence*100:.2f}%',
        'category': severity_category,
        'image_url': img_path
    })

# Global camera object
camera = None
# Route to show the camera feed and capture photo
@app.route('/cam')
def cam():
    return render_template('cam.html')

# Function to initialize the camera
def initialize_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

# Function to release the camera
def release_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None

# Function to generate the camera feed
def generate_camera_feed():
    initialize_camera()
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Convert the frame to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            # Yield the frame in the correct format for MJPEG streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        release_camera()  # Ensure the camera is released when done

# Route to stream the camera feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to take a photo (without saving)
@app.route('/take_photo')
def take_photo():
    initialize_camera()
    ret, frame = camera.read()
    photo_filename = None
    if ret:
        # Generate a unique filename using the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_filename = f"{timestamp}.jpg"
        
        # Path to store the temporary photo for preview
        photo_path = os.path.join('static', 'images', photo_filename)
        
        # Save the temporary photo for preview
        cv2.imwrite(photo_path, frame)
    release_camera()  # Release the camera after capturing
    
    if photo_filename:
        return render_template('take_photo.html', photo_path=f'images/{photo_filename}', photo_filename=photo_filename)
    return redirect(url_for('cam'))  # Redirect if photo capture fails

# Route to actually save the photo (after confirmation)
@app.route('/save_photo')
def save_photo():
    photo_filename = request.args.get('photo_filename')  # Get the filename passed from the template
    if not photo_filename:
        return redirect(url_for('home'))  # Redirect if no filename is provided

    # Paths for source and target
    source_path = os.path.join('static', 'images', photo_filename)
    target_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)

    # Check if the source file exists in 'static/images'
    if os.path.exists(source_path):
        # Ensure the 'static/analysis' folder exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Copy the file from 'images' to 'analysis'
        shutil.copy(source_path, target_path)

        # Remove all files from 'static/images'
        images_path = os.path.join('static', 'images')
        for filename in os.listdir(images_path):
            file_path = os.path.join(images_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # Make prediction using TensorFlow Lite
        label, confidence, category = predict_image(target_path)
        # Set severity category
        severity_category = category if label == "Anthracnose" else "CATEGORY 0"
        image_url = f'analysis/{photo_filename}'
        send_analyze_data(db, label, confidence, severity_category, image_url)
        # Prepare variables for rendering
        confidence_percentage = f'{confidence * 100:.2f}%'
        return render_template(
            'save_photo.html',
            label=label,
            confidence=confidence_percentage,
            category=severity_category,
            image_url=image_url
        )

    # If the photo doesn't exist, redirect back to the home page
    return redirect(url_for('home'))

# Route to reset and retake the photo
@app.route('/retake_photo')
def retake_photo():
    # Get the filename of the previous photo to delete
    previous_photo_filename = request.args.get('photo_filename')  # Get the filename passed from the template
    
    if previous_photo_filename:
        # Remove the previous file from the 'static/images' folder
        previous_photo_path = os.path.join('static', 'images', previous_photo_filename)
        if os.path.exists(previous_photo_path):
            os.remove(previous_photo_path)
    
    return redirect(url_for('cam'))  # Redirect to the camera page

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
