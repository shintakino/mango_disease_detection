import os
# Set the environment variable
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
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
interpreter = tf.lite.Interpreter(model_path="bestMangoModel.tflite")
interpreter.allocate_tensors()

# Path to the uploaded images folder
UPLOAD_FOLDER = 'static/analysis'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# Ensure the 'images' folder exists in the 'static' directory
if not os.path.exists('static/images'):
    os.makedirs('static/images')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

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
def send_analyze_data(db, label, confidence, image_url):
    try:
        # Ensure confidence is a native Python float
        confidence_value = float(confidence)

        # Add data to Firestore
        doc_ref = db.collection('analyzed_data').add({
            'disease': label,
            'confidence': confidence_value,
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


# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

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
    label, confidence = predict_image(img_path)
    send_analyze_data(db, label, confidence, img_path1)
    # Return the prediction result as a JSON response
    return jsonify({
        'label': label,
        'confidence': f'{confidence*100:.2f}%',
        'image_url': img_path
    })

# Route to show the camera feed and capture photo
@app.route('/cam')
def cam():
    return render_template('cam.html')

# Function to initialize the camera
def initialize_camera():
    return cv2.VideoCapture(1)

# Function to release the camera
def release_camera(cam):
    if cam.isOpened():
        cam.release()

# Function to generate the camera feed
def generate_camera_feed():
    camera = initialize_camera()
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
        release_camera(camera)  # Ensure the camera is released when done

# Route to stream the camera feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to take a photo (without saving)
@app.route('/take_photo')
def take_photo():
    camera = initialize_camera()
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
    release_camera(camera)  # Release the camera after capturing
    
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
        label, confidence = predict_image(target_path)
        image_url = f'analysis/{photo_filename}'
        send_analyze_data(db, label, confidence, image_url)
        # Prepare variables for rendering
        confidence_percentage = f'{confidence * 100:.2f}%'
        return render_template(
            'save_photo.html',
            label=label,
            confidence=confidence_percentage,
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
    app.run(debug=True, host='192.168.100.6', port=5000)
