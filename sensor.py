import time
import random
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate("firebaseFirestore.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://mango-monitoring-535a1-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Reference to the Firebase Realtime Database
data_ref = db.reference('sensor_data')

def get_sensor_data():
    """
    Simulate sensor data retrieval.
    Replace these with actual sensor readings.
    """
    temperature = round(random.uniform(20.0, 35.0), 2)  # Simulated temperature in Celsius
    humidity = round(random.uniform(30.0, 70.0), 2)  # Simulated humidity in percentage
    soil_moisture = round(random.uniform(200.0, 800.0), 2)  # Simulated soil moisture level
    return temperature, humidity, soil_moisture

def send_data_to_firebase():
    while True:
        # Get sensor data
        temperature, humidity, soil_moisture = get_sensor_data()
        
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create data payload
        data = {
            'temperature': temperature,
            'humidity': humidity,
            'soil_moisture': soil_moisture,
            'timestamp': timestamp
        }

        # Push data to Firebase
        data_ref.push(data)
        print(f"Data sent to Firebase: {data}")

        # Wait before sending the next set of data
        time.sleep(10)  # Send data every 10 seconds

if __name__ == "__main__":
    send_data_to_firebase()
