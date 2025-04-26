# Import necessary modules
from flask import Flask, request, render_template, send_file, jsonify, redirect
from udp_server import start_udp_server, get_latest_result, launch_udp_server  # UDP logic handled separately
from pipeline import process_csv, real_time_prediction, upload_csv  # CSV processing function
import threading
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


os.makedirs('uploads', exist_ok=True)
os.makedirs('processed', exist_ok=True)
scaler = None

# Initialize the Flask web app
app = Flask(__name__)


scaler_ready = False
udp_thread_started = False
# === Route: Scaler Setup Page ===
# Inside app.py

@app.route('/setup', methods=['POST', 'GET'])
def setup():
    global scaler_ready, scaler, udp_thread_started
    scaler_ready = False
    if request.method == 'POST':
        file = request.files['file']
        if not file or not file.filename.endswith('.csv'):
            return "Invalid file", 400
        
        label = request.form.get("label", "default")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"uploaded_{label}_{timestamp}.csv"
        
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        df = process_csv(df)

        scaler_ready = True
        if not udp_thread_started:
            launch_udp_server()

        return redirect('/')

    return render_template('setup.html')

@app.route('/')
def index():
    return render_template('index.html')  # Corrected file name
  # HTML page that fetches and displays the latest result

# === Route to get the latest result from the model ===
# JavaScript on the page calls this every few milliseconds to update the display
@app.route('/get-latest')
def get_latest():
    return jsonify({'result': get_latest_result()})  # Return latest model output in JSON

# === Route for uploading and processing a CSV file ===
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    # If the user submits the form
    if request.method == 'POST':
        file = request.files['file']  # Get the uploaded file from the form

        # Ensure the file is a CSV
        if not file or not file.filename.endswith('.csv'):
            return "Invalid file", 400  # Return error if not valid
        
        label = "default"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"uploaded_{label}_{timestamp}.csv"
        # Save the uploaded file to the 'uploads' folder
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
    
        # Load the CSV file using pandas
        df = pd.read_csv(filepath)

        # Process the data using your model pipeline
        processed_df = upload_csv(df)

        # Save the processed DataFrame to a new file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"uploaded_{label}_{timestamp}.csv"

        output_path = os.path.join('processed', 'labeled_' + filename)
        processed_df.to_csv(output_path, index=False)

        # Send the processed file back to the user for download
        return send_file(output_path, as_attachment=True)

    # If the user is just visiting the page, show the upload form
    return render_template('upload.html')

# === Start the Flask development server ===
# debug=True gives helpful error messages and auto-reloads the app during development
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
