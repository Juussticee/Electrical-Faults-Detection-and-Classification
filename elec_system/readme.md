# 🔥 Real-Time Fault Classification Flask App
This Flask application hosts a trained machine learning model that predicts faults in a 3-phase electrical system using real-time or batch data.

## 📋 Project Overview
The system receives voltage and current data as input, processes it, and outputs fault classifications for each phase (A, B, C) and Ground.

## Input:

3 Phase Voltages: Va, Vb, Vc

3 Phase Currents: Ia, Ib, Ic

## Output:

4 Binary Classifications:

Phase A Fault (0/1)

Phase B Fault (0/1)

Phase C Fault (0/1)

Ground Fault (0/1)

## ⚙️ How It Works
The Flask server exposes a /predict endpoint.

Incoming data is parsed and passed to the trained Decision Tree Classifier.

The server responds with predicted fault statuses.

Data can be sent via UDP (for real-time) or uploaded as a CSV file (for batch processing).

Our UDP server and Flask app communicate on the port 5005 and flask run on port 5000

## 🚀 Setup Instructions
Clone the Repository

`git clone git clone https://github.com/Juussticee/Electrical-Faults-Detection-and-Classification.git
cd Electrical-Faults-Detection-and-Classification/elec_system`

## Create and Activate a Virtual Environment

`python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)`

## Install Dependencies

`pip install -r requirements.txt
Run the Flask App`
#Run the app
`python app.py`
## 📡 Endpoints

Route	Method	Description
/predict	POST	Accepts voltage and current data, returns fault predictions.
/upload	POST	Accepts CSV file upload for batch predictions.
## 📊 Model Details
Algorithm: Decision Tree Classifier

Training Data: Simulink simulated data representing different fault scenarios

Inputs: [Va, Vb, Vc, Ia, Ib, Ic]

Outputs: [Phase A fault, Phase B fault, Phase C fault, Ground fault]

# 📁 Project Structure
```
├── model/
│ └── __init__.py
│ └── rfc_model.joblib
│ └── lof_model.joblib
├── output/      #for the csv the system retun
├── processed/      #for the csv the system scale , this was added for debugging and can be removed
├── uploads/    #the csv the user input, we keep them at their raw format so we can inpect and decide later if this data can be used for further training the model
├── templates/
│ └── upload.html
│ └── setup.html
│ └── index.html
├── requirements.txt
├── app.py
├── udp_server.py  #Where the operations of connecting the simulink model to our flask are done
├── pipeline.txt
└── README.md
```

# Showcase of the application:
## CSV Upload
[![Watch the Demo](https://img.youtube.com/vi/7N1Ab793O-0/0.jpg)](https://youtu.be/7N1Ab793O-0)

## Real Time Detection:
[![Watch the Demo](https://img.youtube.com/vi/C7GXpJWZbZw/0.jpg)](https://youtu.be/C7GXpJWZbZw)
P.S. The "Error Fetching Update" message observed during the application usage is intentionally triggered by backend debugging operations implemented during development. It does not reflect any issues with the model’s performance or data interpretation. In a production environment, such debugging features would be removed to ensure a seamless user experience
