## 🔥 Real-Time Fault Classification Flask App
This Flask application hosts a trained machine learning model that predicts faults in a 3-phase electrical system using real-time or batch data.

# 📋 Project Overview
The system receives voltage and current data as input, processes it, and outputs fault classifications for each phase (A, B, C) and Ground.

# Input:

3 Phase Voltages: Va, Vb, Vc

3 Phase Currents: Ia, Ib, Ic

# Output:

4 Binary Classifications:

Phase A Fault (0/1)

Phase B Fault (0/1)

Phase C Fault (0/1)

Ground Fault (0/1)

# ⚙️ How It Works
The Flask server exposes a /predict endpoint.

Incoming data is parsed and passed to the trained Decision Tree Classifier.

The server responds with predicted fault statuses.

Data can be sent via UDP (for real-time) or uploaded as a CSV file (for batch processing).

# 🚀 Setup Instructions
Clone the Repository

`git clone git clone https://github.com/Juussticee/Electrical-Faults-Detection-and-Classification.git
cd Electrical-Faults-Detection-and-Classification/elec_system`

# Create and Activate a Virtual Environment

`python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)`
]
# Install Dependencies

`pip install -r requirements.txt
Run the Flask App`
#Run the app
`python app.py`
# 📡 Endpoints

Route	Method	Description
/predict	POST	Accepts voltage and current data, returns fault predictions.
/upload	POST	Accepts CSV file upload for batch predictions.
# 📊 Model Details
Algorithm: Decision Tree Classifier

 ### Training Data: Simulink simulated data representing different fault scenarios

 ### Inputs: [Va, Vb, Vc, Ia, Ib, Ic]

 ### Outputs: [Phase A fault, Phase B fault, Phase C fault, Ground fault]

# 📁 Project Structure
├── app.py
├── model/
│   └── decision_tree_model.pkl
├── static/
├── templates/
│   └── upload.html
├── requirements.txt
└── README.md
