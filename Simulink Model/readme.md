## 📌 Overview
This Simulink model simulates a 3-phase electrical power system and detects faults on phases A, B, C, and Ground (G). The fault types include single-phase, two-phase, and three-phase faults, with real-time sensor data exported for analysis.
it was taken from https://github.com/KingArthur000/Electrical-Fault-detection-and-classification/blob/main/ANN_project.slx and performed some modifications on it 
## ⚙️ Features
- Models 3-phase voltage and current behavior under normal and faulty conditions.
- Detects line-to-ground, line-to-line, and three-phase faults.
- Sends real-time voltage and current values to external systems via UDP for machine learning-based fault classification.

## 🧩 Components
- **Three-Phase Source**
- **Fault Blocks** for triggering faults
- **Current & Voltage Measurement Blocks**
- **UDP Send Block** to transmit real-time data
- **Scope/Display Blocks** for visualization

## 🛠 Requirements
- MATLAB R202x with Simulink
- Instrument Control Toolbox (for UDP communication)
- Control System Toolbox (if using scopes or control blocks)

## 🚀 How to Run
1. Open `fault_detection_simulink_model.slx` in MATLAB Simulink.
2. Configure simulation parameters (time, solver).
3. Click **Run**.
4. Faults will be injected during simulation and corresponding voltages/currents will be sent over UDP.

## 🔄 Input & Output
- **Input:** None (automated fault simulation)
- **Output:** Voltage & current values from each phase via UDP (formatted as CSV or float arrays)

## 🧪 Testing
To test the model:
- Open MATLAB.
- Run the Simulink model.
- Use a separate Python/Flask app to receive and classify data.
