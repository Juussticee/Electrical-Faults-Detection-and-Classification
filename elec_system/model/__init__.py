from joblib import load
import joblib

# Load LOF model (.joblib)
lof_model = joblib.load("model/lof_model.joblib")

# Load ONNX model using ONNX Runtime
rfc_model = joblib.load('model/rfc_model.joblib')
