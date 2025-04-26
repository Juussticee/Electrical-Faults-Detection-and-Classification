from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime
from model import rfc_model, lof_model
import csv

fault_mapping = {
    '0000': 0,  # No Fault
    '1000':0,
    '0100':0,
    '0010':0,
    '0001':0,
    '1101': 2,  # Double Line-to-Ground
    '1110': 4,
    '0101': 1,
    '0011': 1,
    '1010': 3,
    '1100': 3,
    '1001': 1,  # Single Line-to-Ground (SLG)
    '1011': 2,
    '0111': 2,
    '0110': 3,  # Line-to-Line Fault (LL)
    '1111': 5,
}

def binary_to_fault(binary_code):
    # Convert binary code (array of 0s and 1s) to string
    binary_str = ''.join(map(str, binary_code.astype(int)))  # Ensure binary is string
    #fault_codes = int(binary_str)
    fault_codes = binary_str
    # Map binary string to fault class using the fault_mapping dictionary
    return fault_codes
def convert_predictions_to_fault_classes(y_pred_binary):
    fault_classes = []
    for binary_code in y_pred_binary:
        fault_class = binary_to_fault(binary_code) # Convert each binary code to a fault class
        fault_classes.append(fault_class)
    return fault_classes  # Return a list of fault classes (not an array)



voltage_scaler = MinMaxScaler(feature_range=(-2.5, 2.5))
current_scaler = MinMaxScaler(feature_range=(-2.5, 2.5))
ground_current_scaler = MinMaxScaler(feature_range=(-2.5, 2.5))

def append_to_csv(df, results, csv_file="output/data.csv"):
    # First, make sure the results (prediction codes) are in the right format (as a new column)
    df['fault'] = results
    
    # Append the DataFrame to the CSV file
    # The mode 'a' stands for append, and header=False ensures that we don't add the header again
    df.to_csv(csv_file, mode='a', header=False, index=False)
    print("Data appended to CSV.")


data = []
results = []

def pred_values(df, state):
    """
    This function checks for outliers first and then uses RFC for classification if not an outlier.

    df: DataFrame containing the scaled data
    """
    # --- Input validation (keep as is) ---
    if not isinstance(df, pd.DataFrame):
        print("Warning: Input format is incorrect. Please provide a pandas DataFrame.")
        # The loop below might error if df is not a DataFrame. Consider removing or adding checks.
        # for col in df.columns:
        #      print(f"Column '{col}' has dtype {df[col].dtype}")
        return [None] * len(df) if hasattr(df, '__len__') else [None]
    if df.empty:
        print("Warning in pred_values: Input DataFrame is empty.")
        return []

    print("Checked input, now predicting...")
    try:
        # 1. Perform LOF prediction
        if state == 'real-time':
            lof_predictions = lof_model.predict(df)  # Returns array of 1 (inlier) or -1 (outlier)
            print("Predicted LOF, now processing results...")
        else:
            lof_predictions = lof_model.predict(df)
            print("Predicted LOF, now processing results...")

        # --- Minimal Alteration Starts Here ---

        # 2. Create a new array for final results, initialized to 0 (No Fault)
        if state == 'real-time':
            print("creating a new array for final results")
            final_codes = np.zeros(1, dtype=int)

            print("Now Checking whehter it's an outlier or not")
            if lof_predictions[-1] == -1:
                final_codes[0] = 2000
            else: 
                print(f"Processing the real time inlier with RFC...")
                rfc_binary_predictions = rfc_model.predict(df)[0]
                rfc_fault_classes = binary_to_fault(rfc_binary_predictions)
                final_codes = rfc_fault_classes
            return final_codes
        else:
            print("creating a new array for final results")
            final_codes = np.zeros(len(df), dtype=object)

           # 3. Identify outlier indices and assign the outlier code (2000)
            outlier_indices = np.where(lof_predictions == -1)[0]
            final_codes[outlier_indices] = 2000

           # 4. Identify inlier indices
            inlier_indices = np.where(lof_predictions == 1)[0]

           # 5. If there are inliers, predict using RFC and assign results
        if len(inlier_indices) > 0:
            print(f"Processing {len(inlier_indices)} inliers with RFC...")
            inlier_data = df.iloc[inlier_indices]
            # Ensure rfc_model is loaded correctly and is a model object
            rfc_binary_predictions = rfc_model.predict(inlier_data)
            rfc_fault_classes = convert_predictions_to_fault_classes(rfc_binary_predictions)
            print('rfc fault classes:',rfc_fault_classes[:5])

            # Assign RFC results to the correct positions in the final_codes array
            if len(rfc_fault_classes) == len(inlier_indices):
                final_codes[inlier_indices] = rfc_fault_classes
                print('final codes',final_codes[:5])
            else:
                print(f"Warning: Mismatch between RFC predictions ({len(rfc_fault_classes)}) and inliers ({len(inlier_indices)}). Assigning error code.")
                final_codes[inlier_indices] = 9998 # Example error code
                # 6. Return the final codes as a list
            return final_codes.astype(str)

        # --- Minimal Alteration Ends Here ---

    # --- Keep Error Handling ---
    except AttributeError as ae:
         if "'numpy.ndarray' object has no attribute 'predict'" in str(ae):
              print("\n--- CRITICAL ERROR in pred_values ---")
              print("The 'rfc_model' loaded from 'rfc_model.pkl' is NOT a model object.")
              print("It seems to be a NumPy array. Please REGENERATE 'rfc_model.pkl' correctly.")
              print("-------------------------------------\n")
              return [9996] * len(df) # Specific error code
         else:
              print(f"AttributeError during prediction: {ae}")
              return [9999] * len(df)
    except Exception as e:
        print(f"Error during prediction in pred_values: {e}")
        return [9999] * len(df) # General error code


def real_time_prediction(df):
     global voltage_scaler, current_scaler, ground_current_scaler
     columns = [ 'Va', 'Vb', 'Vc','Ia', 'Ib', 'Ic', 'Ig']
     print("Creating the dataframe for the array")
     df = pd.DataFrame([df], columns=columns)
     print(df)
     print("use the scaler to transform the data")
     df.iloc[:, 0:3] = voltage_scaler.transform(df.iloc[:, 0:3])
     df.iloc[:, 3:6] = current_scaler.transform(df.iloc[:, 3:6])
     df.iloc[:, 6:7] = ground_current_scaler.transform(df.iloc[:,6:7])
     print("Scaled the array")
     code = pred_values(df, 'real-time')
     print("Got the code")
     threading.Thread(target=append_to_csv, args=(df, code)).start()
     print("Got the preds sending to the server")
     print("sent to server")
     print('The code:', code)
     return code

def process_csv(df):
    """
    Scales the feature columns of the input DataFrame.
    Assumes the first 7 columns are the features to be scaled.
    Returns a DataFrame with ONLY the scaled features.
    """
    global voltage_scaler, current_scaler, ground_current_scaler
    print("Processing CSV...")
    columns = ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic', 'Ig']

    # Ensure input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        try:
            print("Attempting to convert input to DataFrame...")
            df = pd.DataFrame(df, columns=columns)
            print("Sucessfully converted to DataFrame")
        except Exception as e:
            print(f"Error converting input to DataFrame in process_csv: {e}")
            return None # Indicate failure
    else:
        df = pd.DataFrame(df, columns=columns)
    # Check if there are enough columns
    if df.shape[1] < 7:
        print(f"Error: process_csv requires at least 7 columns, found {df.shape[1]}")
        return None # Indicate failure

    # --- REMOVED: df['Faults'] = "" --- No longer adding this column here
    try:
        df_scaled=df.copy()
        print("Trying to Scale inputs in process_csv")
        # Scale each of the columns based on the corresponding scaler
        # Use .values for potentially better performance and compatibility
        df_scaled.iloc[:, 0:3] = voltage_scaler.fit_transform(df.iloc[:, 0:3])
        print("Scaled voltages")
        df_scaled.iloc[:, 3:6] = current_scaler.fit_transform(df.iloc[:, 3:6])
        print("Scaled Currents")
        # Ensure the 7th column (index 6) is treated as 2D for the scaler
        df_scaled.iloc[:, 6:7] = ground_current_scaler.fit_transform(df.iloc[:, 6:7])
        print("Scaled Ground Current")

        # Ensure dtypes are float after scaling
        print("Ensuring everything is float")
        df_scaled = df_scaled.astype(float)
        print("Everything is float")

    except Exception as e:
        print(f"Error during scaling in process_csv: {e}")
        # Consider logging the problematic data snippet if possible
        # print(f"Data snippet causing scaling error:\n{df_scaled.head()}")
        return None # Indicate failure
    print("returning df")
    print(df_scaled)
    save_file(df_scaled, 'processed')
    # Return ONLY the 7 scaled columns
    return df_scaled

# c:\Users\Ansoufeh\Documents\elec_system\pipeline.py

# ... (keep other functions like binary_to_fault, process_csv, pred_values etc.)
# ... (keep fault_mapping, scalers etc.)
def save_file(df, output_dir):
    label = "default"
    # Corrected timestamp format string in case it was wrong before
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"uploaded_{label}_{timestamp}.csv"

    # Ensure 'output' directory exists
    output_dir = output_dir or 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    try:
        df.to_csv(output_path, index=False)
        print(f"Processed CSV saved to: {output_path}")
    except Exception as e:
        print(f"Error saving processed CSV to {output_path}: {e}")
        # Decide if you should still return original_df or indicate failure

def upload_csv(df, label=None):
    """
    Processes an uploaded CSV file:
      - Checks that there are at least 7 columns.
      - Selects the first 7 columns and scales them using process_csv.
      - Iterates over each row (as a 1-row DataFrame) and computes a prediction via pred_values.
      - Adds the predictions to the original DataFrame.
      - Saves the result in the 'output' folder.
    """
    original_df = df.copy()  # Keep a copy for the final output

    # Validate that the DataFrame has at least 7 columns.
    if df.shape[1] < 7:
        error_msg = "Input DataFrame has less than 7 columns required for processing."
        print(f"Error: {error_msg}")
        original_df['fault'] = error_msg
        return original_df
    columns = ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic', 'Ig']
    # Scale the first 7 feature columns.
    try:
        # Process only the first 7 columns.
        df_scaled = process_csv(df[columns].copy())
        if df_scaled is None or not isinstance(df_scaled, pd.DataFrame):
            raise ValueError("Scaling failed or returned incorrect type.")
    except Exception as e:
        print(f"Error during scaling/setup in upload_csv: {e}")
        original_df['fault'] = f'Error: {e}'
        return original_df

    print(f"Processing {len(df_scaled)} rows from uploaded CSV...")

    # Iterate through each row of the scaled DataFrame.
    up_codes = []
    try:
        code = pred_values(df_scaled, 'upload')
        print("code", code)
        up_codes= list(code)
        print("up_code",up_codes[1])
    except Exception as e:
        print(f"\n❌ Error processing row index")
        print(f"Error during prediction: {e}")
        up_codes = [9999] * len(original_df)

    # If the number of predictions doesn't match, pad with None.
    if len(up_codes) != len(original_df):
        print(f"Warning: Mismatch between original rows ({len(original_df)}) and predictions ({len(up_codes)}).")
        up_codes = (up_codes + [None] * len(original_df))[:len(original_df)]
    original_df['fault']=""
    original_df['fault'] = original_df['fault'].astype(str)
    original_df['fault'] = up_codes
    print("fault col",original_df['fault'].iloc[1])

    # Save the output CSV with a unique filename.
    label = label or "default"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"uploaded_{label}_{timestamp}.csv"
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        print(original_df.info())
        original_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Processed CSV saved to: {output_path}")
    except Exception as e:
        print(f"Error saving processed CSV to {output_path}: {e}")

    return original_df