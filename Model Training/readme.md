# Model Training Description
The machine learning model was trained using labeled data collected from the Simulink simulation of different fault scenarios. Each training sample includes six input features:

3 Phase Voltages (Va, Vb, Vc)

3 Phase Currents (Ia, Ib, Ic)

The target output consists of four classes, representing the fault status for each line (Phase A, Phase B, Phase C, and Ground).
Each output is a binary value e.g. 1110 : A=1, B=1, C=1, G=0.

The dataset was split into a training set and a validation set. A Decision Tree Classifier was trained to learn patterns from the input signals and predict the corresponding fault types.
The model was then exported and integrated into the real-time pipeline for live classification during simulation.
