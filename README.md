# Electrical-Faults-Detection-and-Classification
Intelligent Fault Detection in Electrical Power Systems Using Machine Learning
# Introduction
The integrity and reliability of electrical power transmission systems are critical to the functioning of modern infrastructure. However, these systems are susceptible to faults—such as short circuits and ground faults—that can lead to service disruptions, equipment degradation, or even catastrophic system failures. Timely and accurate fault detection is, therefore, essential for maintaining grid stability and operational safety.
Having grown up in a region prone to frequent power outages, we became deeply curious about how electrical systems could be made smarter and more reliable. This curiosity inspired me to explore how machine learning could be used to predict and classify faults before they escalate.
Conventional fault detection methodologies, including manual inspections and rule-based diagnostic systems, are increasingly inadequate for modern, dynamic power grids. These approaches often lack the adaptability, speed, and scalability needed to detect complex fault patterns in real time.
This capstone project, developed as part of the Zaka.ai AIC machine learning cohort , proposes a machine learning-based framework for the automatic detection and classification of electrical faults using voltage and current sensor data. The objective is to contribute to the development of intelligent, data-driven diagnostic tools that enhance the operational resilience and reliability of modern power systems.
The project was developed by a diverse team of students from Electrical Engineering, Computer Science, and Communications Engineering backgrounds, bringing together deep technical knowledge and AI proficiency to solve real-world infrastructure challenges.
# Problem Statement
Electrical transmission lines are frequently exposed to various environmental and technical challenges that can cause faults. These may include line-to-ground faults, line-to-line faults, or simultaneous multi-phase failures. Undetected or poorly classified faults can lead to cascading effects, causing severe equipment damage and widespread power outages.
Traditionally, such problems have been addressed using:
- Manual inspections: Labor-intensive and subject to human error.
- Threshold-based rule systems: Limited in their ability to generalize to unseen fault types or conditions.
- Protective relays and circuit breakers: While reactive, these mechanisms do not always provide adequate information for root-cause analysis or predictive fault management.
Recent research has explored the integration of artificial intelligence techniques, particularly supervised learning models, for fault classification tasks. However, many implementations rely on limited datasets and fail to demonstrate generalizability or real-world deployment readiness.
# Proposed Approach
This project leverages supervised machine learning to identify and classify fault types based on electrical sensor readings. The approach utilizes a dataset sourced from Kaggle, which comprises simulated and/or measured readings from a three-phase electrical system. The features include:
- Voltage measurements: Va, Vb, Vc
- Current measurements: Ia, Ib, Ic
- Ground current: Ig 
- Fault indicators: Binary flags indicating faults on phases A, B, C, and ground (G)
The primary objective is to train models capable of accurately identifying whether a fault has occurred, and if so, determine its type and which particular phases are involved(e.g., phase-to-phase, phase-to-ground, or multiphase faults).
# Methodology
## Dataset Generation:
We decided we are going to generate a dataset because our current dataset wasn’t representative of all possible faults on different lines (e.g. we had only 1 example of lg fault which was between A and G , we wanted all possible examples so that the model can generalize well)
-	We used a 3 Phases System Simulink model downloaded from this Github repo researching this topic lightly : The system consists of 4 generators of 11 × 103 V, each pair located at each end of the transmission line, with transformers in between to simulate and study the various faults at the midpoint of the transmission line.
-	 
##  Technical Implementation
The following machine learning models were evaluated:
-	 Random Forest Classifier
-	 Gradient Boosting Classifier
-	K Nearest Neighbor
-	Naïve Bayes
-	 Support Vector Machine (SVM)
## Preprocessing Steps
-	Data Cleaning: Null or inconsistent values were addressed via imputation.
-	Feature Scaling: MinMaxScaler with a preset range between (-2.5, 2.5) was applied to normalize voltage and current readings. 
-	Data Split: An 80/20 training-test split was implemented, with stratification to preserve class distribution.
-	Output: The output comes in the shape of 4 columns ABCG , to effectively check the error on each line individually . e.g. A=1 B=0 C=1 G=0 (LL Fault)
## Feature Engineering
-	In the initial trials we were getting a lot of FN and FP between LLL faults and LLLG, and after further data visualization we noticed that these 2 faults are actually so similar, so we needed to find and add a feature that will help distinguish between them 
-	After trials with measurements like frequency, impedance . We found out that the ground current was the best feature to be added sine the difference between these faults is the presence/absence of the ground, and actually it made the model not only perform well no these 3 phases faults but also in other faults increasing accuracy from 89% to 98-99% 
# Experimental Setup
The experiments were conducted in a Jupyter Notebook environment using Python and libraries such as scikit-learn, xgboost, and matplotlib for analysis and visualization.
Hyperparameter tuning was performed using grid search and cross-validation to optimize performance. Models were evaluated using the following metrics:
- Accuracy: Measures overall correctness
- F1-Score: Balances precision and recall for imbalanced data
- Matthews Correlation Coefficient (MCC): Accounts for all confusion matrix categories
- ROC-AUC: Evaluates the ability to discriminate between classes



# Results and Findings
Model	Accuracy	F1-Score	MCC	ROC-AUC
Random Forest Classifier	98.6%	99.4%	98.7%	99.7%
Gradient Boosting	90%	97.1%	93.8%	98.9%
K-NN	97.4%	98.9%	97.6%	99.6%
Naïve Bayes	47.7%	86%	67.9%	90%
SVM	87.4%	85%	79%	91%

The Random Forest Classifier demonstrated superior performance, achieving:
Key observations:
- Voltage signals and ground current (Ig) emerged as the most significant features.
- The models exhibited robust performance in identifying both single-phase and multi-phase faults.
- Low false-positive rates suggest the system’s reliability in minimizing unnecessary alerts.

# Deployment and Integration
To bring the intelligent fault detection system into a practical, real-world environment, we developed a Flask-based application that serves two primary functions: real-time analysis via MATLAB integration and batch processing of electrical system data.
## Real-Time Fault Detection via MATLAB Integration
In this configuration, the system interfaces directly with MATLAB. The user provides a small CSV file containing representative voltage and current measurements to fit a custom MinMaxScaler (range: [-2.5, 2.5]). This ensures consistency in readings across different electrical systems.
Once the scaler is set, MATLAB continuously sends live sensor data to the Flask app. The app processes this data as follows:
•	Normalization: Incoming data is transformed using the pre-fitted MinMaxScaler to maintain consistency with training data.
•	Outlier Detection: The normalized data is passed to the Local Outlier Factor (LOF) model. If an anomaly is detected, it is flagged with fault code 2000.
•	Fault Classification: If the data is not flagged as an outlier, it is passed to the pre-trained Random Forest Classifier (RFC) for fault prediction.
By combining preprocessing, outlier detection, and machine learning classification, the Flask app provides real-time insights into system health.
## CSV Analysis and Labeling
The system also accepts CSV files containing voltage, current, and circuit data for batch processing:
•	Data Normalization: Each batch starts by fitting a new MinMaxScaler based on the provided data.
•	Outlier Check: Each row is checked by the LOF model. If flagged as an anomaly (fault code 2000), it is marked accordingly.
•	Fault Prediction: Valid data is passed to the RFC for fault classification.
This flexible approach supports both real-time and batch fault detection.
## AWS Deployment
To simulate a real-world deployment environment, we hosted our model on an AWS t3.micro instance.
This deployment approach ensures global accessibility and provides an opportunity to evaluate the model’s performance in a cloud-based production setup.
While Docker is certainly a valid and widely used option for deployment, we chose to explore AWS to understand different solutions and benefits for model hosting.
Additionally, AWS services offer powerful tools like CloudWatch for monitoring, logging, and scaling our model in production environments.

# Industry Implications
The successful implementation of intelligent fault classification systems holds substantial promise for the electrical power industry. Specifically:
- Operational Efficiency: Reduces manual inspection efforts and shortens fault resolution time.
- Preventive Maintenance: Enables utilities to shift from reactive to predictive maintenance strategies.
- Cost Reduction: Minimizes asset damage, service interruptions, and labor costs.
- Enhanced Grid Stability: Contributes to the creation of self-healing smart grids.
Furthermore, the proposed solution can be deployed in developing regions, where limited infrastructure and technical expertise necessitate automated, easy-to-deploy systems for grid health monitoring.
# Future Work and Improvements
To enhance the capabilities and scalability of the proposed system, the following directions are recommended:
1. Incorporate Temporal Dynamics: Utilize recurrent neural networks (e.g., LSTM) to capture time-dependent patterns in voltage and current signals.
2. Multimodal Input: Integrate thermal imaging or aerial drone data using convolutional neural networks for visual fault detection.
3. Edge Deployment: Optimize the model for embedded systems to allow real-time inference on microcontrollers or Raspberry Pi devices.
4. Integration with SCADA Systems: Interface with existing supervisory control infrastructure to support end-to-end automation.
# Conclusion
This project demonstrates the viability of machine learning techniques in advancing fault detection and classification in electrical power systems. By leveraging structured voltage and current sensor data, the proposed approach significantly improves diagnostic accuracy, response speed, and overall reliability of power grid operations.
Such innovations are critical as the world moves toward smarter, more sustainable energy infrastructure. With further development, the solution has the potential to contribute meaningfully to the digital transformation of the energy sector.
# Citations
Kaggle Dataset: Electrical Fault detection and Classification by E Sathya Prakash
Simulink Model Github Repo : Electrical Faults Detection and Classification by KingArthur000
