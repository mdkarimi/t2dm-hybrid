# t2dm_hybrid_model.py
# Hybrid Fuzzy Logic, RFE, and Logistic Regression for T2DM Prediction
# License: MIT (https://opensource.org/licenses/MIT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data[data['age'].between(35, 45)]  # Filter age 35-45
    data.fillna(data.median(), inplace=True)  # Median imputation
    X = data.drop('t2dm', axis=1)
    y = data['t2dm']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    return X_balanced, y_balanced, X.columns

# Fuzzy logic system
def create_fuzzy_system():
    # Define fuzzy variables (FPG, HbA1c, BMI)
    fpg = ctrl.Antecedent(np.arange(70, 201, 1), 'fpg')
    hba1c = ctrl.Antecedent(np.arange(4, 8, 0.1), 'hba1c')
    bmi = ctrl.Antecedent(np.arange(18, 50, 0.1), 'bmi')
    risk = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'risk')

    # Membership functions (ADA thresholds)
    fpg['normal'] = fuzz.trimf(fpg.universe, [70, 100, 126])
    fpg['prediabetic'] = fuzz.trimf(fpg.universe, [100, 126, 140])
    fpg['diabetic'] = fuzz.trimf(fpg.universe, [126, 140, 200])
    hba1c['normal'] = fuzz.trimf(hba1c.universe, [4, 5.7, 6.4])
    hba1c['prediabetic'] = fuzz.trimf(hba1c.universe, [5.7, 6.4, 7])
    hba1c['diabetic'] = fuzz.trimf(hba1c.universe, [6.4, 7, 8])
    bmi['normal'] = fuzz.trimf(bmi.universe, [18, 25, 30])
    bmi['overweight'] = fuzz.trimf(bmi.universe, [25, 30, 35])
    bmi['obese'] = fuzz.trimf(bmi.universe, [30, 35, 50])
    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 0.3])
    risk['moderate'] = fuzz.trimf(risk.universe, [0.3, 0.5, 0.7])
    risk['high'] = fuzz.trimf(risk.universe, [0.7, 1, 1])

    # Fuzzy rules (from Table 2 in manuscript)
    rules = [
        ctrl.Rule(fpg['diabetic'] & hba1c['diabetic'], risk['high']),
        ctrl.Rule(fpg['prediabetic'] & hba1c['prediabetic'] & bmi['obese'], risk['high']),
        ctrl.Rule(fpg['normal'] & hba1c['normal'], risk['low']),
        # Add other rules as per Table 2 (12 total)
    ]
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim

# Genetic algorithm for optimizing fuzzy boundaries (simplified)
def optimize_fuzzy_boundaries(X, y, generations=50):
    # Placeholder: Optimize membership function boundaries using AUC
    return create_fuzzy_system()  # Replace with actual optimization

# Main hybrid model
def hybrid_model(X, y, feature_names, k=4):
    # Fuzzy logic transformation
    fuzzy_sim = optimize_fuzzy_boundaries(X, y)
    X_fuzzy = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        fuzzy_sim.input['fpg'] = X[i, feature_names.get_loc('fpg')]
        fuzzy_sim.input['hba1c'] = X[i, feature_names.get_loc('hba1c')]
        fuzzy_sim.input['bmi'] = X[i, feature_names.get_loc('bmi')]
        fuzzy_sim.compute()
        X_fuzzy[i] = fuzzy_sim.output['risk']

    # RFE for feature selection
    model = LogisticRegression(penalty='l2', solver='newton-cg', random_state=42)
    rfe = RFE(model, n_features_to_select=k)
    X_rfe = rfe.fit_transform(X, y)
    selected_features = feature_names[rfe.support_]

    # Logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Decision tree for clinical rules
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_rfe, y)

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'sensitivity': recall_score(y_test, y_pred),
        'specificity': precision_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    return metrics, model, tree, selected_features

# Example usage
if __name__ == "__main__":
    # Replace with actual dataset path (e.g., PIDD)
    X, y, feature_names = load_data('pidd.csv')
    metrics, model, tree, selected_features = hybrid_model(X, y, feature_names)
    print("Performance:", metrics)
    print("Selected Features:", selected_features)