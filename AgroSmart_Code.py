"""
AgroSmart Recommender: Enhanced Crop Recommendation System
Author: AI/ML Developer Intern
Date: December 2025

This code implements a comprehensive crop recommendation system using:
- Data preprocessing with SMOTE balancing
- Feature selection (RFE + Mutual Information)
- Comparative analysis of 5 ML algorithms
- 10-fold cross-validation
- Comprehensive visualizations and metrics

Requirements:
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE, mutual_info_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("AgroSmart Recommender: Crop Recommendation System")
print("="*80)

# ======================== PART 1: DATA LOADING AND PREPROCESSING ========================

print("\n[STEP 1] Loading Data...")
# Load Kaggle Crop Recommendation Dataset
# For demonstration, we'll create a sample dataset (replace with actual Kaggle dataset)
# Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

try:
    df = pd.read_csv('crop_recommendation.csv')
    print(f"✓ Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("⚠ Dataset file not found. Creating sample dataset for demonstration...")
    # Create sample dataset
    np.random.seed(RANDOM_STATE)
    n_samples = 2200
    crops = ['Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans',
             'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate', 'Banana', 'Mango',
             'Grapes', 'Watermelon', 'Muskmelon', 'Apple', 'Orange', 'Papaya',
             'Coconut', 'Cotton', 'Sugarcane', 'Jute']
    
    df = pd.DataFrame({
        'N': np.random.uniform(0, 140, n_samples),
        'P': np.random.uniform(5, 145, n_samples),
        'K': np.random.uniform(5, 205, n_samples),
        'temperature': np.random.uniform(8, 43, n_samples),
        'humidity': np.random.uniform(14, 99.5, n_samples),
        'ph': np.random.uniform(3.5, 9.5, n_samples),
        'rainfall': np.random.uniform(20, 254, n_samples),
        'label': np.random.choice(crops, n_samples)
    })
    print(f"✓ Sample dataset created: {df.shape[0]} samples, {df.shape[1]} features")

print(f"\nDataset Info:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Unique crops: {df['label'].nunique()}")
print(f"\nDataset Sample:")
print(df.head())

# ======================== PART 2: EXPLORATORY DATA ANALYSIS ========================

print("\n[STEP 2] Exploratory Data Analysis...")

print("\n📊 Feature Statistics:")
print(df.describe().round(2))

print("\n📊 Class Distribution:")
class_dist = df['label'].value_counts()
print(class_dist)

# Visualize class distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Class distribution
axes[0, 0].barh(class_dist.index, class_dist.values, color='skyblue')
axes[0, 0].set_xlabel('Count')
axes[0, 0].set_title('Crop Class Distribution', fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# Feature distributions
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for idx, feature in enumerate(['N', 'temperature']):
    ax = axes[0, 1] if feature == 'N' else axes[1, 0]
    ax.hist(df[feature], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{feature} Distribution', fontweight='bold')
    ax.grid(alpha=0.3)

# Feature correlation
corr_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
correlation_matrix = df[corr_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1],
            square=True, cbar_kws={'label': 'Correlation'})
axes[1, 1].set_title('Feature Correlation Matrix', fontweight='bold')

plt.tight_layout()
plt.savefig('01_EDA_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: 01_EDA_Analysis.png")
plt.close()

# ======================== PART 3: DATA PREPROCESSING ========================

print("\n[STEP 3] Data Preprocessing...")

# Prepare features and target
X = df.drop('label', axis=1)
y = df['label']

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# Feature scaling (Min-Max normalization)
print("  Applying Min-Max Scaling...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Encode target variable
print("  Encoding target variable...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"  ✓ Features scaled to [0, 1]")
print(f"  ✓ Target encoded: {len(le.classes_)} classes")
print(f"  Classes: {le.classes_}")

# ======================== PART 4: HANDLE IMBALANCED DATA (SMOTE) ========================

print("\n[STEP 4] Handling Imbalanced Data with SMOTE...")

print("  Checking class distribution...")
unique, counts = np.unique(y_encoded, return_counts=True)
print(f"  Class distribution before SMOTE:")
for cls, cnt in zip(unique, counts):
    print(f"    {le.classes_[cls]}: {cnt} samples")

# Check if imbalance exists
imbalance_ratio = max(counts) / min(counts)
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print("  Applying SMOTE for class balancing...")
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=RANDOM_STATE)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y_encoded)
    print(f"  ✓ SMOTE applied. New dataset shape: {X_balanced.shape}")
else:
    print("  Dataset is already well-balanced. Proceeding without SMOTE.")
    X_balanced, y_balanced = X_scaled.copy(), y_encoded.copy()

# ======================== PART 5: FEATURE SELECTION ========================

print("\n[STEP 5] Feature Selection (RFE + Mutual Information)...")

# Mutual Information scoring
print("  Computing Mutual Information scores...")
mi_scores = mutual_info_classif(X_balanced, y_balanced, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({'Feature': X_balanced.columns, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False)
print("  Mutual Information Scores:")
print(mi_df)

# RFE with Random Forest
print("\n  Applying RFE (Recursive Feature Elimination)...")
rf_for_rfe = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
rfe = RFE(estimator=rf_for_rfe, n_features_to_select=6, step=1)
rfe.fit(X_balanced, y_balanced)

# Get selected features
selected_features = X_balanced.columns[rfe.support_].tolist()
print(f"  ✓ Selected features: {selected_features}")

X_selected = X_balanced[selected_features]
print(f"  ✓ Feature set reduced to {X_selected.shape[1]} features")

# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mutual Information
axes[0].barh(mi_df['Feature'], mi_df['MI_Score'], color='coral')
axes[0].set_xlabel('MI Score')
axes[0].set_title('Mutual Information Scores', fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# RFE Rankings
rfe_ranking = pd.DataFrame({'Feature': X_balanced.columns, 'RFE_Ranking': rfe.ranking_})
rfe_ranking = rfe_ranking.sort_values('RFE_Ranking')
colors = ['green' if r == 1 else 'lightgray' for r in rfe_ranking['RFE_Ranking']]
axes[1].barh(rfe_ranking['Feature'], rfe_ranking['RFE_Ranking'], color=colors)
axes[1].set_xlabel('RFE Ranking (1=Selected)')
axes[1].set_title('RFE Feature Ranking', fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('02_Feature_Selection.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: 02_Feature_Selection.png")
plt.close()

# ======================== PART 6: TRAIN-TEST SPLIT ========================

print("\n[STEP 6] Train-Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_balanced, test_size=0.2, 
    stratify=y_balanced, random_state=RANDOM_STATE
)

print(f"  Training set: {X_train.shape[0]} samples ({100*0.8:.1f}%)")
print(f"  Testing set: {X_test.shape[0]} samples ({100*0.2:.1f}%)")
print(f"  Features: {X_train.shape[1]}")

# ======================== PART 7: MODEL TRAINING ========================

print("\n[STEP 7] Training Machine Learning Models...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=RANDOM_STATE
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=7, learning_rate=0.1, random_state=RANDOM_STATE
    ),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=RANDOM_STATE)
}

trained_models = {}
training_times = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    import time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    trained_models[model_name] = model
    training_times[model_name] = training_time
    print(f"  ✓ {model_name} trained in {training_time*1000:.2f}ms")

# ======================== PART 8: MODEL EVALUATION ========================

print("\n[STEP 8] Model Evaluation on Test Set...")

results = {}

for model_name, model in trained_models.items():
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_times[model_name]
    }
    
    print(f"\n  {model_name}:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")

# Results DataFrame
results_df = pd.DataFrame(results).T
print("\n📊 Overall Results Comparison:")
print(results_df.round(4))

# ======================== PART 9: CROSS-VALIDATION ========================

print("\n[STEP 9] 10-Fold Cross-Validation...")

cv_results = {}
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

for model_name, model in trained_models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_results[model_name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    print(f"\n  {model_name}:")
    print(f"    Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"    Std Dev: {cv_scores.std():.4f}")
    print(f"    Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")

# ======================== PART 10: DETAILED ANALYSIS FOR BEST MODEL ========================

print("\n[STEP 10] Detailed Analysis of Best Model (Random Forest)...")

best_model = trained_models['Random Forest']
y_pred_best = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print(f"\n  Confusion Matrix shape: {cm.shape}")

# Classification Report
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_, zero_division=0))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n  Feature Importance (Random Forest):")
print(feature_importance)

# ======================== PART 11: COMPREHENSIVE VISUALIZATIONS ========================

print("\n[STEP 11] Generating Comprehensive Visualizations...")

# 1. Model Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models_list = list(results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1_score']

for idx, metric in enumerate(metrics):
    row, col = idx // 2, idx % 2
    values = [results[model][metric] for model in models_list]
    colors = ['#2ecc71' if v == max(values) else '#3498db' for v in values]
    
    axes[row, col].bar(models_list, values, color=colors, alpha=0.7, edgecolor='black')
    axes[row, col].set_ylabel(metric.capitalize())
    axes[row, col].set_title(f'{metric.capitalize()} Comparison', fontweight='bold')
    axes[row, col].set_ylim([0.8, 1.0])
    axes[row, col].grid(axis='y', alpha=0.3)
    axes[row, col].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        axes[row, col].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('03_Model_Performance_Comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualization saved: 03_Model_Performance_Comparison.png")
plt.close()

# 2. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, 
            yticklabels=le.classes_, ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold')
ax.set_title('Confusion Matrix - Random Forest Model', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('04_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualization saved: 04_Confusion_Matrix.png")
plt.close()

# 3. Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal', alpha=0.7, edgecolor='black')
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Feature Importance - Random Forest', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add percentage labels
total_importance = feature_importance['Importance'].sum()
for idx, (feature, importance) in enumerate(zip(feature_importance['Feature'], feature_importance['Importance'])):
    percentage = (importance / total_importance) * 100
    ax.text(importance + 0.005, idx, f'{percentage:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('05_Feature_Importance.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualization saved: 05_Feature_Importance.png")
plt.close()

# 4. Cross-Validation Scores
fig, ax = plt.subplots(figsize=(12, 6))

cv_data = []
cv_labels = []
for model_name, cv_dict in cv_results.items():
    cv_data.append(cv_dict['scores'])
    cv_labels.append(model_name)

bp = ax.boxplot(cv_data, labels=cv_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#ffb366')
    patch.set_alpha(0.7)

ax.set_ylabel('Cross-Validation Accuracy', fontweight='bold')
ax.set_title('10-Fold Cross-Validation Results', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('06_Cross_Validation_Results.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualization saved: 06_Cross_Validation_Results.png")
plt.close()

# 5. Training Time Comparison
fig, ax = plt.subplots(figsize=(10, 6))
training_times_list = [training_times[model] for model in models_list]
colors = ['#e74c3c' if t > 0.5 else '#3498db' for t in training_times_list]

bars = ax.bar(models_list, training_times_list, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Training Time (seconds)', fontweight='bold')
ax.set_title('Model Training Time Comparison', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Add time labels
for bar, time in zip(bars, training_times_list):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{time*1000:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('07_Training_Time_Comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualization saved: 07_Training_Time_Comparison.png")
plt.close()

# ======================== PART 12: PREDICTION EXAMPLE ========================

print("\n[STEP 12] Example Crop Recommendation...")

# Create a sample input
sample_input = pd.DataFrame({
    'N': [90],
    'P': [42],
    'K': [43],
    'temperature': [20.9],
    'humidity': [82.0],
    'ph': [6.5],
    'rainfall': [202.9]
})

# Select only the features we used
sample_selected = sample_input[selected_features]

# Scale the sample
sample_scaled = scaler.transform(sample_selected)

# Make prediction
prediction_encoded = best_model.predict(sample_scaled)[0]
prediction_proba = best_model.predict_proba(sample_scaled)[0]
prediction_crop = le.inverse_transform([prediction_encoded])[0]
prediction_confidence = max(prediction_proba)

print(f"\n  Sample Input:")
print(f"    Nitrogen: 90 kg/hectare")
print(f"    Phosphorus: 42 kg/hectare")
print(f"    Potassium: 43 kg/hectare")
print(f"    Temperature: 20.9°C")
print(f"    Humidity: 82.0%")
print(f"    pH: 6.5")
print(f"    Rainfall: 202.9mm")

print(f"\n  🎯 Prediction Result:")
print(f"    Recommended Crop: {prediction_crop}")
print(f"    Confidence Score: {prediction_confidence:.2%}")

# Top 3 recommendations
top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
print(f"\n  Top 3 Recommendations:")
for rank, idx in enumerate(top_3_indices, 1):
    crop_name = le.inverse_transform([idx])[0]
    confidence = prediction_proba[idx]
    print(f"    {rank}. {crop_name}: {confidence:.2%}")

# ======================== PART 13: SUMMARY REPORT ========================

print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

print("\n📊 Dataset Summary:")
print(f"  Total Samples: {len(df)}")
print(f"  Features: {len(selected_features)}")
print(f"  Selected Features: {selected_features}")
print(f"  Crops: {len(le.classes_)}")

print("\n🤖 Model Performance (Test Set):")
best_accuracy = max([results[model]['accuracy'] for model in results])
best_model_name = [model for model, result in results.items() if result['accuracy'] == best_accuracy][0]
print(f"  Best Model: {best_model_name}")
print(f"  Best Accuracy: {best_accuracy:.4f}")

print("\n✅ Model Comparison:")
for model_name in sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True):
    print(f"  {model_name:20s}: {results[model_name]['accuracy']:.4f} accuracy, {results[model_name]['training_time']*1000:6.2f}ms training")

print("\n📈 Cross-Validation:")
for model_name in sorted(cv_results.keys(), key=lambda x: cv_results[x]['mean'], reverse=True):
    cv_mean = cv_results[model_name]['mean']
    cv_std = cv_results[model_name]['std']
    print(f"  {model_name:20s}: {cv_mean:.4f} ± {cv_std:.4f}")

print("\n🎯 Feature Importance (Random Forest):")
for idx, (feature, importance) in enumerate(zip(feature_importance['Feature'], feature_importance['Importance']), 1):
    percentage = (importance / feature_importance['Importance'].sum()) * 100
    print(f"  {idx}. {feature:15s}: {percentage:5.2f}% ({importance:.4f})")

print("\n📁 Generated Files:")
print("  ✓ 01_EDA_Analysis.png")
print("  ✓ 02_Feature_Selection.png")
print("  ✓ 03_Model_Performance_Comparison.png")
print("  ✓ 04_Confusion_Matrix.png")
print("  ✓ 05_Feature_Importance.png")
print("  ✓ 06_Cross_Validation_Results.png")
print("  ✓ 07_Training_Time_Comparison.png")

print("\n" + "="*80)
print("✅ AgroSmart Recommender System Completed Successfully!")
print("="*80)
