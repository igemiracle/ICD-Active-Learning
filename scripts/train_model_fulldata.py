#!/usr/bin/env python
# coding: utf-8

"""
MIMIC-III Multi-label ICD code classification with cross-validation
Uses all available CPUs for parallel training of TF-IDF + Logistic Regression models
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
import pickle
import logging
from datetime import datetime

# Set up logging
log_dir = "../logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create results directory for outputs
results_dir = "../results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
models_dir = os.path.join(results_dir, "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
plots_dir = os.path.join(results_dir, "plots")
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# This section is deleted because we moved it above the data loading

# Record start time
start_time = time.time()
logging.info(f"Starting training process at {timestamp}")

# Load MIMIC-III data
logging.info("Loading preprocessed MIMIC-III data...")
data_path = '../data/'
processed_data_file = os.path.join(data_path, 'mimic3_data.pkl')
data = pd.read_pickle(processed_data_file)
logging.info(f"Loaded {len(data)} records")

# Create train and test sets (80% train, 20% test)
logging.info("Creating train and test splits...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
logging.info(f"Created splits: {len(train_data)} train, {len(test_data)} test")

# Process ICD codes - build a set of all unique codes
logging.info("Processing ICD codes...")
all_codes = set()
for codes in data['ICD9_CODE']:
    # Convert any non-string codes to strings and filter out NaN values
    valid_codes = [str(code) for code in codes if pd.notna(code)]
    all_codes.update(valid_codes)

# Now all codes should be strings, so sorting will work
code_to_idx = {code: i for i, code in enumerate(sorted(all_codes))}
idx_to_code = {i: code for code, i in code_to_idx.items()}
num_codes = len(code_to_idx)
logging.info(f"Total unique ICD-9 codes: {num_codes}")

# Save code mapping for later use
code_mapping_dir = os.path.join(results_dir, "code_mapping")
if not os.path.exists(code_mapping_dir):
    os.makedirs(code_mapping_dir)
with open(os.path.join(code_mapping_dir, "code_to_idx.pkl"), 'wb') as f:
    pickle.dump(code_to_idx, f)
with open(os.path.join(code_mapping_dir, "idx_to_code.pkl"), 'wb') as f:
    pickle.dump(idx_to_code, f)
logging.info(f"Saved code mappings to {code_mapping_dir}")

# Create features and labels
def prepare_data(dataset):
    texts = dataset['TEXT'].tolist()
    
    # Convert ICD codes to multi-hot encoded vectors
    labels = np.zeros((len(dataset), num_codes), dtype=np.int8)
    for i, codes in enumerate(dataset['ICD9_CODE']):
        for code in codes:
            if code in code_to_idx:  # Just in case there's an unknown code
                labels[i, code_to_idx[code]] = 1
    
    return texts, labels

logging.info("Preparing data...")
train_texts, train_labels = prepare_data(train_data)
test_texts, test_labels = prepare_data(test_data)

# Create TF-IDF features
logging.info("Creating TF-IDF features...")
start_time = time.time()
tfidf = TfidfVectorizer(
    max_features=10000,  # Limit features to reduce dimensionality
    min_df=5,            # Ignore terms that appear in less than 5 documents
    max_df=0.5,          # Ignore terms that appear in more than 50% of documents
    ngram_range=(1, 2),  # Use unigrams and bigrams
    stop_words='english' # Remove English stop words
)
train_features = tfidf.fit_transform(train_texts)
test_features = tfidf.transform(test_texts)
logging.info(f"TF-IDF features shape: {train_features.shape}")
logging.info(f"TF-IDF processing time: {time.time() - start_time:.2f} seconds")

# Save the TF-IDF vectorizer for later use
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf, f)
logging.info(f"TF-IDF vectorizer saved to {vectorizer_path}")

# Function to evaluate the model
def evaluate(model, features, labels):
    """Evaluate the model and return metrics"""
    # Get predictions
    pred_probs = model.predict_proba(features)
    
    # Apply threshold of 0.5 for binary classification
    preds = np.zeros_like(labels)
    for i, idx in enumerate(model.trained_indices):
        if i < len(pred_probs):
            preds[:, idx] = (pred_probs[i][:, 1] >= 0.5).astype(int)
    
    # Calculate micro and macro F1 scores
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Calculate micro and macro precision and recall
    micro_precision = precision_score(labels, preds, average='micro', zero_division=0)
    macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
    micro_recall = recall_score(labels, preds, average='micro', zero_division=0)
    macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
    
    # Count non-zero predictions and true positives
    pred_counts = np.sum(preds, axis=0)
    true_counts = np.sum(labels, axis=0)
    
    # Calculate per-class metrics for the top 10 most common codes
    top_codes_idx = np.argsort(-true_counts)[:10]
    per_class_metrics = {}
    
    for idx in top_codes_idx:
        code = idx_to_code[idx]
        tp = np.sum((preds[:, idx] == 1) & (labels[:, idx] == 1))
        fp = np.sum((preds[:, idx] == 1) & (labels[:, idx] == 0))
        fn = np.sum((preds[:, idx] == 0) & (labels[:, idx] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[code] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': true_counts[idx]
        }
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'micro_recall': micro_recall,
        'macro_recall': macro_recall,
        'per_class_metrics': per_class_metrics,
        'pred_counts': pred_counts,
        'true_counts': true_counts
    }

# Create results directory for outputs
results_dir = "../results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
models_dir = os.path.join(results_dir, "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
plots_dir = os.path.join(results_dir, "plots")
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Set up cross-validation
logging.info("Setting up cross-validation...")
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_results = {
    'fold': [],
    'micro_f1': [],
    'macro_f1': [],
    'micro_precision': [],
    'macro_precision': [],
    'micro_recall': [],
    'macro_recall': [],
    'train_time': []
}

# Function to train a single classifier
def train_single_model(i, X_train, y_train):
    """Train a single classifier for the i-th code"""
    logging.debug(f"Training classifier for code {i+1}/{num_codes}")
    
    # Check if there are both positive and negative examples for this code
    if np.sum(y_train[:, i] == 0) > 0 and np.sum(y_train[:, i] == 1) > 0:
        estimator = LogisticRegression(
            C=1.0,
            solver='saga',
            penalty='l1',
            max_iter=500,
            verbose=0,
            random_state=42
        )
        try:
            estimator.fit(X_train, y_train[:, i])
            return (i, estimator)
        except Exception as e:
            logging.error(f"Error training classifier for code {i}: {e}")
            return (i, None)
    else:
        # For codes with only one class, create a dummy classifier that always predicts that class
        majority_class = int(np.bincount(y_train[:, i]).argmax())
        dummy = DummyClassifier(strategy='constant', constant=majority_class)
        dummy.fit(X_train, y_train[:, i])
        return (i, dummy)

# Perform cross-validation
logging.info("Performing cross-validation...")
fold = 1
best_model = None
best_micro_f1 = 0

for train_idx, val_idx in kf.split(train_features):
    fold_start_time = time.time()
    logging.info(f"\nTraining fold {fold}/{n_folds}...")
    
    # Get fold data
    X_train_fold, X_val_fold = train_features[train_idx], train_features[val_idx]
    y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]
    
    # Train classifiers in parallel
    logging.info(f"Training classifiers in parallel using all available CPUs...")
    n_jobs = os.cpu_count()  # Use all available CPUs
    logging.info(f"Using {n_jobs} CPU cores")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_single_model)(i, X_train_fold, y_train_fold) 
        for i in range(num_codes)
    )
    
    # Process results
    estimators = []
    trained_indices = []
    for i, estimator in results:
        if estimator is not None:
            estimators.append(estimator)
            trained_indices.append(i)
    
    logging.info(f"Trained {len(estimators)}/{num_codes} classifiers")
    
    # Create a model object to hold the estimators
    model = MultiOutputClassifier(LogisticRegression())
    model.estimators_ = estimators
    model.classes_ = [np.array([0, 1]) for _ in estimators]
    model.trained_indices = trained_indices  # Store which indices were trained
    
    # Modify the predict_proba function to handle our custom structure
    def custom_predict_proba(self, X):
        probas = []
        for i, estimator in enumerate(self.estimators_):
            # For regular LogisticRegression models
            if isinstance(estimator, LogisticRegression):
                probas.append(estimator.predict_proba(X))
            # For DummyClassifier models
            elif isinstance(estimator, DummyClassifier):
                constant_class = estimator.constant_
                n_samples = X.shape[0]
                if constant_class == 0:
                    # Always predict class 0
                    proba = np.zeros((n_samples, 2))
                    proba[:, 0] = 1.0
                else:
                    # Always predict class 1
                    proba = np.zeros((n_samples, 2))
                    proba[:, 1] = 1.0
                probas.append(proba)
        return probas
    
    # Attach the custom method to the model
    import types
    model.predict_proba = types.MethodType(custom_predict_proba, model)
    
    # Evaluate on validation set
    metrics = evaluate(model, X_val_fold, y_val_fold)
    
    # Calculate training time
    fold_time = time.time() - fold_start_time
    
    logging.info(f"Fold {fold} validation results:")
    logging.info(f"  Micro F1: {metrics['micro_f1']:.4f}")
    logging.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
    logging.info(f"  Training time: {fold_time:.2f} seconds")
    
    # Save metrics
    cv_results['fold'].append(fold)
    cv_results['train_time'].append(fold_time)
    for key, value in metrics.items():
        cv_results[key].append(value)
    
    # Keep track of best model
    if metrics['micro_f1'] > best_micro_f1:
        best_micro_f1 = metrics['micro_f1']
        best_model = model
        logging.info(f"  New best model found with micro F1: {best_micro_f1:.4f}")
        
        # Save the best model
        best_model_path = os.path.join(models_dir, f"best_model_fold_{fold}.pkl")
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        logging.info(f"  Best model saved to {best_model_path}")
    
    fold += 1

# Calculate average metrics across folds
avg_metrics = {
    'micro_f1': np.mean(cv_results['micro_f1']),
    'macro_f1': np.mean(cv_results['macro_f1']),
    'micro_precision': np.mean(cv_results['micro_precision']),
    'macro_precision': np.mean(cv_results['macro_precision']),
    'micro_recall': np.mean(cv_results['micro_recall']),
    'macro_recall': np.mean(cv_results['macro_recall']),
    'train_time': np.mean(cv_results['train_time'])
}

logging.info("\nCross-validation complete!")
logging.info("Average metrics across all folds:")
for key, value in avg_metrics.items():
    logging.info(f"  {key}: {value:.4f}")

# Save cross-validation results to CSV
cv_results_df = pd.DataFrame(cv_results)
cv_results_path = os.path.join(results_dir, f"cv_results_{timestamp}.csv")
cv_results_df.to_csv(cv_results_path, index=False)
logging.info(f"Cross-validation results saved to {cv_results_path}")

# Visualize results
logging.info("Creating visualizations...")

# Plot F1 scores across folds
plt.figure(figsize=(10, 6))
plt.plot(cv_results['fold'], cv_results['micro_f1'], 'o-', label='Micro F1')
plt.plot(cv_results['fold'], cv_results['macro_f1'], 'o-', label='Macro F1')
plt.title('F1 Scores Across Folds')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
f1_plot_path = os.path.join(plots_dir, f"f1_scores_{timestamp}.png")
plt.savefig(f1_plot_path)
logging.info(f"F1 scores plot saved to {f1_plot_path}")

# Plot all metrics in one figure
plt.figure(figsize=(12, 8))
metrics_to_plot = ['micro_f1', 'macro_f1', 'micro_precision', 'macro_precision', 'micro_recall', 'macro_recall']
for metric in metrics_to_plot:
    plt.plot(cv_results['fold'], cv_results[metric], 'o-', label=metric)
plt.title('All Metrics Across Folds')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
all_metrics_plot_path = os.path.join(plots_dir, f"all_metrics_{timestamp}.png")
plt.savefig(all_metrics_plot_path)
logging.info(f"All metrics plot saved to {all_metrics_plot_path}")

# Create a bar chart with average metrics
plt.figure(figsize=(10, 6))
metrics_names = ['micro_f1', 'macro_f1', 'micro_precision', 'macro_precision', 'micro_recall', 'macro_recall']
avg_values = [avg_metrics[m] for m in metrics_names]
plt.bar(metrics_names, avg_values, color='skyblue')
plt.ylim(0, 1.0)
plt.title('Average Metrics Across All Folds')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
for i, v in enumerate(avg_values):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.tight_layout()
avg_metrics_plot_path = os.path.join(plots_dir, f"avg_metrics_{timestamp}.png")
plt.savefig(avg_metrics_plot_path)
logging.info(f"Average metrics plot saved to {avg_metrics_plot_path}")

# Visualize top 10 ICD codes performance
# Extract per-class metrics from the best model validation
if 'per_class_metrics' in metrics:
    plt.figure(figsize=(14, 8))
    
    # Get codes and their metrics
    codes = list(metrics['per_class_metrics'].keys())
    f1_scores = [metrics['per_class_metrics'][code]['f1'] for code in codes]
    precisions = [metrics['per_class_metrics'][code]['precision'] for code in codes]
    recalls = [metrics['per_class_metrics'][code]['recall'] for code in codes]
    supports = [metrics['per_class_metrics'][code]['support'] for code in codes]
    
    # Sort by support (frequency)
    sorted_indices = np.argsort(supports)[::-1]
    codes = [codes[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    
    # Create bar positions
    x = np.arange(len(codes))
    width = 0.25
    
    # Plot bars
    plt.bar(x - width, precisions, width, label='Precision', color='skyblue')
    plt.bar(x, recalls, width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1_scores, width, label='F1 Score', color='salmon')
    
    # Add labels and formatting
    plt.xlabel('ICD-9 Code')
    plt.ylabel('Score')
    plt.title('Performance Metrics for Top 10 Most Common ICD Codes')
    plt.xticks(x, codes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    top_codes_plot_path = os.path.join(plots_dir, f"top_codes_performance_{timestamp}.png")
    plt.savefig(top_codes_plot_path)
    logging.info(f"Top codes performance plot saved to {top_codes_plot_path}")
    
    # Create a detailed table of per-class metrics
    plt.figure(figsize=(12, len(codes)*0.5 + 2))
    plt.axis('off')
    table_data = []
    table_data.append(['ICD Code', 'Precision', 'Recall', 'F1 Score', 'Support'])
    for code in codes:
        metrics_dict = metrics['per_class_metrics'][code]
        table_data.append([
            code, 
            f"{metrics_dict['precision']:.3f}", 
            f"{metrics_dict['recall']:.3f}", 
            f"{metrics_dict['f1']:.3f}", 
            f"{metrics_dict['support']}"
        ])
    
    table = plt.table(cellText=table_data, colWidths=[0.25, 0.15, 0.15, 0.15, 0.15], 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Detailed Metrics for Top ICD Codes', y=0.9)
    
    table_path = os.path.join(plots_dir, f"top_codes_table_{timestamp}.png")
    plt.savefig(table_path, bbox_inches='tight')
    logging.info(f"Top codes metrics table saved to {table_path}")

# Calculate total execution time
total_time = time.time() - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
logging.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

# Save a summary report
summary_path = os.path.join(results_dir, f"summary_{timestamp}.txt")
with open(summary_path, 'w') as f:
    f.write(f"Training Summary Report\n")
    f.write(f"=====================\n\n")
    f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Number of samples: {train_features.shape[0]}\n")
    f.write(f"Number of features: {train_features.shape[1]}\n")
    f.write(f"Number of codes (labels): {num_codes}\n")
    f.write(f"Number of folds: {n_folds}\n\n")
    
    f.write(f"Average Metrics:\n")
    for key, value in avg_metrics.items():
        f.write(f"  {key}: {value:.4f}\n")
    
    f.write(f"\nBest Model:\n")
    f.write(f"  Found in fold: {cv_results['fold'][np.argmax(cv_results['micro_f1'])]}\n")
    f.write(f"  Micro F1: {best_micro_f1:.4f}\n\n")
    
    f.write(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")

logging.info(f"Summary report saved to {summary_path}")
logging.info("Training process completed successfully!")