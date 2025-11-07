#!/usr/bin/env python3
"""
Logistic Regression baseline for Spanish-English code-switching detection
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from typing import List, Dict

# --------------------------------------------------------
# 1. Load your Spanish-English dataset
# --------------------------------------------------------
def load_your_data(json_path="../data/spanglish_dataset.json"):
    """Load your Spanish-English dataset."""
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    data = []
    for conv in dataset['conversations']:
        for utt in conv['utterances']:
            # Extract tokens and their language labels from tagged_tokens
            tokens = []
            labels = []
            
            for token_pair in utt.get('tagged_tokens', []):
                if len(token_pair) == 2:
                    token, lang = token_pair
                    if lang in ['en', 'es']:  # Skip punctuation
                        tokens.append(token)
                        labels.append(lang)
            
            if len(tokens) > 1:  # Need at least 2 tokens to detect switches
                data.append({'tokens': tokens, 'labels': labels})
    
    print(f"Loaded {len(data)} utterances with tokens")
    return data

# --------------------------------------------------------
# 2. Create switch prediction labels
# --------------------------------------------------------
def create_switch_prediction_labels(data: List[Dict]) -> List[Dict]:
    """Create binary labels for switch points between tokens."""
    for example in data:
        switch_labels = []
        for i in range(len(example['labels']) - 1):
            # 1 if language changes, 0 if stays the same
            switch_labels.append(1 if example['labels'][i] != example['labels'][i + 1] else 0)
        example["switch_labels"] = switch_labels
    return data

# --------------------------------------------------------
# 3. Prepare token-level context data for ML model
# --------------------------------------------------------
def flatten_examples(data: List[Dict]):
    """Create context-label pairs for training."""
    contexts, labels = [], []
    for ex in data:
        tokens = ex['tokens']
        switch_labels = ex['switch_labels']
        for i in range(len(switch_labels)):
            # Context: all tokens up to current position
            context = " ".join(tokens[:i+1])
            contexts.append(context)
            labels.append(switch_labels[i])
    return contexts, labels

# --------------------------------------------------------
# 4. Train and evaluate Logistic Regression model
# --------------------------------------------------------
def train_logreg_model(json_path="../data/spanglish_dataset.json"):
    print("Loading Spanish-English dataset...")
    data = load_your_data(json_path)
    data = create_switch_prediction_labels(data)
    contexts, labels = flatten_examples(data)
    
    print(f"Total context-label pairs: {len(contexts)}")
    print(f"Switch points: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    
    # TF-IDF representation
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(contexts)
    y = np.array(labels)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION RESULTS")
    print("="*50)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["no_switch", "switch"]))
    
    return clf, vectorizer

if __name__ == "__main__":
    # Train the model
    model, vectorizer = train_logreg_model()