"""
Baseline Classification Model Training Script

This script implements a production-ready classification pipeline using XGBoost.
It follows a modular, function-based approach with comprehensive logging and
automated output generation.
"""

import logging
import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier


# Configuration Constants
DATA_PATH = Path('./data/source_data.csv')
OUTPUT_DIR = Path('./output')
PLOTS_DIR = OUTPUT_DIR / 'plots'
TARGET_COLUMN = 'target'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_output_dirs() -> None:
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    logger.info("Output directories created successfully")


def load_data(path: Path) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    logger.info(f"Starting data loading from {path}")
    df = pd.read_csv(path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def preprocess_and_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess data and split into train/test sets.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting preprocessing and data splitting")
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Perform stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"Data split completed. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Build the preprocessing and modeling pipeline.
    
    Args:
        X_train: Training feature data
        
    Returns:
        Configured sklearn Pipeline
    """
    logger.info("Building preprocessing and modeling pipeline")
    
    # Identify numerical and categorical columns
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numerical features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")
    
    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create the complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])
    
    logger.info("Pipeline built successfully")
    return pipeline


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Train the model pipeline.
    
    Args:
        pipeline: The sklearn Pipeline to train
        X_train: Training feature data
        y_train: Training target data
    """
    logger.info("Starting model training")
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed successfully")


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the trained model and save results.
    
    Args:
        pipeline: Trained sklearn Pipeline
        X_test: Test feature data
        y_test: Test target data
    """
    logger.info("Starting model evaluation")
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Save metrics to JSON
    metrics_path = OUTPUT_DIR / 'performance_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Performance metrics saved to {metrics_path}")
    logger.info(f"Model Performance - Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = PLOTS_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {cm_path}")


def save_model(pipeline: Pipeline) -> None:
    """
    Save the trained model pipeline to disk.
    
    Args:
        pipeline: Trained sklearn Pipeline
    """
    logger.info("Saving trained model")
    model_path = OUTPUT_DIR / 'model.joblib'
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved successfully to {model_path}")


def main() -> None:
    """Main orchestrator function."""
    logger.info("Starting classification model training pipeline")
    
    try:
        # Execute the complete pipeline
        create_output_dirs()
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_and_split(df)
        pipeline = build_pipeline(X_train)
        train_model(pipeline, X_train, y_train)
        evaluate_model(pipeline, X_test, y_test)
        save_model(pipeline)
        
        logger.info("Classification model training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()