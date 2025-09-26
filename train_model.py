"""
Advanced Production Classification Model

High-performance ensemble model with sophisticated feature engineering,
advanced preprocessing, and multiple algorithms for optimal classification.
"""

import logging
import json
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuration
DATA_PATH = Path('./data/source_data.csv')
OUTPUT_DIR = Path('./output')
PLOTS_DIR = OUTPUT_DIR / 'plots'
TARGET_COLUMN = 'target'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_output_dirs() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    logger.info("Output directories created")


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate data."""
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape}")
    logger.info(f"Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    return df


def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive engineered features."""
    df_enhanced = df.copy()
    
    # Age-based features
    df_enhanced['age_squared'] = df['age'] ** 2
    df_enhanced['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                    labels=['very_young', 'young', 'middle', 'mature', 'senior'])
    
    # Income-based features
    df_enhanced['log_income'] = np.log1p(df['income'])
    df_enhanced['income_squared'] = df['income'] ** 2
    df_enhanced['income_per_age'] = df['income'] / (df['age'] + 1)
    df_enhanced['high_income'] = (df['income'] > df['income'].quantile(0.75)).astype(int)
    
    # Credit score features
    df_enhanced['credit_squared'] = df['credit_score'] ** 2
    df_enhanced['excellent_credit'] = (df['credit_score'] >= 750).astype(int)
    df_enhanced['good_credit'] = ((df['credit_score'] >= 670) & (df['credit_score'] < 750)).astype(int)
    df_enhanced['fair_credit'] = ((df['credit_score'] >= 580) & (df['credit_score'] < 670)).astype(int)
    
    # Interaction features
    df_enhanced['income_credit_product'] = df['income'] * df['credit_score']
    df_enhanced['income_credit_ratio'] = df['income'] / (df['credit_score'] + 1)
    df_enhanced['age_income_interaction'] = df['age'] * df['income']
    df_enhanced['age_credit_interaction'] = df['age'] * df['credit_score']
    
    # Education level encoding
    education_scores = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df_enhanced['education_score'] = df['education'].map(education_scores)
    df_enhanced['high_education'] = (df_enhanced['education_score'] >= 3).astype(int)
    
    # Employment stability
    df_enhanced['stable_employment'] = (df['employment'] == 'Full-time').astype(int)
    df_enhanced['self_employed'] = (df['employment'] == 'Self-employed').astype(int)
    
    # Composite scores
    df_enhanced['financial_score'] = (
        df_enhanced['income'] / 100000 + 
        df_enhanced['credit_score'] / 850 + 
        df_enhanced['education_score'] / 4
    ) / 3
    
    df_enhanced['risk_score'] = (
        (df['age'] < 25).astype(int) * 0.3 +
        (df['income'] < 30000).astype(int) * 0.4 +
        (df['credit_score'] < 600).astype(int) * 0.3
    )
    
    logger.info(f"Advanced features engineered: {df_enhanced.shape}")
    return df_enhanced


def preprocess_and_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Advanced preprocessing with feature engineering."""
    # Engineer features
    df_engineered = advanced_feature_engineering(df)
    
    # Remove original target from features
    X = df_engineered.drop(columns=[TARGET_COLUMN])
    y = df_engineered[TARGET_COLUMN]
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train distribution: {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def build_advanced_preprocessing(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build robust preprocessing pipeline."""
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    # Use RobustScaler for numeric features (less sensitive to outliers)
    numeric_transformer = RobustScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


def build_ensemble_model() -> VotingClassifier:
    """Build ensemble model with multiple high-performance algorithms."""
    
    # Individual models with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=1.0
    )
    
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE
    )
    
    lr_model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced',
        C=1.0
    )
    
    # Ensemble with soft voting
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('gb', gb_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    
    return ensemble


def create_complete_pipeline(preprocessor: ColumnTransformer) -> ImbPipeline:
    """Create complete pipeline with SMOTE and ensemble model."""
    
    # Build ensemble
    ensemble_model = build_ensemble_model()
    
    # Complete pipeline with SMOTE for handling class imbalance
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', SelectKBest(score_func=f_classif, k=15)),
        ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto')),
        ('classifier', ensemble_model)
    ])
    
    return pipeline


def train_and_evaluate(pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Train model and perform comprehensive evaluation."""
    
    # Train the model
    logger.info("Training ensemble model...")
    pipeline.fit(X_train, y_train)
    logger.info("Training completed")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Cross-validation scores
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy'
    )
    
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    # Detailed per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log comprehensive results
    logger.info("=== MODEL PERFORMANCE ===")
    logger.info(f"ACCURACY: {metrics['accuracy']:.4f}")
    logger.info(f"PRECISION: {metrics['precision']:.4f}")
    logger.info(f"RECALL: {metrics['recall']:.4f}")
    logger.info(f"F1-SCORE: {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"CV Accuracy: {metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f}")
    
    logger.info("=== PER-CLASS PERFORMANCE ===")
    logger.info(f"Class 0 (Negative) - Precision: {class_report['0']['precision']:.4f}, Recall: {class_report['0']['recall']:.4f}")
    logger.info(f"Class 1 (Positive) - Precision: {class_report['1']['precision']:.4f}, Recall: {class_report['1']['recall']:.4f}")
    
    return metrics


def save_production_model(pipeline: ImbPipeline, metrics: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save production model and generate comprehensive outputs."""
    
    # Save model
    model_path = OUTPUT_DIR / 'production_model.joblib'
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Save metrics
    metrics_path = OUTPUT_DIR / 'performance_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")
    
    # Generate enhanced confusion matrix
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Negative (0)', 'Positive (1)'],
               yticklabels=['Negative (0)', 'Positive (1)'],
               cbar_kws={'label': 'Count'})
    
    plt.title('Production Model - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Performance summary
    acc = metrics['accuracy']
    prec = metrics['precision']
    rec = metrics['recall']
    f1 = metrics['f1_score']
    auc = metrics['roc_auc']
    
    performance_text = (f'Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}\n'
                       f'F1-Score: {f1:.3f} | ROC-AUC: {auc:.3f}')
    
    plt.figtext(0.02, 0.02, performance_text, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # Save plot
    cm_path = PLOTS_DIR / 'production_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved: {cm_path}")


def main() -> None:
    """Execute production model training pipeline."""
    logger.info("=== STARTING PRODUCTION MODEL TRAINING ===")
    
    try:
        # Setup
        create_output_dirs()
        
        # Load and preprocess data
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_and_split(df)
        
        # Build preprocessing pipeline
        preprocessor = build_advanced_preprocessing(X_train)
        
        # Create complete pipeline
        pipeline = create_complete_pipeline(preprocessor)
        
        # Train and evaluate
        metrics = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
        
        # Save results
        save_production_model(pipeline, metrics, X_test, y_test)
        
        # Performance validation
        target_accuracy = 0.80
        target_precision = 0.80
        
        if metrics['accuracy'] >= target_accuracy and metrics['precision'] >= target_precision:
            logger.info("SUCCESS: Model meets performance targets!")
            logger.info(f"Accuracy: {metrics['accuracy']:.3f} (>={target_accuracy:.2f})")
            logger.info(f"Precision: {metrics['precision']:.3f} (>={target_precision:.2f})")
        else:
            logger.warning("Performance below targets:")
            logger.warning(f"Accuracy: {metrics['accuracy']:.3f} (target: >={target_accuracy:.2f})")
            logger.warning(f"Precision: {metrics['precision']:.3f} (target: >={target_precision:.2f})")
        
        logger.info("=== PRODUCTION MODEL TRAINING COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()