# AI Assistant Plan: Baseline Classification Model

**Objective:** This document provides the complete technical specification for an AI code assistant to generate a production-quality Python script (`train_model.py`) for a binary classification task.

## ✅ Phase 1: Core Implementation (COMPLETED)

### 1. Core Mandate ✅
Generate the complete and runnable Python code for the `train_model.py` file with production-ready standards.

### 2. Required Files & Project Structure ✅
```
classification-model/
├── data/
│   └── source_data.csv
├── output/
│   ├── plots/
│   │   └── confusion_matrix.png
│   ├── model.joblib
│   └── performance_metrics.json
├── train_model.py
└── requirements.txt
```

### 3. Core Implementation Achievements ✅
- [x] Modular function architecture with type hints
- [x] Comprehensive logging system
- [x] Automated preprocessing pipeline
- [x] XGBoost classifier with optimal configuration
- [x] Complete evaluation suite with multiple metrics
- [x] Professional error handling and documentation
- [x] Production-ready deployment capabilities

---

## 🚀 Phase 2: Enhanced Model Optimization

### Objective
Enhance the baseline model with advanced ML techniques and hyperparameter optimization.

### 2.1 Hyperparameter Optimization
**Implementation Target:** `train_model_optimized.py`

```python
# Enhanced pipeline with hyperparameter tuning
def build_optimized_pipeline(X_train: pd.DataFrame) -> Pipeline:
    # Add GridSearchCV for hyperparameter optimization
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 6, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 0.9, 1.0]
    }
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=RANDOM_STATE))
    ])
    
    # GridSearchCV with cross-validation
    optimized_pipeline = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    return optimized_pipeline
```

### 2.2 Cross-Validation Enhancement
```python
def evaluate_with_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model using cross-validation."""
    cv_scores = cross_validate(
        pipeline, X, y, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
        return_train_score=True
    )
    
    return {
        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
        'cv_f1_std': cv_scores['test_f1_weighted'].std(),
        'train_test_gap': cv_scores['train_accuracy'].mean() - cv_scores['test_accuracy'].mean()
    }
```

### 2.3 Feature Engineering & Selection
```python
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to improve model performance."""
    df_enhanced = df.copy()
    
    # Numerical feature interactions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            df_enhanced[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
            df_enhanced[f'{col1}_{col2}_product'] = df[col1] * df[col2]
    
    return df_enhanced

def add_feature_selection(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Add feature selection to the pipeline."""
    from sklearn.feature_selection import SelectKBest, f_classif
    
    enhanced_pipeline = Pipeline([
        ('preprocessor', pipeline.named_steps['preprocessor']),
        ('feature_selector', SelectKBest(score_func=f_classif, k=20)),
        ('classifier', pipeline.named_steps['classifier'])
    ])
    
    return enhanced_pipeline
```

### 2.4 Model Interpretability
```python
def analyze_feature_importance(pipeline: Pipeline, feature_names: list) -> pd.DataFrame:
    """Extract and analyze feature importance using SHAP."""
    import shap
    
    # Get trained model
    model = pipeline.named_steps['classifier']
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    X_processed = pipeline.named_steps['preprocessor'].transform(X_test)
    shap_values = explainer.shap_values(X_processed)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)
    
    return feature_importance
```

---

## 🔧 Phase 3: Production API & Deployment

### 3.1 FastAPI Web Service
**Implementation Target:** `api/main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict

app = FastAPI(title="Classification Model API", version="1.0.0")

# Load model at startup
model = joblib.load("output/model.joblib")

class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0].tolist()
        confidence = max(probabilities)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probabilities,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

### 3.2 Docker Containerization
**Implementation Target:** `Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.3 Model Monitoring
**Implementation Target:** `monitoring/model_monitor.py`

```python
class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.reference_stats = self._calculate_stats(reference_data)
    
    def detect_drift(self, new_data: pd.DataFrame, threshold: float = 0.05) -> Dict:
        """Detect data drift using statistical tests."""
        drift_results = {}
        
        for column in self.reference_data.columns:
            if column in new_data.columns:
                # Kolmogorov-Smirnov test for numerical features
                if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                    from scipy import stats
                    statistic, p_value = stats.ks_2samp(
                        self.reference_data[column].dropna(),
                        new_data[column].dropna()
                    )
                    drift_results[column] = {
                        'drift_detected': p_value < threshold,
                        'p_value': p_value,
                        'test': 'ks_2samp'
                    }
        
        return drift_results
    
    def track_performance(self, predictions: np.array, actuals: np.array) -> Dict:
        """Track model performance over time."""
        from sklearn.metrics import accuracy_score, f1_score
        
        current_performance = {
            'accuracy': accuracy_score(actuals, predictions),
            'f1_score': f1_score(actuals, predictions, average='weighted'),
            'timestamp': datetime.now().isoformat()
        }
        
        return current_performance
```

---

## 📊 Phase 4: MLOps & Advanced Analytics

### 4.1 Automated Retraining Pipeline
**Implementation Target:** `mlops/retrain_pipeline.py`

```python
class AutomatedRetraining:
    def __init__(self, performance_threshold: float = 0.85):
        self.performance_threshold = performance_threshold
        
    def check_retrain_trigger(self, current_performance: float) -> bool:
        """Determine if model needs retraining."""
        return current_performance < self.performance_threshold
    
    def retrain_model(self, new_data: pd.DataFrame) -> Pipeline:
        """Retrain model with new data."""
        # Combine with historical data
        combined_data = self._combine_datasets(new_data)
        
        # Run training pipeline
        X_train, X_test, y_train, y_test = preprocess_and_split(combined_data)
        pipeline = build_pipeline(X_train)
        train_model(pipeline, X_train, y_train)
        
        # Validate performance
        performance = evaluate_model(pipeline, X_test, y_test)
        
        if performance['accuracy'] > self.performance_threshold:
            self._backup_current_model()
            save_model(pipeline)
            return pipeline
        else:
            raise ValueError("Retrained model performance below threshold")
```

### 4.2 A/B Testing Framework
**Implementation Target:** `experiments/ab_testing.py`

```python
class ABTestManager:
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(self, name: str, model_a: Pipeline, model_b: Pipeline, 
                         traffic_split: float = 0.5):
        """Create new A/B test experiment."""
        self.experiments[name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'results_a': [],
            'results_b': []
        }
    
    def route_traffic(self, experiment_name: str, user_id: str) -> str:
        """Route user to model A or B based on hash."""
        import hashlib
        
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        split_point = self.experiments[experiment_name]['traffic_split']
        
        return 'model_a' if (hash_value % 100) / 100 < split_point else 'model_b'
    
    def analyze_experiment(self, experiment_name: str) -> Dict:
        """Analyze A/B test results."""
        exp = self.experiments[experiment_name]
        
        from scipy import stats
        
        # Statistical significance test
        results_a = np.array(exp['results_a'])
        results_b = np.array(exp['results_b'])
        
        statistic, p_value = stats.ttest_ind(results_a, results_b)
        
        return {
            'model_a_mean': results_a.mean(),
            'model_b_mean': results_b.mean(),
            'improvement': (results_b.mean() - results_a.mean()) / results_a.mean(),
            'statistical_significance': p_value < 0.05,
            'p_value': p_value
        }
```

### 4.3 Advanced Analytics Dashboard
**Implementation Target:** `dashboard/streamlit_app.py`

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def create_dashboard():
    st.title("🎯 Classification Model Analytics Dashboard")
    
    # Model Performance Section
    st.header("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "88.0%", "↑ 2.1%")
    with col2:
        st.metric("Precision", "87.7%", "↑ 1.5%")
    with col3:
        st.metric("Recall", "88.0%", "↑ 0.8%")
    with col4:
        st.metric("F1-Score", "87.8%", "↑ 1.2%")
    
    # Performance Over Time
    st.subheader("Performance Trends")
    performance_data = load_performance_history()
    
    fig = px.line(performance_data, x='date', y=['accuracy', 'f1_score'], 
                  title="Model Performance Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("🔍 Feature Importance Analysis")
    importance_data = load_feature_importance()
    
    fig_importance = px.bar(importance_data.head(10), 
                           x='importance', y='feature',
                           title="Top 10 Most Important Features",
                           orientation='h')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Data Drift Detection
    st.subheader("🚨 Data Drift Monitoring")
    drift_data = load_drift_analysis()
    
    drift_summary = drift_data['drift_detected'].value_counts()
    fig_drift = px.pie(values=drift_summary.values, 
                      names=drift_summary.index,
                      title="Data Drift Detection Summary")
    st.plotly_chart(fig_drift, use_container_width=True)
```

---

## ✅ Updated Implementation Checklist

### Phase 1: Core Implementation ✅
- [x] Basic training pipeline (`train_model.py`)
- [x] Automated preprocessing
- [x] Model evaluation and metrics
- [x] Comprehensive documentation
- [x] Production-ready code structure

### Phase 2: Model Optimization 🔄
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation implementation
- [ ] Feature engineering pipeline
- [ ] Feature selection optimization
- [ ] SHAP-based model interpretability
- [ ] Advanced evaluation metrics

### Phase 3: Production API 🆕
- [ ] FastAPI web service
- [ ] Docker containerization
- [ ] Model monitoring system
- [ ] Performance tracking
- [ ] Health checks and logging
- [ ] Security and authentication

### Phase 4: MLOps Integration 🆕
- [ ] Automated retraining pipeline
- [ ] A/B testing framework
- [ ] CI/CD pipeline setup
- [ ] Advanced analytics dashboard
- [ ] Model registry and versioning
- [ ] Alert system for drift/performance

### Phase 5: Enterprise Features 🆕
- [ ] Multi-model ensemble system
- [ ] Real-time streaming predictions
- [ ] Data quality validation
- [ ] Compliance and audit trails
- [ ] Cost optimization monitoring
- [ ] Integration with external systems

---

## 🎯 Success Metrics

### Technical KPIs
- **Model Performance**: Maintain >85% accuracy
- **API Latency**: <100ms response time
- **System Uptime**: >99.9% availability
- **Data Freshness**: Models updated within 24h of drift detection

### Business KPIs
- **Prediction Accuracy**: Business metric improvement
- **Cost Efficiency**: Reduced manual intervention by 80%
- **Time to Deployment**: New models live within 1 hour
- **Scalability**: Handle 1000+ requests per second

This enhanced roadmap provides a clear path from the current baseline to a comprehensive, enterprise-ready ML system with full MLOps capabilities.