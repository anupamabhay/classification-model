# AI Assistant Plan: Baseline Classification Model

**Objective:** This document provides the complete technical specification for an AI code assistant to generate a production-quality Python script (`train_model.py`) for a binary classification task.

### **1. Core Mandate**

Your primary task is to generate the complete and runnable Python code for the `train_model.py` file described below. Additionally, you will generate the content for the `requirements.txt` file.

### **2. Required Files & Project Structure**

The generated code must operate within this exact directory structure. The script must be responsible for creating the `output/` and `output/plots/` directories.

```
classification-project/
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

### **3. Detailed Specifications for `train_model.py`**

#### **A. Global Setup & Configuration**

1. **Imports:** Import all necessary libraries: `logging`, `json`, `pathlib`, `pandas`, `numpy`, `seaborn`, `matplotlib.pyplot`, `joblib`, and relevant `sklearn` and `xgboost` modules.
    
2. **Logging:** Configure the `logging` module for INFO level with a timestamped format.
    
3. **Configuration Constants:** Create a dedicated section at the top for constants:
    
    - `DATA_PATH = Path('./data/source_data.csv')`
        
    - `OUTPUT_DIR = Path('./output')`
        
    - `PLOTS_DIR = OUTPUT_DIR / 'plots'`
        
    - `TARGET_COLUMN = 'target'`
        
    - `TEST_SIZE = 0.2`
        
    - `RANDOM_STATE = 42`
        

#### **B. Function-by-Function Requirements**

Generate the script using the following modular functions with the specified signatures and logic.

1. **`create_output_dirs() -> None:`**
    
    - Creates the `OUTPUT_DIR` and `PLOTS_DIR` using `.mkdir(exist_ok=True)`.
        
2. **`load_data(path: Path) -> pd.DataFrame:`**
    
    - Logs the start and end of the data loading process.
        
    - Loads the CSV from the provided `path` into a pandas DataFrame and returns it.
        
3. **`preprocess_and_split(df: pd.DataFrame) -> tuple:`**
    
    - Logs the start of the preprocessing and splitting step.
        
    - Separates features (X) from the `TARGET_COLUMN` (y).
        
    - Performs an 80/20 train/test split using `train_test_split`.
        
    - **Crucially**, the split must be stratified: `stratify=y`.
        
    - Use the `RANDOM_STATE` constant for reproducibility.
        
    - Returns `X_train, X_test, y_train, y_test`.
        
4. **`build_pipeline(X_train: pd.DataFrame) -> Pipeline:`**
    
    - Logs the pipeline building process.
        
    - Automatically identifies numerical and categorical feature column names from `X_train`.
        
    - Creates a `numeric_transformer` pipeline containing `StandardScaler`.
        
    - Creates a `categorical_transformer` pipeline containing `OneHotEncoder(handle_unknown='ignore')`.
        
    - Combines these transformers using a `ColumnTransformer`.
        
    - Creates the final `sklearn.pipeline.Pipeline` with two steps:
        
        1. `('preprocessor', ...)`: The `ColumnTransformer` instance.
            
        2. `('classifier', XGBClassifier(...))`: The XGBoost model, configured with `random_state=RANDOM_STATE`, `use_label_encoder=False`, and `eval_metric='logloss'`.
            
    - Returns the fully constructed (but not yet trained) pipeline object.
        
5. **`train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:`**
    
    - Logs the start and end of the model training process.
        
    - Trains the pipeline using `pipeline.fit(X_train, y_train)`.
        
6. **`evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:`**
    
    - Logs the start of the evaluation process.
        
    - Generates predictions using `pipeline.predict(X_test)`.
        
    - Calculates `accuracy`, `precision`, `recall`, and `f1_score`.
        
    - Saves these metrics as a dictionary to `OUTPUT_DIR / 'performance_metrics.json'`.
        
    - Generates and saves a confusion matrix plot using `seaborn.heatmap` to `PLOTS_DIR / 'confusion_matrix.png'`.
        
7. **`save_model(pipeline: Pipeline) -> None:`**
    
    - Logs the model saving process.
        
    - Serializes the entire trained pipeline object to `OUTPUT_DIR / 'model.joblib'` using `joblib.dump()`.
        
8. **`main() -> None:`**
    
    - The main orchestrator function.
        
    - It must call the other functions in this specific order: `create_output_dirs`, `load_data`, `preprocess_and_split`, `build_pipeline`, `train_model`, `evaluate_model`, `save_model`.
        

#### **C. Main Execution Block**

- The script must end with a standard `if __name__ == "__main__":` block that calls the `main()` function.
    

### **4. Specifications for `requirements.txt`**

Generate a `requirements.txt` file with the following exact content:

```
pandas==2.2.2
scikit-learn==1.5.0
xgboost==2.0.3
matplotlib==3.9.0
seaborn==0.13.2
```