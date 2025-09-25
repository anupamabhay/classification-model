# Project Guide: Baseline Classification Model

### **1. Vision & Core Philosophy**

This project's goal is to create a **production-ready, reproducible baseline classification model**. Our philosophy separates the workflow into two distinct, equally important phases:

1. **Engineering (`train_model.py`):** We first build a robust, automated, and reliable Python script. This is our "engine"—the core, reusable asset that performs the work. It is developed and finalized in a professional IDE like Visual Studio 2022.
    
2. **Storytelling (`documentation.ipynb`):** After the engine is built, we create a Jupyter Notebook to explain and demonstrate the results. This is our "walkthrough"—an interactive document that tells the story of our findings to colleagues and stakeholders.
    

This script-first approach ensures that our core logic is solid and reproducible before we focus on presentation.

### **2. Environment & Project Setup**

A clean, version-controlled environment is mandatory. Follow these steps precisely.

1. **Create Project Directory:**
    
    ```
    mkdir classification-project
    cd classification-project
    ```
    
2. **Initialize Git:**
    
    ```
    git init
    ```
    
3. **Create a Virtual Environment:**
    
    ```
    # For macOS/Linux
    python3 -m venv venv
    
    # For Windows
    python -m venv venv
    ```
    
4. **Activate the Environment:**
    
    ```
    # For macOS/Linux
    source venv/bin/activate
    
    # For Windows
    .\venv\Scripts\activate
    ```
    
    _Your terminal prompt should now be prefixed with `(venv)`._
    
5. **Install Dependencies:**
    
    - Create a `requirements.txt` file with the following pinned versions:
        
        ```
        pandas==2.2.2
        scikit-learn==1.5.0
        xgboost==2.0.3
        matplotlib==3.9.0
        seaborn==0.13.2
        ```
        
    - Install them:
        
        ```
        pip install -r requirements.txt
        ```
        
6. **Create Project Structure:**
    
    ```
    mkdir data output
    touch data/.gitkeep output/.gitkeep train_model.py documentation.ipynb .gitignore
    ```
    
7. **Configure `.gitignore`:**
    
    - Add the following to your `.gitignore` file to prevent committing unnecessary files:
        
        ```
        # Virtual Environment
        venv/
        
        # Python cache
        __pycache__/
        *.pyc
        
        # IDE settings
        .vscode/
        .idea/
        
        # Output data
        output/
        ```
        

### **3. The Development Workflow: A Detailed Guide**

#### **Phase A: Engineering the Script in Visual Studio 2022 (`train_model.py`)**

**Goal:** Build the complete, automated training script from start to finish.

1. **Interactive Prototyping (using VS Code's features):**
    
    - Open `train_model.py`. Use `#%%` comments to create "cells". This allows you to run chunks of code interactively in the "Interactive Window" without needing a separate notebook for development.
        
    - **Load & Validate:** In the first cell, load your data with Pandas. Immediately validate it—check `.info()`, look for duplicates with `.duplicated().sum()`, and examine the first few rows with `.head()`.
        
    - **Explore:** In subsequent cells, perform your EDA. Create plots (e.g., `seaborn.countplot`, `seaborn.heatmap`) to understand distributions and correlations. Since you're in a `.py` file, Copilot will provide excellent assistance.
        
    - **Finalize Strategy:** Based on your exploration, make firm decisions on your preprocessing logic (e.g., "Use Median imputation for `age`," "Use One-Hot Encoding for `category`").
        
2. **Refactoring into a Production Script:**
    
    - Once your strategy is clear, refactor the interactive cells into a structured, modular script.
        
    - **Create Functions:** Each logical step (loading, splitting, building the pipeline, evaluating) becomes a separate function.
        
    - **Apply Best Practices:**
        
        - **Type Hinting:** Add type hints to all function signatures (e.g., `def load_data(path: Path) -> pd.DataFrame:`).
            
        - **Logging:** Use Python's `logging` module for all informational output. This is superior to `print()` as it can be configured, timed, and leveled.
            
        - **Configuration:** Hardcoded values (like file paths or model parameters) should be defined as constants at the top of the script. Use `pathlib` for robust path handling.
            
    - **Build the Main Orchestrator:** Create the `if __name__ == "__main__":` block to call all your functions in the correct sequence.
        

#### **Phase B: Storytelling with the Documentation Notebook (`documentation.ipynb`)**

**Goal:** Create a clean, presentable document for demonstrating the project's results. This is done _after_ `train_model.py` is complete and functional.

1. **Open `documentation.ipynb` in VS Code.**
    
2. **Introduction:** Start with a markdown cell explaining the business problem, the dataset, and the project's goal.
    
3. **Load the Artifacts:** In a code cell, use `joblib` to load your saved `model.joblib` and the `json` module to load `performance_metrics.json`.
    
4. **Display Results:**
    
    - Print the performance metrics in a clean, readable format.
        
    - Load and display the saved `confusion_matrix.png` image.
        
5. **Explain the Model:** Add markdown cells to explain what the results mean. For example, describe what the precision and recall scores signify in the context of the business problem.
    
6. **Next Steps:** Conclude with a summary and suggest potential next steps for future sprints (e.g., hyperparameter tuning, feature engineering).
    

### **4. Execution & Verification**

1. **Place Data:** Ensure your `source_data.csv` is in the `/data` directory.
    
2. **Run the Script:** From your activated virtual environment, run the main script from the project's root directory:
    
    ```
    python train_model.py
    ```
    
3. **Verify Outputs:** After a successful run, check the `/output` directory. It should contain:
    
    - `model.joblib`: The saved model pipeline.
        
    - `performance_metrics.json`: The evaluation scores.
        
    - `/plots/confusion_matrix.png`: The saved visualization.