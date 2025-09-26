# Jupyter Notebook Setup Guide - Classification Model Documentation

## Overview

This guide provides step-by-step instructions for setting up and using the `documentation.ipynb` Jupyter notebook to explore and present the classification model results interactively.

## Prerequisites

  - Completed Phase 1 implementation (`train_model.py` executed successfully).
  - The Python virtual environment must be activated.
  - All dependencies must be installed from `requirements.txt`.

## Quick Setup

### 1\. Install Jupyter Notebook

```bash
# Ensure you're in the project directory and the virtual environment is active
# cd classification-model
# .\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install Jupyter notebook and the kernel
pip install notebook ipykernel
```

### 2\. Register the Virtual Environment as a Jupyter Kernel

```bash
# Register the project environment as a Jupyter kernel
python -m ipykernel install --user --name=classification-model --display-name="Classification Model"
```

### 3\. Install Additional Notebook Dependencies

```bash
# Install packages for enhanced notebook functionality
pip install jupyterlab-widgets ipywidgets plotly kaleido
```

-----

## Running the Documentation Notebook

### Method 1: Command Line Launch

```bash
# From the project root directory
jupyter notebook documentation.ipynb
```

### Method 2: Jupyter Lab (Recommended)

```bash
# Install JupyterLab for a better experience
pip install jupyterlab

# Launch JupyterLab
jupyter lab documentation.ipynb
```

### Method 3: VS Code Integration

1.  **Install Python Extension**: Ensure VS Code has the official Python extension installed.
2.  **Open Notebook**: Open `documentation.ipynb` directly in VS Code.
3.  **Select Kernel**: Choose the "Classification Model" kernel when prompted (usually in the top-right corner).
4.  **Run Cells**: Use `Shift+Enter` to execute cells individually.

-----

## Notebook Configuration

### Kernel Selection

1.  **In Jupyter Notebook/Lab**:
      - Click "Kernel" -\> "Change Kernel" -\> "Classification Model"
2.  **In VS Code**:
      - Click the kernel selector in the top-right of the notebook interface.
      - Choose "Classification Model" from the list.

### Required Data Files

Ensure these files exist in the `output/` directory before running the notebook.

```
output/
├── model.joblib              # Trained model pipeline
├── performance_metrics.json  # Model evaluation results
└── plots/
    └── confusion_matrix.png  # Confusion matrix visualization
```

If these files don't exist, run the main training script first:

```bash
python train_model.py
```

-----

## Notebook Structure & Usage

### Cell-by-Cell Guide

  - **Cell 1: Setup and Imports**
      - Loads required libraries and sets up plotting configurations. No action needed, just run.
  - **Cell 2: Load Model Artifacts**
      - Loads the trained model from `output/model.joblib`.
      - **Expected Output**: "Model loaded successfully"
  - **Cell 3: Load Performance Metrics**
      - Reads evaluation results from `output/performance_metrics.json`.
      - **Expected Output**: A JSON object with accuracy, precision, recall, and F1-score.
  - **Cell 4: Display Performance Summary**
      - Shows a formatted performance table and creates a visual metrics dashboard.
      - **Expected Output**: A professional metrics table.
  - **Cell 5: Show Confusion Matrix**
      - Displays the confusion matrix visualization.
      - **Expected Output**: A heatmap image of the confusion matrix.
  - **Cell 6: Model Pipeline Analysis**
      - Explores the model architecture, showing preprocessing steps and classifier details.
      - **Expected Output**: A breakdown of the pipeline's structure.
  - **Cell 7: Results Interpretation**
      - Provides business context for the metrics and explains performance implications.
      - **Expected Output**: Analysis and recommendations.

-----

## Troubleshooting

### Common Issues and Solutions

  - **Issue: Kernel Not Found**
    ```bash
    # Solution: Re-register the kernel with the --force flag
    python -m ipykernel install --user --name=classification-model --display-name="Classification Model" --force
    ```
  - **Issue: Module Not Found Errors**
    ```bash
    # Solution: Ensure the virtual environment is active and all dependencies are installed
    .\venv\Scripts\activate
    pip install -r requirements.txt
    pip install notebook ipykernel
    ```
  - **Issue: Files Not Found Error**
    ```bash
    # Solution: Run the main training script to generate the output files
    python train_model.py
    ```
  - **Issue: Widgets Not Working**
    ```bash
    # Solution: Install and enable the ipywidgets extension
    pip install ipywidgets
    jupyter nbextension enable --py widgetsnbextension --sys-prefix
    ```

-----

## Quick Start Summary

```bash
# Complete setup in 3 commands
pip install notebook ipykernel
python -m ipykernel install --user --name=classification-model
jupyter notebook documentation.ipynb
```
