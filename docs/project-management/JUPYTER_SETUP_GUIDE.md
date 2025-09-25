# Jupyter Notebook Setup Guide - Classification Model Documentation

## ?? Overview

This guide provides step-by-step instructions for setting up and using the `documentation.ipynb` Jupyter notebook to explore and present the classification model results interactively.

## ?? Prerequisites

- Completed Phase 1 implementation (train_model.py executed successfully)
- Python virtual environment activated
- All dependencies installed from requirements.txt

## ?? Quick Setup

### 1. Install Jupyter Notebook

```bash
# Ensure you're in the project directory and virtual environment is active
cd classification-model
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install Jupyter notebook
pip install notebook ipykernel
```

### 2. Register the Virtual Environment as Jupyter Kernel

```bash
# Register the project environment as a Jupyter kernel
python -m ipykernel install --user --name=classification-model --display-name="Classification Model"
```

### 3. Install Additional Notebook Dependencies

```bash
# Install packages for enhanced notebook functionality
pip install jupyterlab-widgets ipywidgets plotly kaleido
```

## ?? Running the Documentation Notebook

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

1. **Install Python Extension**: Ensure VS Code has the Python extension installed
2. **Open Notebook**: Open `documentation.ipynb` directly in VS Code
3. **Select Kernel**: Choose "Classification Model" kernel when prompted
4. **Run Cells**: Use Shift+Enter to execute cells

## ?? Notebook Configuration

### Kernel Selection

1. **In Jupyter Notebook/Lab**:
   - Click "Kernel" ? "Change Kernel" ? "Classification Model"

2. **In VS Code**:
   - Click kernel selector in top-right
   - Choose "Classification Model" from the list

### Required Data Files

Ensure these files exist before running the notebook:

```
output/
??? model.joblib              # Trained model pipeline
??? performance_metrics.json  # Model evaluation results
??? plots/
    ??? confusion_matrix.png  # Confusion matrix visualization
```

If these files don't exist, run the training script first:

```bash
python train_model.py
```

## ?? Notebook Structure & Usage

### Cell-by-Cell Guide

**Cell 1: Setup and Imports**
- Loads required libraries
- Sets up plotting configuration
- No action needed - just run

**Cell 2: Load Model Artifacts**
- Loads the trained model from `output/model.joblib`
- Verifies model is loaded correctly
- **Expected Output**: "? Model loaded successfully"

**Cell 3: Load Performance Metrics**
- Reads evaluation results from `output/performance_metrics.json`
- Displays metrics in readable format
- **Expected Output**: JSON with accuracy, precision, recall, F1-score

**Cell 4: Display Performance Summary**
- Shows formatted performance table
- Creates visual metrics dashboard
- **Expected Output**: Professional metrics table

**Cell 5: Show Confusion Matrix**
- Displays the confusion matrix visualization
- Shows classification results breakdown
- **Expected Output**: Heatmap image

**Cell 6: Model Pipeline Analysis**
- Explores the model architecture
- Shows preprocessing steps and classifier details
- **Expected Output**: Pipeline structure breakdown

**Cell 7: Results Interpretation**
- Provides business context for metrics
- Explains model performance implications
- **Expected Output**: Analysis and recommendations

## ??? Troubleshooting

### Common Issues and Solutions

**Issue: Kernel Not Found**
```bash
# Solution: Re-register the kernel
python -m ipykernel install --user --name=classification-model --display-name="Classification Model" --force
```

**Issue: Module Not Found Errors**
```bash
# Solution: Ensure virtual environment is active and dependencies installed
.\venv\Scripts\activate
pip install -r requirements.txt
pip install notebook ipykernel
```

**Issue: Files Not Found Error**
```bash
# Solution: Run the training script first
python train_model.py
```

**Issue: Plotting Issues**
```bash
# Solution: Install additional plotting dependencies
pip install matplotlib seaborn plotly kaleido
```

**Issue: Widgets Not Working**
```bash
# Solution: Install and enable widgets
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

## ?? Advanced Features

### Interactive Widgets

Add interactive elements to your notebook:

```python
import ipywidgets as widgets
from IPython.display import display

# Example: Interactive threshold selector
threshold_slider = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description='Threshold:',
    readout_format='.2f'
)

display(threshold_slider)
```

### Export Options

**Export to HTML**:
```bash
jupyter nbconvert --to html documentation.ipynb
```

**Export to PDF** (requires additional setup):
```bash
jupyter nbconvert --to pdf documentation.ipynb
```

**Export to Slides**:
```bash
jupyter nbconvert --to slides documentation.ipynb --post serve
```

## ?? Customization Tips

### Enhanced Visualizations

```python
# Custom plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Interactive plots with Plotly
import plotly.express as px
import plotly.graph_objects as go

# Create interactive confusion matrix
fig = px.imshow(cm, text_auto=True, aspect="auto")
fig.update_layout(title="Interactive Confusion Matrix")
fig.show()
```

### Performance Comparison

```python
# Compare multiple models (if available)
models_comparison = {
    'Baseline XGBoost': metrics,
    'Optimized XGBoost': optimized_metrics,  # From Phase 2
    'Random Forest': rf_metrics  # From Phase 2
}

comparison_df = pd.DataFrame(models_comparison).T
comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.show()
```

## ?? Integration with Development Workflow

### Update Notebook After Model Changes

1. **Re-run Training Script**: `python train_model.py`
2. **Restart Kernel**: In Jupyter, click "Kernel" ? "Restart & Run All"
3. **Verify Updates**: Check that new results are loaded

### Version Control Best Practices

```bash
# Before committing notebook changes
jupyter nbconvert --clear-output documentation.ipynb

# Commit the clean notebook
git add documentation.ipynb
git commit -m "docs: Update notebook analysis with new results"
```

## ?? Performance Dashboard Creation

### Real-time Metrics Display

```python
# Create a performance dashboard
def create_performance_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy over time
    axes[0,0].plot(accuracy_history)
    axes[0,0].set_title('Accuracy Trend')
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, ax=axes[0,1])
    axes[0,1].set_title('Confusion Matrix')
    
    # Feature Importance
    feature_importance.plot(kind='barh', ax=axes[1,0])
    axes[1,0].set_title('Top Features')
    
    # ROC Curve
    axes[1,1].plot([0, 1], [0, 1], 'k--')
    axes[1,1].plot(fpr, tpr)
    axes[1,1].set_title('ROC Curve')
    
    plt.tight_layout()
    plt.show()
```

## ?? Production Deployment Integration

### Model Serving Preparation

```python
# Test model serving functionality
def test_model_serving():
    # Load model
    model = joblib.load('output/model.joblib')
    
    # Test prediction
    sample_data = pd.DataFrame({
        'feature1': [1.0],
        'feature2': [2.0],
        # Add your actual features
    })
    
    prediction = model.predict(sample_data)
    probability = model.predict_proba(sample_data)
    
    print(f"Prediction: {prediction[0]}")
    print(f"Confidence: {max(probability[0]):.3f}")
    
    return prediction, probability
```

## ?? Sharing and Collaboration

### NBViewer Integration

1. **Upload to GitHub**: Ensure notebook is in your repository
2. **Share NBViewer Link**: 
   `https://nbviewer.jupyter.org/github/anupamabhay/classification-model/blob/main/documentation.ipynb`

### Binder Integration

Create `requirements.txt` for Binder:
```txt
pandas==2.2.2
scikit-learn==1.5.0
xgboost==2.0.3
matplotlib==3.9.0
seaborn==0.13.2
notebook
ipywidgets
```

Share Binder link:
`https://mybinder.org/v2/gh/anupamabhay/classification-model/main?filepath=documentation.ipynb`

## ? Success Checklist

Before sharing your notebook, ensure:

- [ ] All cells run without errors
- [ ] Outputs are visible and properly formatted
- [ ] Model artifacts are loaded correctly
- [ ] Visualizations display properly
- [ ] Explanatory text is clear and professional
- [ ] Notebook is committed with clean outputs

## ?? Support and Next Steps

### Getting Help

1. **Check Error Messages**: Read the full error traceback
2. **Verify Environment**: Ensure correct kernel is selected
3. **Check File Paths**: Verify all required files exist
4. **Restart Kernel**: Often resolves memory/import issues

### Enhancement Opportunities

- **Interactive Widgets**: Add parameter sliders for model exploration
- **Real-time Updates**: Connect to live data sources
- **Automated Reporting**: Schedule notebook execution
- **Dashboard Integration**: Convert to Streamlit/Dash app

---

## ?? Quick Start Summary

```bash
# Complete setup in 3 commands
pip install notebook ipykernel
python -m ipykernel install --user --name=classification-model
jupyter notebook documentation.ipynb
```

**Ready to explore your model results interactively!** ??

---

*This guide covers everything from basic setup to advanced customization. For additional questions or enhancements, refer to the project documentation or create an issue in the repository.*