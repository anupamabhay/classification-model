# Classification Model Project

A production-ready baseline classification model built with Python, scikit-learn, and XGBoost.

## Project Overview

This project implements a robust, reproducible classification pipeline following engineering best practices. The workflow is separated into two phases:

1. **Engineering Phase**: Building a solid, automated Python script (`train_model.py`)
2. **Documentation Phase**: Creating interactive documentation (`documentation.ipynb`)

## Project Structure

```
classification-model/
??? data/
?   ??? source_data.csv
??? output/
?   ??? plots/
?   ?   ??? confusion_matrix.png
?   ??? model.joblib
?   ??? performance_metrics.json
??? documents/
?   ??? copilot/
?       ??? project_guide.md
?       ??? plans.md
?       ??? checklist.md
??? train_model.py
??? documentation.ipynb
??? requirements.txt
??? README.md
```

## Quick Start

1. **Set up virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your data:**
   - Add your CSV file as `data/source_data.csv`
   - Ensure it has a 'target' column for classification

4. **Run the training script:**
   ```bash
   python train_model.py
   ```

5. **View results:**
   - Check `output/` folder for model artifacts and metrics
   - Open `documentation.ipynb` for detailed analysis

## Features

- **Automated preprocessing pipeline** with StandardScaler and OneHotEncoder
- **XGBoost classifier** with optimized parameters
- **Comprehensive evaluation** with multiple metrics and visualizations
- **Production-ready code** with logging, type hints, and modular design
- **Reproducible results** with fixed random seeds

## Development

This project follows a script-first development approach, ensuring the core logic is solid and reproducible before focusing on presentation.

---

*Built with ?? for reliable machine learning workflows*