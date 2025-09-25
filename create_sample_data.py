import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 1000

# Create features
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
    'years_experience': np.random.randint(0, 40, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Create target variable based on some logic
# Higher income, credit score, and experience increase probability of positive outcome
target_prob = (
    (df['income'] / 100000) * 0.3 +
    (df['credit_score'] / 850) * 0.4 +
    (df['years_experience'] / 40) * 0.2 +
    np.random.random(n_samples) * 0.1  # Add some randomness
)

# Convert to binary target
df['target'] = (target_prob > 0.5).astype(int)

# Save to CSV
df.to_csv('data/source_data.csv', index=False)
print(f"Sample dataset created with {len(df)} rows and {len(df.columns)} columns")
print(f"Target distribution: {df['target'].value_counts().to_dict()}")
print(f"Dataset saved to data/source_data.csv")