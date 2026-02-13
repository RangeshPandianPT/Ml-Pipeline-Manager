# Sample Data Files for ML Pipeline Dashboard

The dashboard accepts the following file formats for upload:

## Supported File Formats

1. **CSV Files** (`.csv`) - Recommended
2. **JSON Files** (`.json`)
3. **Excel Files** (`.xlsx`, `.xls`)

## What Kind of Data?

The pipeline works with **any tabular dataset** that has:
- **Features (columns)**: Numerical or categorical data
- **Target column**: The column you want to predict (for supervised learning)

## Sample Datasets You Can Use

### Option 1: Generate Sample Data (Built-in)

The pipeline has a built-in function to generate sample data. You can use it from Python:

```python
from ingestion import create_sample_dataset

# Generate sample data
df = create_sample_dataset(n_samples=1000, n_features=10)
df.to_csv('sample_data.csv', index=False)
```

### Option 2: Use Your Own Data

Any CSV/Excel/JSON file with:
- Multiple columns (features)
- One target column you want to predict
- Examples:
  - Customer churn data (features: age, tenure, usage → target: churned)
  - House prices (features: bedrooms, sqft, location → target: price)
  - Iris dataset (features: sepal length, width → target: species)

## Example Data Structure

Your CSV should look like this:

```csv
feature1,feature2,feature3,target
1.5,2.3,4.1,0
2.1,3.4,5.2,1
1.8,2.9,4.7,0
...
```

## Quick Test Files

I'll create some sample files for you to test with right now!
