# Advertising Data Analysis

Analysis of advertising spending and sales data using statistical and mathematical methods.

**Course:** Statistics and Mathematics for Data Analysts  
**Academy:** Hamrah Academy

---

## About

This project analyzes advertising data to understand the relationship between spending on different media channels (TV, Radio, Newspaper) and product sales. It uses various statistical and mathematical techniques learned in the course.

## Project Structure

```
advertising-analysis/
├── data/
│   ├── raw/
│   │   └── advertising_Dataset.csv
│   └── processed/
├── notebooks/
│   └── advertising_analysis.ipynb
├── src/
│   └── utils.py
├── docs/
│   └── project_specification.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## What's Inside

### 1. Exploratory Data Analysis (EDA)
- Descriptive statistics (mean, median, variance)
- Data visualizations (histograms, box plots)
- Correlation analysis

### 2. Matrix Analysis & SVD
- Covariance matrix calculation
- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- Dimensionality reduction

### 3. Linear Regression
- Building the model: `Sales = β₀ + β₁×TV + β₂×Radio + β₃×Newspaper`
- Solving using matrix operations (normal equation and SVD)
- Statistical tests (t-tests, p-values)
- Model evaluation (R², RMSE)

### 4. Bayesian Inference
- Defining "high sales" (above median)
- Beta distribution for probability estimation
- Bayesian updating with data
- Credible intervals

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/advertising-analysis.git
cd advertising-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
```bash
jupyter notebook notebooks/advertising_analysis.ipynb
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- jupyter

See `requirements.txt` for specific versions.

## Usage

Open and run `notebooks/advertising_analysis.ipynb` to see the complete analysis. The notebook is organized into clear sections that follow the project requirements.

## Dataset

The dataset contains 200 observations with the following variables:
- **TV**: Advertising budget for TV (in thousands)
- **Radio**: Advertising budget for Radio (in thousands)
- **Newspaper**: Advertising budget for Newspaper (in thousands)
- **Sales**: Product sales (in thousands of units)

## Key Findings

- TV advertising shows the strongest correlation with sales
- The linear regression model achieves good predictive performance
- TV and Radio are significant predictors of sales
- Bayesian analysis provides probabilistic insights into sales outcomes

## Methods Used

- Descriptive Statistics
- Data Visualization
- Matrix Operations
- Eigendecomposition
- Singular Value Decomposition (SVD)
- Linear Regression (from scratch using matrix operations)
- Hypothesis Testing
- Bayesian Inference

## License

MIT License - feel free to use this code for learning purposes.

## Contact

For questions or feedback, please open an issue or contact via email.

---

**Note:** This is an project for the Statistics and Mathematics course at Hamrah Academy.
