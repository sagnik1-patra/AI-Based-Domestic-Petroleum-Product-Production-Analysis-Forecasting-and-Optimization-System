
# AI-Based Domestic Petroleum Product Production Analysis, Forecasting, and Optimization System

## Overview

The **AI-Based Domestic Petroleum Product Production Analysis, Forecasting, and Optimization System** is a machine learning research project designed to analyze historical petroleum production data, develop predictive models, evaluate production trends, and investigate optimization techniques for improving model performance.

The project combines traditional Machine Learning (ML), Artificial Neural Networks (ANN), Artificial Immune Systems (AIS), and Particle Swarm Optimization (PSO).

It uses historical domestic petroleum production data to investigate relationships between petroleum product categories, annual production totals, and production growth.

The project also examines the application of bio-inspired and swarm-intelligence optimization algorithms to machine learning hyperparameter selection.

### Main Objectives

- Analyze historical domestic petroleum production.
- Identify patterns and trends in petroleum product output.
- Perform data cleaning and exploratory data analysis.
- Develop regression-based production prediction models.
- Compare conventional machine learning algorithms.
- Implement an Artificial Neural Network.
- Optimize Random Forest hyperparameters using AIS.
- Optimize Random Forest hyperparameters using PSO.
- Evaluate models using MAE, MSE, RMSE, and R².
- Visualize predictions, correlations, residuals, and optimization convergence.
- Generate reusable models and structured evaluation reports.

---

## 1. Problem Statement

Petroleum products play an important role in transportation, manufacturing, energy generation, and industrial development.

Analyzing historical production data can help researchers understand production patterns and investigate methods for estimating future output.

Traditional statistical and machine learning approaches may require careful feature engineering and hyperparameter tuning.

This project investigates whether optimization techniques such as AIS and PSO can improve Random Forest regression performance.

### Research Questions

1. How can machine learning be applied to historical petroleum production data?
2. Which conventional regression model provides the lowest prediction error?
3. Can Artificial Neural Networks model historical production relationships?
4. Can AIS improve Random Forest hyperparameter selection?
5. Can PSO improve Random Forest hyperparameter selection?
6. How do baseline, AIS-optimized, and PSO-optimized models compare?
7. What limitations arise when forecasting with a small historical dataset?

---

## 2. Dataset Description

### Dataset

**DOMESTIC_PRODUCTION_OF_PETROLEUM_PRODUCTS_1.xls**

### Dataset Characteristics

| Property | Description |
|---|---|
| Dataset Type | Historical petroleum production |
| File Format | Microsoft Excel (.xls) |
| Time Period | 2000–01 to 2010–11(P) |
| Historical Observations | 11 annual records |
| Prediction Target | Total petroleum production |
| Task Type | Regression |
| Data Structure | Annual tabular data |
| Forecasting Frequency | Annual |

The dataset contains historical production quantities for different petroleum product categories.

### Important Features

#### Light Distillates

- Liquefied Petroleum Gas (LPG)
- Petrol
- Naphtha

#### Middle Distillates

- Kerosene
- Aviation Turbine Fuel
- High-Speed Diesel Oil
- Light Diesel Oil

#### Heavy Ends

- Fuel Oils
- Lubricants
- Petroleum Coke
- Bitumen

#### Additional Variables

- Others
- Annual Growth over Previous Year
- Year
- Total Production

### Target Variable

The primary target variable is:

`total`

It represents the total domestic production of petroleum products.

### Modeling Consideration

Two different prediction settings are considered:

**Production estimation:** Individual petroleum product quantities may be used to estimate total production. However, if the total is calculated directly from these components, this is an accounting relationship rather than an independent forecasting task.

**Time-based forecasting:** Historical year information is used to estimate production in later periods. Product components and annual growth values from the predicted year are excluded to avoid target leakage.

The PSO forecasting implementation uses the second approach.

---

## 3. Technologies Used

### Programming Language

- Python 3.11

### Data Processing

- NumPy
- Pandas
- xlrd

### Machine Learning

- Scikit-learn
- Random Forest Regression
- Extra Trees Regression
- Ridge Regression

### Deep Learning

- TensorFlow
- Keras
- Artificial Neural Networks

### Optimization Algorithms

- Artificial Immune System (AIS)
- Clonal Selection Algorithm
- Particle Swarm Optimization (PSO)

### Data Visualization

- Matplotlib

### Model Serialization

- Joblib
- HDF5 (.h5)
- JSON
- YAML

### Development Environment

- Jupyter Notebook
- Visual Studio Code
- Python

---

## 4. Project Architecture

```text
                  Historical Petroleum Dataset
                              |
                              v
                       Data Collection
                              |
                              v
                       Data Cleaning
                              |
                              v
                  Exploratory Data Analysis
                              |
                              v
                      Feature Selection
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
       Conventional ML       AIS              PSO
             |                |                |
             v                v                v
      RF / Extra Trees   Clonal Selection   Particle Swarm
         / Ridge              |                |
             |                v                v
             |           Optimized RF      Optimized RF
             |                |                |
             +----------------+----------------+
                              |
                              v
                       Model Evaluation
                              |
                              v
                    Performance Comparison
                              |
                              v
                 Predictions and Visualization
                              |
                              v
                   Saved Models and Reports
```

An additional Artificial Neural Network is trained in the conventional modeling pipeline.

---

## 5. Project Workflow

### Phase 1: Data Collection

The historical dataset is loaded from an Excel workbook using Pandas.

```python
import pandas as pd

df = pd.read_excel(
    "DOMESTIC_PRODUCTION_OF_PETROLEUM_PRODUCTS_1.xls",
    engine="xlrd"
)
```

### Phase 2: Data Preprocessing

Preprocessing operations include:

1. Removing empty rows and columns.
2. Standardizing column names.
3. Converting numeric strings into numeric values.
4. Handling missing values.
5. Extracting financial-year information.
6. Removing invalid target observations.
7. Identifying relevant features.
8. Preparing training and testing datasets.

For financial-year strings:

```python
import re

def extract_year(value):
    match = re.search(
        r"(19\d{2}|20\d{2})",
        str(value)
    )

    return int(match.group(1)) if match else None
```

Example:

```text
2000-01    -> 2000
2001-02    -> 2001
2010-11(P) -> 2010
```

### Phase 3: Exploratory Data Analysis

The system analyzes:

- Historical production trends
- Production distribution
- Relationships between petroleum products
- Correlation coefficients
- Production variability

### Phase 4: Model Development

The project investigates conventional regression models and an Artificial Neural Network.

### Phase 5: AIS Optimization

The Artificial Immune System searches for suitable Random Forest hyperparameters using a clonal-selection-inspired optimization strategy.

### Phase 6: PSO Optimization

Particle Swarm Optimization searches for suitable Random Forest hyperparameters by updating particle positions based on personal and global best solutions.

### Phase 7: Evaluation

Models are evaluated using regression metrics and prediction visualizations.

---

## 6. Machine Learning Models

### 6.1 Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

It can model nonlinear relationships and interactions between input variables.

**Advantages:**

- Handles nonlinear relationships.
- Requires limited feature scaling.
- Reduces variance through ensemble learning.
- Supports hyperparameter optimization.

### 6.2 Extra Trees Regressor

Extra Trees is an ensemble algorithm that introduces additional randomness during decision-tree construction.

It is used as an alternative regression baseline.

### 6.3 Ridge Regression

Ridge Regression is a regularized linear regression algorithm.

It provides a useful comparison against nonlinear ensemble methods.

### 6.4 Artificial Neural Network

The project includes a feedforward neural network.

Example architecture:

```text
Input Layer
     |
Dense Layer (64, ReLU)
     |
Dropout Layer
     |
Dense Layer (32, ReLU)
     |
Dense Layer (16, ReLU)
     |
Output Layer (1, Linear)
```

The network is trained using the Adam optimizer and Mean Squared Error loss.

**Note:** With only 11 annual observations, neural-network performance may be unstable and should be interpreted cautiously.

---

## 7. Artificial Immune System (AIS)

### Overview

The Artificial Immune System is a bio-inspired computational approach based on principles of biological immune systems.

This project uses a clonal-selection-inspired algorithm to optimize Random Forest hyperparameters.

### AIS Optimization Process

1. Initialize a population of candidate hyperparameter configurations.
2. Evaluate candidate fitness using validation RMSE.
3. Select promising antibodies.
4. Clone selected antibodies.
5. Apply hypermutation.
6. Replace weaker candidates.
7. Repeat optimization for multiple generations.
8. Train the final Random Forest using the selected parameters.

### Hyperparameters Optimized

- Number of estimators
- Maximum tree depth
- Minimum samples required to split
- Minimum samples per leaf
- Maximum features considered

### AIS Fitness Function

The objective is to minimize validation Root Mean Squared Error:

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

### AIS Output Files

```text
ais_petroleum_model.pkl
ais_config.yaml
ais_model_metadata.json
ais_accuracy_graph.png
ais_heatmap.png
ais_comparison_graph.png
ais_result.csv
ais_result_graph.png
ais_prediction.csv
ais_prediction_graph.png
ais_convergence_graph.png
ais_residual_graph.png
ais_optimization_history.csv
ais_correlation_matrix.csv
```

### AIS Convergence Visualization

![AIS Optimization Convergence](ais_convergence_graph.png)

The convergence graph illustrates how the best validation RMSE changes across AIS generations.

---

## 8. Particle Swarm Optimization (PSO)

### Overview

Particle Swarm Optimization is a population-based optimization algorithm inspired by collective behavior in natural swarms.

Each particle represents a possible Random Forest hyperparameter configuration.

Particles update their positions based on:

- Their previous velocity
- Their personal best solution
- The global best solution

### Velocity Update Equation

\[
v_i^{t+1}
=
wv_i^t
+
c_1r_1(pbest_i-x_i^t)
+
c_2r_2(gbest-x_i^t)
\]

Where:

- \(v_i\): Particle velocity
- \(x_i\): Particle position
- \(w\): Inertia weight
- \(c_1\): Cognitive coefficient
- \(c_2\): Social coefficient
- \(r_1,r_2\): Random values
- \(pbest_i\): Personal best
- \(gbest\): Global best

### Position Update Equation

\[
x_i^{t+1}=x_i^t+v_i^{t+1}
\]

### PSO Configuration

| Parameter | Value |
|---|---|
| Particles | 8 |
| Iterations | 12 |
| Inertia Weight | 0.7 |
| Cognitive Coefficient | 1.5 |
| Social Coefficient | 1.5 |
| Random State | 42 |
| Fitness Function | Validation RMSE |

These values correspond to the corrected PSO forecasting implementation.

### PSO Workflow

```text
Initialize Particle Population
              |
              v
     Evaluate Particle Fitness
              |
              v
       Update Personal Best
              |
              v
        Update Global Best
              |
              v
       Update Particle Velocity
              |
              v
       Update Particle Position
              |
              v
       Repeat Until Completion
              |
              v
      Select Best Hyperparameters
              |
              v
       Train Optimized RF Model
              |
              v
        Evaluate Test Results
```

### PSO Convergence Visualization

![PSO Optimization Convergence](pso_convergence_graph.png)

This graph shows the best validation RMSE obtained across optimization iterations.

---

## 9. Model Evaluation Metrics

The project evaluates regression performance using the following metrics.

### 9.1 Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE measures the average absolute difference between actual and predicted values.

A lower MAE indicates smaller prediction errors.

### 9.2 Mean Squared Error (MSE)

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

MSE penalizes larger prediction errors more heavily.

### 9.3 Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{MSE}
\]

RMSE measures prediction error in the same units as the target variable.

### 9.4 Coefficient of Determination (R²)

\[
R^2 =
1 -
\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
\]

R² describes how well model predictions explain variation in the observed target values.

An R² value can be negative.

### 9.5 Mean Absolute Percentage Error (MAPE)

\[
MAPE =
\frac{100}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
\]

MAPE expresses prediction errors as percentages.

Zero-valued actual observations require special handling.

---

## 10. Model Performance Visualization

### Main Performance Graph

![Model Performance](accuracy_graph.png)

**Figure 1: Machine Learning Model Performance**

The graph compares regression models using their R² scores.

Although the image is named `accuracy_graph.png`, it represents regression performance rather than classification accuracy.

A higher R² generally indicates better performance on the evaluated observations.

The graph should be interpreted together with MAE and RMSE.

### Baseline Model Comparison

![Model Comparison](comparison_graph.png)

**Figure 2: Comparison of Model Prediction Errors**

This graph compares MAE and RMSE across conventional machine learning models.

### Correlation Heatmap

![Correlation Heatmap](heatmap.png)

**Figure 3: Correlation Analysis**

The heatmap displays pairwise correlations between numeric variables in the petroleum production dataset.

Correlation does not establish causation.

### Actual vs Predicted Production

![Prediction Graph](prediction_graph.png)

**Figure 4: Actual and Predicted Production**

This visualization compares observed production values against model predictions on the evaluation dataset.

### Prediction Results

![Result Graph](result_graph.png)

**Figure 5: Prediction Scatter Plot**

The graph illustrates the relationship between actual and predicted production values.

Points closer to the diagonal reference line indicate smaller prediction errors.

---

## 11. AIS Performance Visualization

### AIS Model Performance

![AIS Performance](ais_accuracy_graph.png)

### AIS Model Comparison

![AIS Comparison](ais_comparison_graph.png)

### AIS Prediction Visualization

![AIS Predictions](ais_prediction_graph.png)

### AIS Prediction Results

![AIS Results](ais_result_graph.png)

### AIS Residual Analysis

![AIS Residuals](ais_residual_graph.png)

These figures support evaluation of the AIS-optimized Random Forest against its baseline.

---

## 12. PSO Performance Visualization

### PSO Model Performance

![PSO Performance](pso_accuracy_graph.png)

### PSO Model Comparison

![PSO Comparison](pso_comparison_graph.png)

### PSO Prediction Visualization

![PSO Predictions](pso_prediction_graph.png)

### PSO Prediction Results

![PSO Results](pso_result_graph.png)

### PSO Residual Analysis

![PSO Residuals](pso_residual_graph.png)

### Historical Production

![Historical Production](pso_historical_graph.png)

### Exploratory Future Production Projection

![Future Production Projection](pso_future_prediction_graph.png)

The future projection is exploratory. A Random Forest using year alone cannot reliably extrapolate trends beyond its training range.

---

## 13. Results and Analysis

The project produces separate evaluation results for conventional machine learning, AIS optimization, and PSO optimization.

### Conventional Machine Learning

Results are saved in:

`result.csv`

### AIS Optimization

Results are saved in:

`ais_result.csv`

### PSO Optimization

Results are saved in:

`pso_result.csv`

### Model Comparison Framework

| Approach | Model | Evaluation |
|---|---|---|
| Conventional ML | Random Forest | MAE, MSE, RMSE, R² |
| Conventional ML | Extra Trees | MAE, MSE, RMSE, R² |
| Conventional ML | Ridge Regression | MAE, MSE, RMSE, R² |
| Deep Learning | ANN | MAE, MSE, RMSE, R² |
| AIS | Optimized Random Forest | MAE, MSE, RMSE, R² |
| PSO | Optimized Random Forest | MAE, MSE, RMSE, R² |

### Interpretation

- Lower MAE indicates smaller average absolute prediction errors.
- Lower RMSE indicates better performance under a squared-error criterion.
- Higher R² indicates better fit relative to a mean-prediction baseline.
- AIS and PSO may improve hyperparameter selection, but improvement is not guaranteed.
- Comparisons are only directly valid when models use identical evaluation observations and compatible feature sets.

**Important:** The corrected PSO forecasting pipeline uses year-only features and a two-year test period. Earlier conventional ML and AIS scripts used different feature configurations and evaluation splits. Their existing CSV results should not be treated as a controlled head-to-head comparison without retraining under a shared experimental protocol.

---

## 14. Project Directory Structure

```text
AI-Based Domestic Petroleum Product Production Analysis/
│
├── DOMESTIC_PRODUCTION_OF_PETROLEUM_PRODUCTS_1.xls
│
├── README.md
│
├── petroleum_model.h5
├── petroleum_model.pkl
├── neural_preprocessing.pkl
├── config.yaml
├── model_metadata.json
│
├── cleaned_petroleum_data.csv
├── production_statistics.csv
├── model_comparison.csv
├── training_history.csv
│
├── accuracy_graph.png
├── heatmap.png
├── comparison_graph.png
├── result.csv
├── result_graph.png
├── prediction.csv
├── prediction_graph.png
├── residual_graph.png
├── training_graph.png
│
├── ais_petroleum_model.pkl
├── ais_config.yaml
├── ais_model_metadata.json
├── ais_accuracy_graph.png
├── ais_heatmap.png
├── ais_comparison_graph.png
├── ais_result.csv
├── ais_result_graph.png
├── ais_prediction.csv
├── ais_prediction_graph.png
├── ais_convergence_graph.png
├── ais_residual_graph.png
├── ais_optimization_history.csv
├── ais_correlation_matrix.csv
│
├── pso_petroleum_model.pkl
├── pso_future_model.pkl
├── pso_config.yaml
├── pso_model_metadata.json
├── pso_cleaned_data.csv
├── pso_accuracy_graph.png
├── pso_heatmap.png
├── pso_comparison_graph.png
├── pso_result.csv
├── pso_result_graph.png
├── pso_prediction.csv
├── pso_prediction_graph.png
├── pso_convergence_graph.png
├── pso_residual_graph.png
├── pso_historical_graph.png
├── pso_future_prediction.csv
├── pso_future_prediction_graph.png
├── pso_optimization_history.csv
└── pso_correlation_matrix.csv
```

The directory structure lists the intended outputs from the different project scripts. Files will be present only after their corresponding scripts have executed successfully.

---

## 15. Installation

### Prerequisites

- Python 3.10 or later
- Jupyter Notebook or VS Code
- pip package manager

### Install Dependencies

```bash
pip install numpy pandas xlrd scikit-learn matplotlib tensorflow h5py joblib pyyaml
```

### Open the Project Directory

```bash
cd "C:\Users\sagni\Downloads\AI-Based Domestic Petroleum Product Production Analysis, Forecasting, and Optimization System"
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## 16. How to Run the Project

### Step 1: Prepare the Dataset

Place the original Excel dataset in the project directory.

```text
DOMESTIC_PRODUCTION_OF_PETROLEUM_PRODUCTS_1.xls
```

### Step 2: Execute the Conventional ML Pipeline

Run the initial machine learning training code.

It generates:

```text
petroleum_model.pkl
petroleum_model.h5
config.yaml
model_metadata.json
cleaned_petroleum_data.csv
```

### Step 3: Generate Conventional Model Visualizations

Run the visualization code to generate:

```text
accuracy_graph.png
heatmap.png
comparison_graph.png
result_graph.png
prediction_graph.png
result.csv
prediction.csv
```

### Step 4: Execute AIS Optimization

Run the AIS optimization code.

It produces files with the prefix:

`ais_`

### Step 5: Execute PSO Optimization

Run the corrected PSO optimization code.

It reads the original Excel workbook and produces files with the prefix:

`pso_`

### Step 6: Review Evaluation Results

Open the generated CSV files to examine model performance.

```text
result.csv
ais_result.csv
pso_result.csv
```

### Step 7: Review Visualizations

Open the generated PNG files to examine model performance, prediction errors, and optimization behavior.

---

## 17. Model Loading and Prediction

### Load Conventional Machine Learning Model

```python
import joblib

model = joblib.load(
    "petroleum_model.pkl"
)
```

The saved pipeline expects the same input feature names and preprocessing structure used during training.

### Load AIS Model

```python
ais_model = joblib.load(
    "ais_petroleum_model.pkl"
)
```

### Load PSO Model

```python
pso_model = joblib.load(
    "pso_petroleum_model.pkl"
)
```

### Example PSO Prediction

```python
import pandas as pd

future_data = pd.DataFrame({
    "year": [2011, 2012, 2013]
})

predictions = pso_model.predict(
    future_data
)

print(predictions)
```

These predictions are extrapolations beyond the training years and should not be interpreted as validated forecasts.

The separate `pso_future_model.pkl` is fitted on the full historical dataset for exploratory projection.

---

## 18. Applications

Potential applications of the project include:

### Petroleum Production Analysis

Studying historical production quantities and changes over time.

### Energy Sector Research

Investigating the use of machine learning techniques in energy-related datasets.

### Predictive Modeling

Comparing regression models for production estimation.

### Optimization Research

Evaluating bio-inspired algorithms for machine learning hyperparameter optimization.

### Academic Research

Supporting experiments in:

- Artificial Intelligence
- Machine Learning
- Swarm Intelligence
- Computational Optimization
- Energy Informatics
- Time-Series Analysis

---

## 19. Advantages

- Combines conventional ML with optimization techniques.
- Supports multiple regression algorithms.
- Includes neural-network experimentation.
- Implements AIS and PSO optimization.
- Generates reusable trained models.
- Produces CSV-based evaluation reports.
- Creates publication-oriented visualizations.
- Supports exploratory historical production analysis.
- Uses reproducible random seeds.
- Separates model training and testing.

---

## 20. Limitations

### Small Dataset

The historical dataset contains only 11 annual observations.

This substantially limits the reliability of complex machine learning models and hyperparameter optimization.

### Limited Forecasting Features

The corrected PSO forecasting model uses year as its input feature.

It does not incorporate economic, policy, refinery-capacity, demand, or energy-market variables.

### Random Forest Extrapolation

Random Forest models do not naturally extrapolate trends beyond the observed training range.

### Unstable Evaluation Metrics

A test set containing only two observations produces highly variable R² estimates.

### Potential Target Leakage

Predicting total production from its constituent petroleum-product quantities may represent a direct accounting relationship rather than a genuine forecasting problem.

### Experimental Comparability

Existing conventional ML, AIS, and PSO implementations do not all use identical features and evaluation protocols.

Their metrics require alignment before drawing conclusions about algorithm superiority.

---

## 21. Future Enhancements

Future development may include:

1. Collecting additional years of petroleum production data.
2. Incorporating refinery capacity and demand indicators.
3. Adding macroeconomic variables.
4. Implementing ARIMA and SARIMA forecasting.
5. Comparing gradient boosting and XGBoost models.
6. Developing LSTM and GRU models with sufficient data.
7. Applying time-series cross-validation.
8. Implementing consistent AIS and PSO evaluation protocols.
9. Conducting repeated experiments with multiple random seeds.
10. Adding uncertainty intervals to forecasts.
11. Developing a Streamlit dashboard.
12. Integrating real-time data sources where available.
13. Comparing models against naïve forecasting baselines.
14. Investigating multi-objective production optimization.

---

## 22. Research Contributions

The project provides an experimental framework combining conventional regression models with bio-inspired optimization.

Its main methodological contributions include:

- Implementing conventional machine learning baselines.
- Investigating neural-network regression.
- Applying clonal-selection-inspired AIS to hyperparameter search.
- Applying PSO to Random Forest hyperparameter optimization.
- Producing standardized evaluation visualizations.
- Examining the limitations of forecasting with short historical time series.
- Identifying target-leakage risks in aggregate production prediction.

These are implementation-level contributions. Claims of superior predictive accuracy require empirical evidence from controlled experiments.

---

## 23. Conclusion

The **AI-Based Domestic Petroleum Product Production Analysis, Forecasting, and Optimization System** demonstrates the application of machine learning and computational optimization to historical petroleum production data.

The project integrates conventional regression models, Artificial Neural Networks, Artificial Immune Systems, and Particle Swarm Optimization.

AIS and PSO are used to search for Random Forest hyperparameters, while regression metrics and visualizations support model evaluation.

The project also highlights important methodological considerations, including target leakage, temporal validation, limited historical observations, and the inability of tree-based models to reliably extrapolate long-term trends.

The system serves as a foundation for further research into petroleum production forecasting, machine learning optimization, and energy-sector data analytics.

---

## 24. References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324

2. Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely Randomized Trees. *Machine Learning*, 63, 3–42. https://doi.org/10.1007/s10994-006-6226-1

3. Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization. *Proceedings of ICNN'95 - International Conference on Neural Networks*. https://doi.org/10.1109/ICNN.1995.488968

4. de Castro, L. N., & Von Zuben, F. J. (2002). Learning and Optimization Using the Clonal Selection Principle. *IEEE Transactions on Evolutionary Computation*, 6(3), 239–251. https://doi.org/10.1109/TEVC.2002.1011539

5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

---

## 25. License

This project is intended for academic, educational, and research purposes.

A specific software license has not yet been assigned. Dataset reuse and redistribution should follow the terms of the original data provider.

---

## 26. Acknowledgments

This project uses open-source Python libraries for machine learning, optimization, data analysis, and visualization.

The implementations are intended to support reproducible experimentation and further research in energy-sector analytics.

---

**Project:** AI-Based Domestic Petroleum Product Production Analysis, Forecasting, and Optimization System

**Techniques:** Machine Learning | Deep Learning | AIS | PSO

**Domain:** Artificial Intelligence | Energy Informatics | Production Forecasting
