# Machine Learning Classification — Model Selection and Evaluation

A collection of two supervised machine learning projects developed in Python using **scikit-learn**, focusing on classification algorithms, exploratory data analysis, preprocessing, cross-validation, hyperparameter optimization, and model comparison.

The projects investigate machine learning classification in two different scientific contexts:

1. **Wine Classification** — Multiclass classification using chemical measurements of wine samples.
2. **Higgs Boson Signal Classification** — Binary classification of simulated high-energy particle collision events.

The primary objective is to develop reproducible machine learning workflows, compare different classification approaches, and investigate the influence of model complexity and hyperparameter selection on predictive performance.

---

## Projects Overview

| Property | Wine Classification | HIGGS Classification |
|---|---|---|
| Scientific domain | Chemical data analysis | High-energy particle physics |
| Classification task | Multiclass | Binary |
| Observations | 178 | 20,000 |
| Numerical features | 13 | 28 |
| Target classes | 3 | 2 |
| Initial classifiers | 7 | 7 |
| Cross-validation | 5-Fold Stratified | 5-Fold Stratified |
| Hyperparameter optimization | GridSearchCV | RandomizedSearchCV |
| Model selection metric | Macro F1-score | Macro F1-score |

---

## Project 1 — Wine Classification

### Dataset

The Wine dataset is available directly through scikit-learn using `load_wine()`.

It contains 178 wine samples described by 13 numerical chemical features, including alcohol content, flavonoids, color intensity, and other chemical measurements.

The objective is to classify each wine sample into one of three categories.

### Methodology

The project includes:

- Exploratory Data Analysis (EDA)
- Descriptive statistics and feature distributions
- Feature correlation analysis
- Stratified train-test splitting
- Feature standardization using scikit-learn Pipelines
- Training and comparison of seven classification algorithms
- Accuracy and Macro F1-score evaluation
- Stratified K-Fold Cross-Validation
- Hyperparameter optimization using GridSearchCV
- Final test-set evaluation
- Confusion matrix and classification report
- Error analysis and permutation feature importance

### Classification Algorithms

The following algorithms were investigated:

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Linear Support Vector Machine
5. RBF Support Vector Machine
6. Decision Tree
7. Random Forest

### Model Selection Results

The optimized RBF SVM achieved the highest observed mean cross-validation Macro F1-score.

| Metric | Result |
|---|---|
| Selected model | RBF SVM |
| Best regularization parameter (`C`) | 10 |
| Best RBF parameter (`gamma`) | 0.1 |
| Best CV Macro F1-score | 0.9927 |

These results indicate strong classification performance on the Wine dataset.

The final test evaluation is included in the notebook and should be consulted for the independently measured test-set performance.

---

## Project 2 — Higgs Boson Signal Classification

### Scientific Background

The discovery of the Higgs boson at CERN in 2012 was a major milestone in particle physics.

In high-energy particle collisions, multiple physical processes can produce similar observable signatures, making the identification of interesting signal events a challenging statistical problem.

Machine learning can help distinguish signal-like events from background events by analyzing reconstructed physical observables.

### Dataset

The HIGGS dataset is available through the UCI Machine Learning Repository.

It contains Monte Carlo simulated particle collision events associated with Higgs-related signal and competing background processes.

The original dataset contains approximately 11 million observations. For this project, a reduced subset of **20,000 observations** was used to make model experimentation computationally manageable.

The dataset contains:

- **21 low-level features**, including transverse momenta, pseudorapidities, azimuthal angles, and b-tagging information.
- **7 high-level features**, including reconstructed invariant masses derived from physical combinations of reconstructed objects.

The binary classification target is:

- `0` — Background
- `1` — Higgs Signal

The selected subset contains 9,543 background events and 10,457 signal events.

### Methodology

The project includes:

- Dataset inspection and data quality assessment
- Physical interpretation of the input features
- Class distribution and balance analysis
- Exploratory Data Analysis
- Signal and background feature distribution comparisons
- Correlation matrix and heatmap
- Stratified train-test splitting
- Feature scaling using scikit-learn Pipelines
- Training of seven classification algorithms
- Accuracy and Macro F1-score cross-validation
- Randomized hyperparameter optimization
- Model comparison and final model selection

### Model Comparison

Seven classifiers were initially evaluated using 5-Fold Stratified Cross-Validation.

| Classifier | Mean CV Accuracy |
|---|---:|
| Random Forest | 0.7000 |
| RBF SVM | 0.6716 |
| Linear SVM | 0.6408 |
| Logistic Regression | 0.6403 |
| Decision Tree | 0.6082 |
| Gaussian Naive Bayes | 0.5933 |
| K-Nearest Neighbors | 0.5923 |

Random Forest achieved the highest initial cross-validation accuracy.

### Hyperparameter Optimization

RandomizedSearchCV was applied to Random Forest and RBF SVM using Macro F1-score as the optimization metric.

| Classifier | Best CV Macro F1-score |
|---|---:|
| Random Forest | 0.7081 |
| RBF SVM | 0.6673 |

The optimized Random Forest achieved the highest observed cross-validation Macro F1-score and was selected as the final model.

### Scientific Interpretation

The HIGGS classification problem is substantially more challenging than the Wine classification problem.

The signal and background processes can produce overlapping distributions of reconstructed physical observables.

The results suggest that nonlinear relationships and interactions between features are useful for signal-background discrimination.

Random Forest provides a flexible approach to modeling such relationships through an ensemble of decision trees.

**Project scope:** This notebook concludes after cross-validation, hyperparameter optimization, and model selection. An independent final test-set evaluation is not reported for the HIGGS project.

Consequently, the reported optimized scores should be interpreted as model-selection results rather than unbiased estimates of generalization performance.

---

## Tools and Technologies

The projects were developed using:

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- Google Colab
- Jupyter Notebooks

---

## Repository Structure

```text
machine-learning-classification/
│
├── README.md
├── wine_classification.ipynb
└── higgs_classification.ipynb
```

The notebooks contain the implementation, visualizations, mathematical explanations, and evaluation results.

The filenames shown above are suggested names and should be adjusted to match the files in the repository.

---

## Running the Projects

The notebooks can be opened and executed using Google Colab or a local Jupyter environment.

### Wine Classification

The Wine dataset is included in scikit-learn and can be loaded directly:

```python
from sklearn.datasets import load_wine

wine = load_wine()
```

### HIGGS Classification

The HIGGS notebook uses a previously prepared subset of 20,000 observations.

The dataset is not bundled with scikit-learn. To reproduce the notebook, the prepared CSV file must be available and its loading path updated accordingly.

The original dataset can be obtained from the UCI Machine Learning Repository.

For reproducibility, a fixed `random_state=42` was used in the principal dataset-splitting and model-selection procedures.

---

## Key Learning Outcomes

These projects demonstrate practical experience with:

- Designing end-to-end classification workflows
- Performing statistical exploratory data analysis
- Understanding feature scaling and data leakage
- Building reusable scikit-learn Pipelines
- Comparing linear, nonlinear, probabilistic, and ensemble classifiers
- Applying Stratified K-Fold Cross-Validation
- Evaluating Accuracy and Macro F1-score
- Performing hyperparameter optimization
- Interpreting classification results in scientific contexts
- Organizing reproducible machine learning experiments

---

## Limitations

The Wine dataset contains relatively few observations, making evaluation results sensitive to the selected data split.

The HIGGS project uses only a subset of the original simulated dataset, and its class proportions are not necessarily representative of real experimental signal-to-background rates.

Moreover, cross-validation scores obtained after hyperparameter selection may be optimistic.

The HIGGS results therefore demonstrate comparative model-selection performance rather than a final independent estimate of predictive accuracy.

---

## References

- [scikit-learn — Wine Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- [UCI Machine Learning Repository — HIGGS Dataset](https://archive.ics.uci.edu/dataset/280/higgs)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [CERN — The Higgs Boson](https://home.cern/science/physics/higgs-boson)

---

## Author

**Anastasios Markellos**

Background in Physics, Theoretical and Computational Physics, and scientific programming.

Interested in Machine Learning, numerical methods, computational physics, and scientific applications of data analysis.
