# Breast Cancer Classification: SVM vs. Neural Network

## Project Overview
This project compares the performance of Support Vector Machine (SVM) and Neural Network (NN) models in classifying breast cancer tumors as benign or malignant. The dataset used contains 30 features describing tumor characteristics, with a binary target variable (benign: 0, malignant: 1). The evaluation metrics include accuracy, F1-score, confusion matrix, and ROC curve analysis.

## Dataset
- **Source**: The dataset is sourced from `data.csv`, containing 569 samples and 33 columns (including 'id', 'diagnosis', and 30 tumor features).
- **Preprocessing**:
  - Features are normalized using `StandardScaler`.
  - Dataset split: 80% training, 20% testing.

## Methodology
### Models
1. **SVM**:
   - Kernel: Radial Basis Function (RBF)
   - Implementation: `sklearn.svm.SVC`
2. **Neural Network**:
   - Architecture: 1 hidden layer (16 neurons), output layer (sigmoid activation)
   - Activation Functions: ReLU, Sigmoid, Tanh
   - Optimizer: Adam
   - Loss Function: Binary cross-entropy
   - Implementation: `tensorflow.keras`

### Evaluation Metrics
- Accuracy
- F1-score
- Confusion Matrix
- ROC Curve (Area Under Curve - AUC)

## Results
| Model                  | Accuracy | F1-Score | Confusion Matrix    |
|------------------------|----------|----------|---------------------|
| SVM (RBF Kernel)       | 0.9825   | 0.9800   | [[71, 0], [2, 41]]  |
| Neural Network (Sigmoid) | 0.9912 | 0.9882   | [[71, 0], [1, 42]]  |
| Neural Network (Tanh)  | 0.9825   | 0.9767   | [[70, 1], [1, 42]]  |
| Neural Network (ReLU)  | 0.9825   | 0.9762   | [[71, 0], [2, 41]]  |

- **Best Performer**: Neural Network with Sigmoid activation (Accuracy: 0.9912, F1-Score: 0.9882).
- **ROC Analysis**: Sigmoid-based NN achieved the highest AUC, indicating superior classification ability.
- **Observations**: No overfitting observed; ReLU and Tanh models show potential underfitting, which could be addressed by adjusting architecture or adding regularization.

## Requirements
To run the code, install the following dependencies:
- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow`

Install them using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## How to Run
1. **Setup**:
   - Ensure the dataset (`data.csv`) is in the same directory as the notebook (`SVM&NN_ASS.ipynb`).
   - Install the required libraries as mentioned above.
2. **Execution**:
   - Open the Jupyter notebook:
     ```bash
     jupyter notebook SVM&NN_ASS.ipynb
     ```
   - Run all cells to preprocess the data, train the models, and evaluate their performance.
3. **Output**:
   - The notebook will display performance metrics (accuracy, F1-score, confusion matrices) and visualizations (ROC curves).

## Files
- `SVM&NN_ASS.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- `SVM and Neural Network.pdf`: Project report summarizing the methodology, results, and discussion.
- `data.csv`: Dataset (not included; user must provide).

## Future Work
- Experiment with deeper neural network architectures or additional regularization techniques (e.g., Dropout) to address potential underfitting in ReLU and Tanh models.
- Explore other SVM kernels (e.g., linear, polynomial) for comparison.
- Incorporate cross-validation for more robust performance evaluation.

## Contacnt
(LinkedIn)[https://www.linkedin.com/posts/josam-hany-76b449301_machinelearning-artificialintelligence-healthcareinnovation-activity-7327691440501784578-m_YT?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE0hRQMBJwwXzE_2WIlbIlC2-W8nTypJdkU]
