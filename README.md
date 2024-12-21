# covid19-prediction-deep-learning
# Project Overview
This project implements a Deep Neural Network (DNN) model to predict COVID-19 positive cases using survey data collected from various U.S. states. The model analyzes multiple features including state-wide statistics, symptom observations, and mental wellness indicators to predict the proportion of positive cases.
Table of Contents

# Dataset Description
- Model Architecture
- Installation
- Usages
- Results
- Project Structure

# Dataset Description
The dataset contains survey responses from different U.S. states collected over three days. Each row represents:

- State-wise COVID-19 statistics
- Symptom observations
- Behavioral patterns
- Mental wellness indicators
- The target variable 'tested_positive' representing the proportion of confirmed cases on day three

# Model Architecture
# Deep Neural Network Structure:

- Input Layer: Matches feature dimensions
- Hidden Layers:

	- Layer 1: 128 neurons + ReLU activation + Dropout (0.2)
	- Layer 2: 64 neurons + ReLU activation + Dropout (0.2)
	- Layer 3: 32 neurons + ReLU activation + Dropout (0.2)


- Output Layer: 1 neuron (regression output)

# Training Configuration:

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate = 0.001)
- Epochs: 1000
- Dropout Rate: 0.2

# Installation
# Clone the repository
git clone https://github.com/vshaladhav97/covid19-prediction-deep-learning.git

# Navigate to project directory
cd covid19-prediction-deep-learning

# Install required packages
pip install -r requirements.txt

# Usage
# Run the training script
python src/train.py

# For predictions
python src/predict.py

# Results

- Training successfully completed over 1000 epochs
- Achieved validation loss of 1.1615
- Model shows stable learning curve with consistent improvement
- Successfully generates predictions for test data

# Training Progress:
```
Epoch [100/1000], Loss: 15.1926
Epoch [200/1000], Loss: 11.0573
Epoch [300/1000], Loss: 9.6359
...
Epoch [1000/1000], Loss: 8.5280

```

# Project Structure
```
├── data/
│   ├── covid.train.csv
│   └── covid.test.csv
├── src/
│   ├── model.py          # Neural network architecture
│   ├── train.py          # Training script
│   ├── predict.py        # Prediction script
│   └── utils.py          # Helper functions
├── results/
│   └── predictions.csv   # Model predictions
├── README.md
└── requirements.txt
```

# Requirements
- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Author
Vishal Adhav
