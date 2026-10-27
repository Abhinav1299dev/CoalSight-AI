# Coal Production Prediction Using Machine Learning

> An end-to-end machine learning application that predicts coal production from mining and geographical attributes using Random Forest Regression, with an interactive Streamlit web interface for real-time predictions and visualization.

---

## 📌 Overview

**Coal Production Prediction** is a machine learning project developed to estimate coal production using real-world mining data.

The project uses **Random Forest Regression** to learn patterns between mining characteristics and production levels. The trained model is integrated with a **Streamlit web application**, allowing users to provide mining-related inputs and obtain production predictions along with graphical and geographical insights.

The project covers the complete machine learning workflow:

**Data Collection → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction → Web Application**

---

## 🎯 Problem Statement

Coal production varies based on multiple factors such as coal type, mine ownership, geographical location, and other mining characteristics.

The objective of this project is to develop a machine learning-based system that uses historical mining data to estimate coal production and presents the results through an interactive and user-friendly web application.

---

## 🚀 Objectives

* Develop a machine learning model for coal production prediction.
* Process and prepare real-world mining data for machine learning.
* Handle categorical and numerical features.
* Train a Random Forest Regression model.
* Evaluate the trained model using regression metrics.
* Build an interactive Streamlit application.
* Provide production-level classification.
* Present prediction results through visual and geographical insights.

---

## 📸 Dashboard Preview

<p align="center">
  <img
    src="screenshots/coal-analytics-dashboard.png"
    alt="Coal Analytics Dashboard"
    width="100%"
  />
</p>

## ✨ Key Features

### Machine Learning

* Data preprocessing and cleaning
* Feature engineering
* Categorical feature encoding
* Train-test splitting
* Random Forest Regression
* Model evaluation
* Prediction pipeline

### Interactive Application

* User-friendly input interface
* Real-time production prediction
* Production category classification
* Prediction vs. average production comparison
* Graphical data visualization
* Location visualization using geographical coordinates

---

## 📊 Dataset

The project uses real-world mining data containing information about coal production, mine characteristics, ownership, coal type, and geographical location.

### Dataset Features

| Feature                 | Description                                        |
| ----------------------- | -------------------------------------------------- |
| State/UT Name           | State or Union Territory where the mine is located |
| Mine Name               | Name of the mining operation                       |
| Coal/Lignite Production | Recorded production value                          |
| Owner                   | Mine owner or operating organization               |
| Coal/Lignite            | Type of mineral                                    |
| Govt Owned/Private      | Ownership category                                 |
| TypeofMine              | Type/category of mine                              |
| Latitude                | Geographical latitude                              |
| Longitude               | Geographical longitude                             |

The dataset contains both **categorical and numerical attributes**, which are processed before model training.

---

## 🤖 Machine Learning Model

### Random Forest Regression

The project uses **Random Forest Regression** as the primary machine learning algorithm.

Random Forest is an ensemble learning technique that combines multiple decision trees to produce a final prediction. It can capture complex and non-linear relationships between input features and the target production value.

### Target Variable

The model predicts:

**Coal/Lignite Production**

The application displays the predicted production in **million tonnes**.

---

## 🔄 Machine Learning Workflow

```text
                  Raw Mining Data
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
                Categorical Encoding
                         │
                         ▼
                  Train-Test Split
                         │
                         ▼
              Random Forest Regression
                         │
                         ▼
                 Model Evaluation
                         │
                         ▼
                  Trained Model
                         │
                         ▼
                 Prediction Pipeline
                         │
                         ▼
              Streamlit Web Application
                         │
                  ┌──────┴──────┐
                  ▼             ▼
              Prediction   Visualization
```

---

## 📈 Application Output

The application provides the following results based on the entered parameters:

### Estimated Production

Displays the predicted coal production in **million tonnes**.

### Production Category

The predicted production is classified into:

* **Low**
* **Medium**
* **High**

### Production Comparison

A graphical comparison is provided between the predicted production and the average production.

### Location Visualization

The selected location is visualized using its **latitude and longitude** coordinates.

---

## 🖥️ Streamlit Application

The machine learning model is integrated into a **Streamlit web application**.

Users can:

1. Enter the required mining parameters.
2. Submit the input data.
3. Generate a coal production prediction.
4. View the production category.
5. Analyze the prediction through visualizations.
6. View the selected mining location on a map.

---

## 🛠️ Technology Stack

| Category             | Technologies             |
| -------------------- | ------------------------ |
| Programming Language | Python                   |
| Data Processing      | Pandas, NumPy            |
| Machine Learning     | Scikit-learn             |
| ML Algorithm         | Random Forest Regression |
| Visualization        | Matplotlib, Seaborn      |
| Web Application      | Streamlit                |
| Version Control      | Git, GitHub              |

---

## 📂 Project Structure

```text
coal_project/
│
├── data/
│
├── model/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites

Make sure the following are installed:

* Python 3.x
* Git
* pip

### 1. Clone the Repository

```bash
git clone <repository-url>
cd coal_project
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the Model

Run the training script:

```bash
python src/train.py
```

This prepares the data and trains the Random Forest Regression model.

### Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your default web browser.

---

## 🧪 Model Evaluation

The model can be evaluated using standard regression metrics, including:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

These metrics help measure the difference between actual and predicted production values.

---

## 📊 Visualizations

The application provides visual insights to make the prediction results easier to interpret.

Current visualization capabilities include:

* Predicted production comparison
* Average production comparison
* Production-related graphs
* Geographical location visualization

---

## 💡 Skills Demonstrated

This project demonstrates practical implementation of:

* Python programming
* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Categorical data encoding
* Regression modeling
* Model evaluation
* Data visualization
* Streamlit application development
* Git and GitHub

---

## 🔮 Future Improvements

* Experiment with additional machine learning algorithms.
* Perform automated hyperparameter optimization.
* Add historical production trend analysis.
* Integrate regularly updated mining datasets.
* Enhance geographical analysis.
* Add model performance monitoring.
* Improve dashboard design and visualization.
* Deploy the application on a cloud platform such as Streamlit Community Cloud or AWS.

---

## 🌐 Deployment

The Streamlit application can be deployed to cloud platforms such as:

* **Streamlit Community Cloud**
* **AWS**
* Other cloud hosting platforms


---

## 👨‍💻 Author

**Abhinav Yadav**

Computer Science Engineering Student
MIT 
CSE

Work in progress


---

## ⭐ Support

If you find this project useful, consider giving the repository a **star**.

---

## 📄 License

This project is developed for **educational and academic purposes**.
