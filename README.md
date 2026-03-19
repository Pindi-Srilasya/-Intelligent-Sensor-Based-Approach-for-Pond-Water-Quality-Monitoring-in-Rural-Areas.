# Intelligent Sensor-Based Approach for Pond Water Quality Monitoring in Rural Areas

##  Overview

This project focuses on monitoring and predicting pond water quality in rural areas using sensor data such as Total Dissolved Solids (TDS) and Turbidity. The system analyzes current water conditions and forecasts future trends using machine learning and deep learning techniques.

---

##  Features

* Real-time monitoring of water quality parameters (TDS, Turbidity)
* Water quality classification using Random Forest model
* Time-series forecasting using LSTM model
* Future prediction of water parameters for up to 7 days
* REST API built using Flask

---

##  Tech Stack

* Python
* Flask
* Machine Learning (Random Forest)
* Deep Learning (LSTM - TensorFlow/Keras)
* Pandas, NumPy

---

##  Project Structure

* `app.py` → Flask API for prediction and forecasting
* `train_model.py` → Model training script
* `rf_model.pkl` → Random Forest model
* `lstm_model.h5` → LSTM model
* `scaler.pkl` → Data preprocessing
* `label_encoder.pkl` → Label encoding
* `water_quality_big_dataset.csv` → Dataset
* `requirements.txt` → Dependencies

---

##  API Endpoints

###  Predict Current Water Quality

**POST** `/predict`

Input:

```json
{
  "TDS": 450,
  "Turbidity": 12
}
```

Output:

```json
{
  "prediction": "Safe",
  "confidence": 92.5
}
```

---

###  Predict Future Values

**POST** `/predict_future`

```json
{
  "steps": 7
}
```

---

###  Predict Future Water Quality

**POST** `/predict_future_quality`

---

###  Health Check

**GET** `/healthcheck`

---

##  How to Run

1. Clone the repository:

```
git clone https://github.com/Pindi-Srilasya/-Intelligent-Sensor-Based-Approach-for-Pond-Water-Quality-Monitoring-in-Rural-Areas..git
cd <project-folder>
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Train models (optional):

```
python train_model.py
```

4. Run the server:

```
python app.py
```

---

##  System Working

* Sensor devices collect water parameters such as TDS and Turbidity
* Data is processed and fed into machine learning models
* Random Forest classifies current water quality
* LSTM predicts future values based on time-series data

---

##  Project Type

This is a collaborative project developed as part of a team.

---

##  My Contribution

* Designed and implemented hardware setup using water quality sensors
* Collected real-time data (TDS, Turbidity)
* Assisted in integrating sensor data with the system

---

##  Future Enhancements

* Integration with live IoT devices
* Cloud deployment (AWS)
* Data visualization dashboard
* Mobile application

---

##  Conclusion

This project demonstrates the use of intelligent systems for environmental monitoring, helping improve water safety and decision-making in rural areas.
