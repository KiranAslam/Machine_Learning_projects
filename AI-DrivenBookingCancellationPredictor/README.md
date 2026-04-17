# AI-Driven Booking Cancellation Predictor

An end-to-end Machine Learning solution designed to predict hotel booking cancellations, helping hospitality businesses protect revenue and optimize room occupancy.

## Key Features
- **End-to-End Pipeline:** Automated data cleaning, scaling, and encoding using Scikit-Learn Pipelines.
- **Advanced Encoding:** Implemented Target Encoding for high-cardinality features like 'Country'.
- **Class Imbalance Handling:** Utilized SMOTE (Synthetic Minority Over-sampling Technique) to improve prediction for canceled bookings.
- **Interactive UI:** A dashboard built with Streamlit for real-time risk assessment.

## Project Structure
- `data/`: Contains the hotel booking dataset.
- `models/`: Saved joblib file of the trained model pipeline.
- `src/`: Source code for preprocessing and model training scripts.
- `app.py`: Streamlit application code.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn, Imbalanced-Learn, Category-Encoders, Joblib
- **Frontend:** Streamlit

## Business Impact
By identifying high-risk bookings with ~80%+ confidence, hotel managers can:
1. Implement stricter deposit policies for high-risk segments.
2. Optimize overbooking strategies to minimize revenue leakage.
3. Improve staff allocation based on confirmed booking forecasts.

---
Developed by **Kiran Aslam**