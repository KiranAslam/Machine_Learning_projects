import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from preprocessing import get_pipeline

def train_model():
    df = pd.read_csv('../Data/hotel_bookings.csv')

    df.drop(columns=['reservation_status', 'reservation_status_date', 'company', 'agent'], inplace=True)
    df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
    df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    X = df.drop(columns=['is_canceled'], axis=1)
    Y = df['is_canceled']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model_pipeline = get_pipeline()
    model_pipeline.fit(X_train, Y_train)
    joblib.dump(model_pipeline, '../models/booking_cancellation_predictor.pkl')
    
    print('Training completed and model saved successfully.')

if __name__ == "__main__":
    train_model()
