import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM,Dense,Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from Preprocessing import load_data, preprocess_data, prepare_sequences
from tensorflow.keras.callbacks import EarlyStopping

X = np.load('../Data/X_data.npy')
y = np.load('../Data/y_data.npy')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=y)

def build_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128,input_length=300,mask_zero=True),
        Bidirectional(LSTM(64,return_sequences=False)),
        Dropout(0.5), 
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model()
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3,           
    restore_best_weights=True,
    verbose=1
)
history=model.fit(X_train,y_train,epochs=1, validation_data=(X_test,y_test),batch_size=64,callbacks=[early_stop])
y_pred=(model.predict(X_test) > 0.5).astype("int32") 
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy Score: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")  
print(classification_report(y_test, y_pred))
if accuracy >= 0.85:
        model.save('../Models/fake_news_lstm_model.h5')
        print("Model saved Successfully!")