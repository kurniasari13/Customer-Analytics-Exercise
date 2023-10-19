import numpy as np
import tensorflow as tf 
import pickle

## Load the scaler and the model
scaler_deep_learning = pickle.load(open('scaler_deep_learning_customer_analytics_audiobooks.pickle', 'rb'))
model = tf.keras.models.load_model('Audiobooks_model_Customer_Analytics.h5')

## Load the new data
raw_data = np.loadtxt('D:/DATA ANALYST/belajar_python/CUSTOMER ANALYTICS/New_Audiobooks_Data.csv', delimiter=',')
new_data_inputs = raw_data[:, 1:]

## Predict the probability of a customer to convert
new_data_inputs_scaled = scaler_deep_learning.transform(new_data_inputs)
print(model.predict(new_data_inputs_scaled))
print(model.predict(new_data_inputs_scaled)[:, 1].round(2))
print(np.argmax(model.predict(new_data_inputs_scaled), axis=1))
