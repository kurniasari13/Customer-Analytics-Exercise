import numpy as np
import tensorflow as tf 

## Data
npz = np.load('Audiobooks_data_train_Customer_Analytics.npz')
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_validation_Customer_Analytics.npz')
validation_inputs = npz['inputs'].astype(np.float)
validation_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_test_Customer_Analytics.npz')
test_inputs = npz['inputs'].astype(np.float)
test_targets = npz['targets'].astype(np.int)

## Model: outline, optimizers, loss, early stopping and training
#input_size = 10
output_size = 2
hidden_layer_size = 50

model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax')
                            ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100
max_epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(train_inputs,
          train_targets,
          batch_size=batch_size,
          epochs=max_epochs,
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose=2
          )

## Hasil: we have reached a valuation accuracy of around ninety three percent. This means that a ninety three percent of the cases our model has correctly predicted whether a
#  customer will convert again. In other words, if we're given 10 customers and their audiobook activity, we'll be able to accurately identify future customer behavior of nine
#  point three of them. Now, although we said some early stopping mechanisms in place, we still need to test our model because it is possible that we have overfit the validation data.

## Test the model 
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\Test loss: {0:.2f}. Test accuracy: {1:.2f}'.format(test_loss, test_accuracy*100.))

## Obtain the probability for a customer to convert
# hasil predict ada 2 kolom yaitu kolom probability customer will not convert (0) and customer will convert(1)
print(model.predict(test_inputs).round(2))

# hasil predict di slicing dan ambil kolom yang berisi probability customer will convert
print(model.predict(test_inputs)[:, 1].round(0))

# hasil predict menampilkan kolom dengan probability tertinggi, dalam kasus ini yg tertinggi yaitu probability customer beli lagi, method ini bisa dipake jika memiliki binary lebih dari 2
print(np.argmax(model.predict(test_inputs), axis=1))

## Save the model
model.save('Audiobooks_model_Customer_Analytics.h5')
