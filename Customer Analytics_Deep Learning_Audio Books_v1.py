import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

## Extract the data from csv
raw_csv_data = np.loadtxt('D:/DATA ANALYST/belajar_python/CUSTOMER ANALYTICS/Audiobooks_data.csv', delimiter=',')
print(raw_csv_data)

unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:, -1]

## Balance the dataset
num_one_targets = int(np.sum(targets_all))
print(num_one_targets)
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] ==0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

## Standardize the inputs
scaler_deep_learning = StandardScaler()
scaled_inputs = scaler_deep_learning.fit_transform(unscaled_inputs_equal_priors)
print(scaled_inputs)

## Shuffle the data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]
print(shuffled_inputs)

## Split the dataset into train, validation, and test
samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = int(samples_count - train_samples_count - validation_samples_count)

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

## Cek balance the dataset after split
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

# Save the new dataset
np.savez('Audiobooks_data_train_Customer_Analytics', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation_Customer_Analytics', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test_Customer_Analytics', inputs=test_inputs, targets=test_targets)

# Save the scaler
pickle.dump(scaler_deep_learning, open('scaler_deep_learning_customer_analytics_audiobooks.pickle', 'wb'))
