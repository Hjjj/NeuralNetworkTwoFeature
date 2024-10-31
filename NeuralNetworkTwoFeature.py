
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential()

# Input layer with 2 neurons (for x1 and x2)
# Hidden layer with 3 neurons and ReLU activation function
model.add(Dense(3, input_dim=2, activation='relu'))

# Output layer with 1 neuron and sigmoid activation function
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Example data (replace with your actual dataset)
import numpy as np
x_train = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8], [0.9, 1.0]])
y_train = np.array([0, 1, 0, 1])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=1)

# 2 pieced of test data to run through the model
x_new = np.array([[0.2, 0.3], [0.6, 0.7]])

# Make predictions
predictions = model.predict(x_new)

# Print the predictions
print(predictions)