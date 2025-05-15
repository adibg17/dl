import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Set seed for reproducibility
np.random.seed(2017)

# Load dataset
data = pd.read_csv('Data/winequality-red.csv', sep=';')

# Split features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2017)

# Define a simple neural network model
def create_model():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))  # Regression output
    return model

# Prepare directory to save models
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Define optimizers to test
optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.001)
}

# Train and evaluate each optimizer with callbacks
for name, optimizer in optimizers.items():
    print(f"\nTraining with optimizer: {name}")
    model = create_model()
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Callbacks: Early stopping and model checkpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=f'{checkpoint_dir}/best_model_{name}.h5', 
                        monitor='val_loss', save_best_only=True, verbose=0)
    ]

    model.fit(X_train, y_train, epochs=100, batch_size=32, 
              validation_split=0.2, callbacks=callbacks, verbose=1)

    # Evaluate on test data
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE with {name}: {mae:.4f}")
