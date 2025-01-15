import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Add, Multiply, GlobalAveragePooling2D, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
import os

# Function to load and prepare data
def load_and_prepare_data(file_paths):
    all_samples = []
    all_labels = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)

        rows_per_sample = 8

        for start in range(0, len(df), rows_per_sample):
            end = start + rows_per_sample
            if end <= len(df):
                sample = df.iloc[start:end, :-1].values  # Exclude the last column (label)
                label = df.iloc[start, -1]  # Last column value of the start row (class label)

                all_samples.append(sample)
                all_labels.append(label)

    return np.array(all_samples), np.array(all_labels)

# Function to define the Spatial Attention block
def spatial_attention(x):
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalAveragePooling2D()(x)
    concat = Add()([avg_pool, max_pool])
    dense = Dense(x.shape[-1], activation='sigmoid')(concat)
    attention = Multiply()([x, Reshape((1, 1, x.shape[-1]))(dense)])
    return attention

# Function to define the Multi Convolutional Residual block
def multi_conv_residual_block(x, filters):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    residual = Add()([x, conv2])
    return residual

# Function to define the model
def define_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = multi_conv_residual_block(x, 32)
    x = spatial_attention(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = multi_conv_residual_block(x, 64)
    x = spatial_attention(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to perform federated averaging
def federated_averaging(local_weights_list):
    averaged_weights = []
    num_clients = len(local_weights_list)
    for weights in zip(*local_weights_list):
        averaged_weights.append(np.mean(weights, axis=0))
    return averaged_weights

# Main function for federated learning
def main():
    # Paths to your data files
    file_paths = [f'Client {i}.csv' for i in range(1, 11)]  # 10 clients

    # Load and prepare data for each client
    client_data = [load_and_prepare_data([file_path]) for file_path in file_paths]

    # Normalize and split data into training and test sets for each client
    test_data = []
    scaler = StandardScaler()
    for i in range(len(client_data)):
        samples, labels = client_data[i]
        samples = samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2], 1)
        samples = scaler.fit_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)
        client_data[i] = (X_train, y_train)
        test_data.append((X_test, y_test))

    # Define model
    input_shape = client_data[0][0].shape[1:]  # Shape of a single sample
    num_classes = len(np.unique(client_data[0][1]))  # Number of unique classes
    global_model = define_model(input_shape, num_classes)

    # Federated learning settings
    num_rounds = 20
    num_clients = len(client_data)
    epochs_per_client = [15] * 10  # 15 epochs for each client
    batch_size = 32

    for round_num in range(num_rounds):
        print(f'Round {round_num + 1}/{num_rounds}')
        local_weights = []
        train_loss, train_acc = [], []

        # Train model on each client's data
        for client_num in range(num_clients):
            print(f'  Training on client {client_num + 1}/{num_clients}')
            client_model = define_model(input_shape, num_classes)
            client_model.set_weights(global_model.get_weights())
            X_train, y_train = client_data[client_num]
            history = client_model.fit(X_train, y_train, epochs=epochs_per_client[client_num], batch_size=batch_size, verbose=0)
            local_weights.append(client_model.get_weights())
            train_loss.append(history.history['loss'][-1])
            train_acc.append(history.history['accuracy'][-1])

        # Federated Averaging
        new_weights = federated_averaging(local_weights)
        global_model.set_weights(new_weights)

        # Evaluate the global model on the aggregated test data
        X_test_all = np.concatenate([data[0] for data in test_data])
        y_test_all = np.concatenate([data[1] for data in test_data])
        test_loss, test_acc = global_model.evaluate(X_test_all, y_test_all, verbose=0)

        print(f'  Training loss: {np.mean(train_loss):.4f}, Training accuracy: {np.mean(train_acc):.4f}')
        print(f'  Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

    # Save the final model
    model_path = 'final_model.h5'
    global_model.save(model_path)

    # Calculate and display the size of the final model
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB
    print(f'The size of the final trained model is: {model_size:.2f} MB')

    # Quantize the model
    converter = tf.lite.TFLiteConverter.from_keras_model(global_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the quantized model
    quantized_model_path = 'quantized_model.tflite'
    with open(quantized_model_path, 'wb') as f:
        f.write(tflite_model)

    # Calculate and display the size of the quantized model
    quantized_model_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # Convert bytes to MB
    print(f'The size of the quantized model is: {quantized_model_size:.2f} MB')

if __name__ == '__main__':
    main()
