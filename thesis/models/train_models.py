from pathlib import Path

import pandas as pd
import librosa
import librosa.feature
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import keras.layers
import keras.optimizers
import keras.regularizers
import genres
from models.models_consts import model_name, folder_name


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    features = np.mean(mfccs_features.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr = np.mean(zcr.T, axis=0)

    rms = librosa.feature.rms(y=audio)
    rms = np.mean(rms.T, axis=0)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid = np.mean(spectral_centroid.T, axis=0)

    beat = librosa.beat.tempo(y=audio, sr=sample_rate)

    features = np.array(features)
    features = np.append(features, zcr)
    features = np.append(features, rms)
    features = np.append(features, spectral_centroid)
    features = np.append(features, beat)

    return features


def load_data():
    data = []

    path = folder_name.TRAIN_DATA_FOLDER.value

    for root, dirs, files in os.walk(path):

        if root == path:
            continue

        genre = root.split("\\")[1]

        for file in files:

            file = path + "/" + genre + "/" + file

            try:
                song_features = features_extractor(file)
            except Exception:
                pass
            else:
                data.append([song_features, genre])

    return data


def modify_csv_input_data(audio_data_frame):
    X = np.array(audio_data_frame["audio_data"].tolist())

    X_fixed = []

    for i in range(0, X.size):
        split_audio_data = re.split("[\n \]\[]", X[i])
        split_audio_data = list(filter(bool, split_audio_data))
        split_audio_data = [float(numeric_string) for numeric_string in split_audio_data]

        X_fixed.append([split_audio_data])

    X_fixed = np.array(X_fixed)
    y = np.array(audio_data_frame["genre"].tolist())

    y_fixed = []

    for i in y:
        y_fixed.append([genres.Genre[i].value])

    y_fixed = np.array(y_fixed)

    return X_fixed, y_fixed


def show_model_results(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_test_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def train_genre_recognizer(audio_data_frame):
    X, y = modify_csv_input_data(audio_data_frame)

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_test_data(X, y)

    model = keras.Sequential()

    model.add(keras.layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(keras.layers.LSTM(128))

    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(10, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.0002)


    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_validation, y_validation),
                        epochs=250,
                        batch_size=64)

    show_model_results(history)

    model.save(folder_name.MODELS_FOLDER.value + model_name.GENRE_CLASSIFIER_MODEL.value)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)

    return model


def predict_genre(model, X):
    res = model.predict(X)
    predicted_index = np.argmax(res, axis=1)
    genre = genres.Genre(predicted_index)

    return genre


def build_user_recommendation_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(2,)),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model


def train_user_recommendation_model(X, y, model, folder):
    history = model.fit(X,
                        y,
                        epochs=16,
                        batch_size=1)

    model.save(folder + model_name.USER_RECOMMENDATION_MODEL.value)


if __name__ == "__main__":

    try:
        data_frame = pd.read_csv("dataframes/genres.csv")
    except Exception as ex:
        data = load_data()
        data_frame = pd.DataFrame(data, columns=["audio_data", "genre"])
        data_frame.to_csv(Path("dataframes/genres.csv"))
        pass

    data_frame = pd.read_csv("dataframes/genres.csv")
    train_genre_recognizer(data_frame)

