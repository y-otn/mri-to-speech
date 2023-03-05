import glob
import os
import pickle
from datetime import datetime

import tensorflow as tf


def select_model(py_dir_path, models_dirname='models'):
    model_path = None
    paths = glob.glob(os.path.join(py_dir_path, models_dirname, '*.h5'))
    if paths:
        for num, path in enumerate(paths):
            print(str(num + 1) + ':', path)
        number = input('For inference only, select a trained model: ')
        if number:
            model_path = paths[int(number) - 1]
    return model_path


def load_model(model_path, num_items, print_summary=True):
    model = tf.keras.models.load_model(model_path)
    if print_summary:
        model.summary()
    with open(os.path.splitext(model_path)[0] + '.joblib', 'rb') as file:
        others = []
        for _ in range(num_items):
            others.append(pickle.load(file))
    return model, tuple(others)


def save_model(py_dir_path, model, others, models_dirname='models'):
    model_dir_path = os.path.join(py_dir_path, models_dirname)
    os.makedirs(model_dir_path, exist_ok=True)
    file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model.save(os.path.join(model_dir_path, file_name + '.h5'))
    with open(os.path.join(model_dir_path, file_name + '.joblib'), 'wb') as file:
        for other in others:
            pickle.dump(other, file, pickle.HIGHEST_PROTOCOL)
