import os

import numpy as np
import tensorflow as tf
from PIL import Image

from models import cnn_lstm_model

_mri_dir = None
_temp_dir_path = None


def tf_parse_sub(ns, ps):
    mris, wavs = [], []
    for n, p in zip([n.decode() for n in ns], ps):
        mri = []
        for i in range(p[0], p[1]):
            mri.append(np.array(Image.open(os.path.join(_mri_dir, n, f'{n}_{i:04}.png')))[..., None].astype(np.float32)
                       / 65535.0 * 255.0)
        mris.append(mri)
        wavs.append(np.load(os.path.join(_temp_dir_path, f'{n}.npy'))[p[2]])
    return np.stack(mris), np.stack(wavs)


def tf_parse(example):
    features = tf.io.parse_example(example, features={
        'n': tf.io.FixedLenFeature([], tf.string),
        'p': tf.io.FixedLenFeature([3], tf.int64)
    })
    return tf.numpy_function(tf_parse_sub, [features['n'], features['p']], [tf.float32, tf.float32])


def train_model(mri_dir, temp_dir_path, py_dir_path, ref_frames, mri_size, n_mels, batch_size, max_epochs, patience):
    global _mri_dir, _temp_dir_path
    _mri_dir, _temp_dir_path = mri_dir, temp_dir_path
    model = cnn_lstm_model(ref_frames, *mri_size, n_mels)
    ds_train = tf.data.TFRecordDataset([os.path.join(temp_dir_path, 'train.tfrecord')])
    ds_train = ds_train.shuffle(len(list(ds_train))).batch(batch_size).map(tf_parse).prefetch(tf.data.AUTOTUNE)
    ds_valid = tf.data.TFRecordDataset([os.path.join(temp_dir_path, 'valid.tfrecord')]
                                       ).batch(batch_size).map(tf_parse).prefetch(tf.data.AUTOTUNE)
    cp_path = os.path.join(py_dir_path, 'ckpt', 'weights_epoch_{epoch:02d}.h5')
    model.fit(ds_train, validation_data=ds_valid, epochs=max_epochs,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(cp_path),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=patience // 2, verbose=1),
                         tf.keras.callbacks.EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)])
    return model
