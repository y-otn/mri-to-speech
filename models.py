import tensorflow as tf
import tensorflow_addons as tfa


def cnn_lstm_model(ref_frames, mri_height, mri_width, mri_channels, n_mels, print_summary=True):
    cnn_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(
        input_shape=(mri_height, mri_width, mri_channels), include_top=False, weights=None, pooling='avg')

    inputs = tf.keras.Input(shape=(ref_frames, mri_height, mri_width, mri_channels))
    x = tf.keras.layers.TimeDistributed(cnn_model)(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(tfa.rnn.PeepholeLSTMCell(640)), merge_mode='sum')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(n_mels)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if print_summary:
        model.summary()
    model.compile(
        optimizer=tfa.optimizers.AdaBelief(),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return model
