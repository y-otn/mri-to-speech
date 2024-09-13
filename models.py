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


def cnn_fc_model(trained_model_path, mri_height, mri_width, mri_channels, print_summary=True):
    cnn_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(
        input_shape=(mri_height, mri_width, mri_channels), include_top=False, weights=None, pooling='avg')
    with tf.device('/CPU:0'):
        trained_model = tf.keras.models.load_model(trained_model_path)
    cnn_model.set_weights(trained_model.layers[1].layer.get_weights())
    cnn_model.trainable = False

    inputs = tf.keras.Input(shape=(mri_height, mri_width, mri_channels))
    x = cnn_model(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(640, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if print_summary:
        model.summary()
    model.compile(
        optimizer=tfa.optimizers.AdaBelief(),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return model
