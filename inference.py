import glob
import os

import librosa
import numpy as np
import soxr
import tensorflow as tf
from PIL import Image

from utils.wav_io import read_wav, write_wav


def load_mri(mri_dir, file_name, ref_frames):
    mri_paths = sorted(glob.glob(os.path.join(mri_dir, file_name, '*.png')))
    mris = [np.array(Image.open(mri_path))[..., None].astype(np.float32) / 65535.0 * 255.0 for mri_path in mri_paths]
    return np.stack([mris[i:i + ref_frames] for i in range(len(mris) - ref_frames + 1)])


def save_audio(model, scaler, mri_dir, py_dir_path, file_names,
               ref_frames, rate_t, rate_o, coef, n_fft, hop_length, n_iter, batch_size):
    spec_dir_path = os.path.join(py_dir_path, 'spec')
    wav_dir_path = os.path.join(py_dir_path, 'wav')
    os.makedirs(spec_dir_path, exist_ok=True)
    os.makedirs(wav_dir_path, exist_ok=True)
    wavs = []
    for file_name in file_names:
        mris = load_mri(mri_dir, file_name, ref_frames)
        with tf.device('/CPU:0'):
            ds_test = tf.data.Dataset.from_tensor_slices(mris).batch(batch_size)
        spec = librosa.db_to_power((model.predict(ds_test, verbose=1) * scaler[1] + scaler[0]).T)
        np.save(os.path.join(spec_dir_path, file_name + '_gen.npy'), spec)
        wave = soxr.resample(librosa.effects.deemphasis(librosa.feature.inverse.mel_to_audio(
            spec, sr=rate_t, n_fft=n_fft, hop_length=hop_length, center=False, n_iter=n_iter), coef=coef),
            rate_t, rate_o, 'VHQ')
        wave[:hop_length], wave[-hop_length:] = 0.0, 0.0
        wavs.append(wave)
    scale = np.amax([np.amax(np.abs(wav)) for wav in wavs])
    for i, file_name in enumerate(file_names):
        write_wav(os.path.join(wav_dir_path, file_name + '_gen.wav'), rate_o, wavs[i] / scale, np.dtype('f4'))


def write_filenames(path, file_names):
    with open(path, 'w') as file:
        for file_name in file_names:
            print(file_name, file=file)


def save_pred_spec(model, scaler, ampl, mri_dir, dir_path, file_names, ref_frames, batch_size):
    for file_name in file_names:
        mris = load_mri(mri_dir, file_name, ref_frames)
        with tf.device('/CPU:0'):
            ds_test = tf.data.Dataset.from_tensor_slices(mris).batch(batch_size)
        spec = librosa.db_to_power((model.predict(ds_test, verbose=1) * scaler[1] + scaler[0]).T).clip(0.0)
        if ampl:
            spec = np.sqrt(spec)
        np.save(os.path.join(dir_path, file_name + '.npy'), spec)


def save_wav_gt_spec(ampl, wav_dir, dir_paths, file_names, ref_frames, rate_t, coef, n_fft, hop_length, n_mels):
    wavs = []
    for file_name in file_names:
        rate_i, data, _ = read_wav(os.path.join(wav_dir, file_name + '.WAV'))
        wavs.append(librosa.effects.preemphasis(soxr.resample(data, rate_i, rate_t, 'VHQ'), coef=coef))
    scale = np.amax([np.amax(np.abs(wav)) for wav in wavs])
    for file_name, wav in zip(file_names, wavs):
        spec_len = np.load(os.path.join(dir_paths[0], file_name + '.npy')).shape[1]
        wav = wav[:(spec_len + ref_frames - 1) * hop_length] / scale
        np.save(os.path.join(dir_paths[2], file_name + '.npy'), wav)
        spec = librosa.feature.melspectrogram(y=wav, sr=rate_t, n_fft=n_fft, hop_length=hop_length,
                                              center=False, n_mels=n_mels)
        if ampl:
            spec = np.sqrt(spec)
        np.save(os.path.join(dir_paths[3], file_name + '_gt.npy'), spec)


def save_vocoder_dataset(model, scaler, ampl, wav_dir, mri_dir, py_dir_path, file_names,
                         ref_frames, rate_t, coef, n_fft, hop_length, n_mels, batch_size):
    ds_dir_path = os.path.join(py_dir_path, 'vocoder_ds')
    dir_paths = [os.path.join(ds_dir_path, dir_name) for dir_name in
                 ['ft_dataset', 'test_mel_files', 'wavs', 'ft_dataset_gt']]
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)
    write_filenames(os.path.join(ds_dir_path, 'training.txt'), file_names[0])
    write_filenames(os.path.join(ds_dir_path, 'validation.txt'), file_names[1])
    file_names_01 = file_names[0] + file_names[1]
    save_pred_spec(model, scaler, ampl, mri_dir, dir_paths[0], file_names_01, ref_frames, batch_size)
    save_pred_spec(model, scaler, ampl, mri_dir, dir_paths[1], file_names[2], ref_frames, batch_size)
    save_wav_gt_spec(ampl, wav_dir, dir_paths, file_names_01, ref_frames, rate_t, coef, n_fft, hop_length, n_mels)
