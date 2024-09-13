import glob
import itertools
import math
import os
import random

import librosa
import numpy as np
import soxr
import tensorflow as tf

from utils.wav_io import read_wav


def train_valid_test(random_state):
    random.seed(random_state)
    train = []
    for a in map(chr, range(ord('a'), ord('i') + 1)):
        for n in range(1, 50 + 1):
            train.append(f'{a}{n:02}')
    random.shuffle(train)
    valid_test = []
    for a in map(chr, range(ord('j'), ord('j') + 1)):
        for n in range(1, 53 + 1):
            valid_test.append(f'{a}{n:02}')
    random.shuffle(valid_test)
    valid_size = math.ceil(len(valid_test) / 2)
    return [train, sorted(valid_test[:valid_size]), sorted(valid_test[valid_size:])]


def load_wav(wav_dir, file_names, rate_t, coef, n_fft, hop_length, n_mels):
    wavs = []
    for wav_path in [os.path.join(wav_dir, file_name + '.WAV') for file_name in file_names]:
        rate_i, data, _ = read_wav(wav_path)
        wavs.append(librosa.effects.preemphasis(soxr.resample(data, rate_i, rate_t, 'VHQ'), coef=coef))
    scale = np.amax([np.amax(np.abs(wav)) for wav in wavs])
    specs = [librosa.power_to_db(librosa.feature.melspectrogram(
        y=wav / scale, sr=rate_t, n_fft=n_fft, hop_length=hop_length, center=False, n_mels=n_mels)).T for wav in wavs]
    len_specs = [len(spec) for spec in specs]
    specs = np.concatenate(specs)
    mean, std = specs.mean(), specs.std()
    specs = (specs - mean) / std
    return np.split(specs, np.cumsum(len_specs[:-1])), (mean, std)


def save_wav_npy(temp_dir_path, file_names, wavs):
    for file_name, wav in zip(file_names, wavs):
        np.save(os.path.join(temp_dir_path, f'{file_name}.npy'), wav)


def get_mri_range(mri_dir, file_names, ref_frames):
    mris = [len(glob.glob(os.path.join(mri_dir, file_name, '*.png'))) for file_name in file_names]
    return [[[file_names[j], i, i + ref_frames] for i in range(mri - ref_frames + 1)] for j, mri in enumerate(mris)]


def fix_wavs_mris(wavs, mris):
    wavs_fixed, mris_fixed = [], []
    for wav, mri in zip(wavs, mris):
        wav_fixed, mri_fixed = [], []
        for wav_, mri_ in zip(wav, mri):
            wav_fixed.append(wav_)
            mri_fixed.append(mri_)
        wavs_fixed.append(wav_fixed)
        mris_fixed.append(mri_fixed)
    return list(itertools.chain.from_iterable(wavs_fixed)), list(itertools.chain.from_iterable(mris_fixed))


def write_dataset(wavs, mris, temp_dir_path, filename):
    with tf.io.TFRecordWriter(os.path.join(temp_dir_path, f'{filename}.tfrecord')) as writer:
        for wav, mri in itertools.zip_longest(wavs, mris):
            assert wav[0] == mri[0], 'Alignment error'
            example = tf.train.Example(features=tf.train.Features(feature={
                'n': tf.train.Feature(bytes_list=tf.train.BytesList(value=[wav[0].encode()])),
                'p': tf.train.Feature(int64_list=tf.train.Int64List(value=[mri[1], mri[2], wav[1]]))
            }))
            writer.write(example.SerializeToString())


def save_dataset(wav_dir, mri_dir, temp_dir_path, file_names, random_state,
                 ref_frames, rate_t, coef, n_fft, hop_length, n_mels):
    print('Loading wav files')
    file_names_01 = file_names[0] + file_names[1]
    wavs, scaler = load_wav(wav_dir, file_names_01, rate_t, coef, n_fft, hop_length, n_mels)
    save_wav_npy(temp_dir_path, file_names_01, wavs)
    wavs = [[[file_names_01[j], i] for i in range(len(wav))] for j, wav in enumerate(wavs)]
    wavs, wavs_v = wavs[:len(file_names[0])], wavs[len(file_names[0]):]
    print('Loading MRI files')
    mris = get_mri_range(mri_dir, file_names[0], ref_frames)
    mris_v = get_mri_range(mri_dir, file_names[1], ref_frames)
    print('Saving dataset')
    wavs, mris = fix_wavs_mris(wavs, mris)
    wavs_v, mris_v = fix_wavs_mris(wavs_v, mris_v)
    rng = np.random.default_rng(random_state)
    indices = np.argsort(rng.random(len(wavs)))
    wavs, mris = [wavs[index] for index in indices], [mris[index] for index in indices]
    write_dataset(wavs, mris, temp_dir_path, 'train')
    write_dataset(wavs_v, mris_v, temp_dir_path, 'valid')
    return scaler


def load_f0(f0_dir, file_names):
    nfs, f0s = [], []
    for file_name in file_names:
        base_path = os.path.join(f0_dir, file_name)
        nfs.append(np.load(base_path + '.WAV_nf.npy'))
        f0s.append(np.load(base_path + '.WAV_f0.npy'))
    len_f0s = [len(f0) for f0 in f0s]
    f0s = np.concatenate(f0s)
    mean, std = f0s.mean(), f0s.std()
    f0s = (f0s - mean) / std
    return nfs, np.split(f0s, np.cumsum(len_f0s[:-1])), (mean, std)


def format_ds(ds):
    ret_ds = []
    for fn, nf, f0 in ds:
        for nf_, f0_ in zip(nf, f0):
            ret_ds.append([fn, nf_, f0_])
    return ret_ds


def write_dataset_f0(ds, temp_dir_path, filename):
    with tf.io.TFRecordWriter(os.path.join(temp_dir_path, f'{filename}.tfrecord')) as writer:
        for fn, nf, f0 in ds:
            example = tf.train.Example(features=tf.train.Features(feature={
                'n': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fn.encode()])),
                'p': tf.train.Feature(int64_list=tf.train.Int64List(value=[nf])),
                'f': tf.train.Feature(float_list=tf.train.FloatList(value=[f0]))
            }))
            writer.write(example.SerializeToString())


def save_dataset_f0(f0_dir, temp_dir_path, file_names, random_state):
    print('Loading f0 files')
    file_names_01 = file_names[0] + file_names[1]
    nfs, f0s, scaler = load_f0(f0_dir, file_names_01)
    train = [[fn, nf, f0] for fn, nf, f0 in zip(file_names_01, nfs, f0s)]
    train, valid = train[:len(file_names[0])], train[len(file_names[0]):]
    print('Saving dataset')
    train = format_ds(train)
    valid = format_ds(valid)
    random.seed(random_state)
    random.shuffle(train)
    write_dataset_f0(train, temp_dir_path, 'train')
    write_dataset_f0(valid, temp_dir_path, 'valid')
    return scaler
