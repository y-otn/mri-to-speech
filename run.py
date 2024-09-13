import os
import tempfile

from dataset import train_valid_test, save_dataset
from inference import save_audio, save_vocoder_dataset
from train import train_model
from utils.models import select_model, load_model, save_model

wav_dir = r'/workspace/mounted/realTimeMRI_ATR503/WAV'
mri_dir = r'/workspace/mounted/realTimeMRI_ATR503/PNG'
temp_dirs_dir = None
py_dir_path = os.path.dirname(os.path.abspath(__file__))

ref_frames = 4
mri_size = (256, 256, 1)

hop_length = 420
rate_t = 20000 / 736 * hop_length
rate_o = 20000
coef = 0.62
n_fft = hop_length * ref_frames
n_mels = 64
n_iter = 512

random_state = 42  # None
batch_size = 32
max_epochs = 64
patience = 8


def main():
    file_names = train_valid_test(random_state)
    model_path = select_model(py_dir_path)
    if model_path:
        model, (scaler,) = load_model(model_path, 1)
    else:
        scaler = save_dataset(wav_dir, mri_dir, temp_dir_path, file_names, random_state,
                              ref_frames, rate_t, coef, n_fft, hop_length, n_mels)
        model = train_model(mri_dir, temp_dir_path, py_dir_path, ref_frames, mri_size, n_mels,
                            batch_size, max_epochs, patience)
        save_model(py_dir_path, model, (scaler,))
    save_audio(model, scaler, mri_dir, py_dir_path, file_names[2],
               ref_frames, rate_t, rate_o, coef, n_fft, hop_length, n_iter, batch_size)
    save_vocoder_dataset(model, scaler, True, wav_dir, mri_dir, py_dir_path, file_names,
                         ref_frames, rate_t, coef, n_fft, hop_length, n_mels, batch_size)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(dir=temp_dirs_dir) as temp_dir_path:
        main()
