import os
import tempfile

from dataset import train_valid_test, save_dataset_f0
from inference import test_model_f0
from train import train_model_f0
from utils.models import select_model, load_model, save_model

f0_dir = r'/workspace/mounted/realTimeMRI_ATR503/F0'
mri_dir = r'/workspace/mounted/realTimeMRI_ATR503/PNG'
temp_dirs_dir = None
py_dir_path = os.path.dirname(os.path.abspath(__file__))

mri_height = 256
mri_width = 256
mri_channels = 1

random_state = 42  # None
batch_size = 32 * 4
n_accum_grads = 1
max_epochs = 256
patience = 8


def main():
    file_names = train_valid_test(random_state)
    model_path = select_model(py_dir_path)
    if model_path:
        model, (scaler,) = load_model(model_path, 1)
    else:
        print('Please select a model pre-trained on speech synthesis')
        trained_model_path = select_model(py_dir_path)
        scaler = save_dataset_f0(f0_dir, temp_dir_path, file_names, random_state)
        model = train_model_f0(mri_dir, temp_dir_path, py_dir_path, trained_model_path,
                               mri_height, mri_width, mri_channels, batch_size, max_epochs, patience)
        save_model(py_dir_path, model, (scaler,))
    test_model_f0(model, scaler, f0_dir, mri_dir, py_dir_path, file_names[2],
                  mri_height, mri_width, mri_channels, batch_size)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(dir=temp_dirs_dir) as temp_dir_path:
        main()
