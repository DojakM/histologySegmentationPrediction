import click
import numpy as np
import os
import sys
import tifffile as tiff
import torch
from rich import traceback
from model.unet_instance import Unet
from utils import weights_init
import glob

WD = os.path.dirname(__file__)
@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to data file to predict.')
@click.option('--is-dir', required=False, type=bool, help="Allows iterative description")
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-suf', '--suffix', type=str, help='Path to write the output to')
@click.option('-o', '--output', default="", required=True, type=str, help='Path to write the output to')
@click.option('-h', '--ome', type=bool, default=False,
              help='human readable output (OME-TIFF format), input and output as image channels')
def main(input: str, suffix: str,  cuda: bool, output: str, ome: bool, is_dir: bool):
    model = get_pytorch_model(input)
    if cuda:
        model.cuda()
    if is_dir:
        for file in glob.glob(input + "*.ome.tiff"):
            image, label = read_data_to_predict(file)
            result = predict(image, model)
            if ome:
                write_ome_out(result, label, file.split("/")[-1])
            write_results(result, output)
    else:
        image, label = read_data_to_predict(input)
        result = predict(image, model)
        if ome:
            write_ome_out(image, result, input.split("/")[-1])
        write_results(result, output)


def read_data_to_predict(path_to_data_to_predict: str):
    data = tiff.imread(path_to_data_to_predict)
    label = data[3, :, :]
    image = data[:3, :, :]
    return image, label

def download_model(download: str) -> str:
    return "model.cpkt"

def write_results(predictions: np.ndarray, path_to_write_to) -> None:
    pass

def write_ome_out(image, classification, out_name) -> None:
    full_image = np.zeros((256, 256, 4))
    full_image[:, :, 0] = image[0, :, :]
    full_image[:, :, 1] = image[1, :, :]
    full_image[:, :, 2] = image[2, :, :]
    full_image[:, :, 3] = mask_binning(classification[0,:,:,:])
    full_image = np.transpose(full_image, (2, 0, 1))
    with tiff.TiffWriter(os.path.join(".", out_name), bigtiff=True) as tif_file:
        metadata = {"axes": "CYX",
                    'Channel': {"Name": ["red", "green", "blue", "mask"]}}
        tif_file.write(full_image, photometric="rgb", metadata=metadata)

def mask_binning(classification: torch.Tensor):
    classification = classification.detach().numpy()
    classification = np.argmax(classification, axis=0)
    return classification

def get_pytorch_model(path_to_pytorch_model: str, model_retention: bool = True):
    if len(glob.glob(os.getcwd()+path_to_pytorch_model)) > 0:
        model = Unet(len_test_set=128, hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
        model.apply(weights_init)
        state_dict = torch.load(path_to_pytorch_model, map_location="cpu")
        model.load_state_dict(state_dict["state_dict"], strict=False)
        model.eval()
        return model
    else:
        download_model(path_to_pytorch_model)
        model = Unet(len_test_set=128, hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
        model.apply(weights_init)
        state_dict = torch.load("model.ckpt", map_location="cpu")
        model.load_state_dict(state_dict["state_dict"], strict=False)
        model.eval()
        if not model_retention:
            os.remove("model.ckpt")
        return model

def predict(data_to_predict, model):
    imgs = []
    img = data_to_predict[:, :, :].astype(np.float32)
    imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32)
    img_tensor = torch.from_numpy(imgs)
    prediction = model(img_tensor)
    return prediction


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover