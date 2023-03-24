import pathlib

from torchvision.datasets.utils import download_url
from urllib.error import URLError
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
@click.option('--is-dir', required=False, type=bool, help="Allows iterative predicition")
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-suf', '--suffix', type=str, help='Path to write the output to')
@click.option('-o', '--output', default="", required=True, type=str, help='Path to write the output to')
@click.option('-s/-ns', '--sanitize/--no-sanitize', type=bool, default=False,
              help='Whether to remove model after prediction or not.')
@click.option('-m', '--model', type=str, default="models/model.ckpt", help="Path to model")
@click.option('-h', '--ome', type=bool, default=False,
              help='human readable output (OME-TIFF format), input and output as image channels')
def main(input: str, suffix: str, cuda: bool, output: str, ome: bool, is_dir: bool, sanitize: bool, model: str):
    model = get_pytorch_model(model, sanitize)
    if cuda:
        model.cuda()
    if is_dir:
        for file in glob.glob(input + "*.ome.tiff"):
            image, label = read_data_to_predict(file)
            result = predict(image, model)
            if ome:
                write_ome_out(result, label, suffix + file.split("/")[-1])
            write_results(result, suffix + output)
    else:
        image, label = read_data_to_predict(input)
        result = predict(image, model)
        if ome:
            write_ome_out(image, result, input.split("/")[-1])
        write_results(result.detach().numpy(), output)


def read_data_to_predict(path_to_data_to_predict: str):
    data = tiff.imread(path_to_data_to_predict)
    label = data[3, :, :]
    image = data[:3, :, :]
    return image, label


def _check_exists(filepath) -> bool:
    return os.path.exists(filepath)


def download(filepath) -> None:
    """Download the model if it doesn't exist in processed_folder already."""
    if _check_exists(filepath):
        return
    mirrors = [
        'https://zenodo.org/record/',
    ]
    resources = [
        ("model.ckpt", "7650631/files/model.ckpt", "17511a0af673df264179fb93d73c9dd5"),
    ]
    # download files
    for filename, uniqueID, md5 in resources:
        for mirror in mirrors:
            url = "{}{}".format(mirror, uniqueID)
            try:
                print("Downloading {}".format(url))
                download_url(
                    url, root=str(pathlib.Path(filepath).parent.absolute()),
                    filename=filename,
                    md5=md5
                )
            except URLError as error:
                print(
                    "Failed to download (trying next):\n{}".format(error)
                )
                continue
            finally:
                print()
            break
        else:
            raise RuntimeError("Error downloading {}".format(filename))
    print('Done!')


def write_results(predictions: np.ndarray, path_to_write_to: str) -> None:
    np.save(path_to_write_to, predictions)


def write_ome_out(image, classification, out_name) -> None:
    full_image = np.zeros((256, 256, 4))
    full_image[:, :, 0] = image[0, :, :]
    full_image[:, :, 1] = image[1, :, :]
    full_image[:, :, 2] = image[2, :, :]
    full_image[:, :, 3] = mask_binning(classification[0, :, :, :])
    full_image = np.transpose(full_image, (2, 0, 1))
    with tiff.TiffWriter(os.path.join(".", out_name), bigtiff=True) as tif_file:
        metadata = {"axes": "CYX",
                    'Channel': {"Name": ["red", "green", "blue", "mask"]}}
        tif_file.write(full_image, photometric="rgb", metadata=metadata)


def mask_binning(classification: torch.Tensor):
    classification = classification.detach().numpy()
    classification = np.argmax(classification, axis=0)
    return classification


def get_pytorch_model(path_to_pytorch_model: str, sanitize: bool):
    if path_to_pytorch_model == "models/model.ckpt":
        if not _check_exists(os.getcwd() + "/models/model.ckpt"):
            download("models/model.ckpt")
    path = path_to_pytorch_model
    model = Unet(len_test_set=128, hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
    model.apply(weights_init)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model.eval()
    if sanitize:
        os.remove("models/model.ckpt")
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
    sys.exit(main())
