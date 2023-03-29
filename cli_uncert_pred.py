import click
import glob
import numpy as np
import os
import pathlib
import sys
import tifffile as tiff
import torch
from rich import traceback, print
from torchvision.datasets.utils import download_url
from urllib.error import URLError

from model.unet_instance import Unet
from utils import monte_carlo_dropout_proc, weights_init

WD = os.path.dirname(__file__)


@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to data file to predict.')
@click.option('-m', '--model', type=str,
              help='Path to an already trained XGBoost model. If not passed a default model will be loaded.')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-s/-ns', '--sanitize/--no-sanitize', type=bool, default=False,
              help='Whether to remove model after prediction or not.')
@click.option('-suf', '--suffix', default=".", type=str, help='Path to write the output to')
@click.option('-o', '--output', default="", required=True, type=str, help='Path to write the output to')
@click.option('-t', '--iter', default=10, required=True, type=int, help='Number of MC-Dropout interations')
@click.option('-h', '--ome', type=bool, default=False,
              help='human readable output (OME-TIFF format), input and output as image channels')
def main(input: str, suffix: str, model: str, cuda: bool, output: str, sanitize: bool, iter: int, ome: bool):
    """Command-line interface for rts-pred-uncert"""

    print(r"""[bold blue]
        rts-pred-uncert
        """)

    print('[bold blue]Run [green]rts-pred-uncert --help [blue]for an overview of all commands\n')
    if not model:
        model = get_pytorch_model(os.path.join(f'{os.getcwd()}', "models", "model.ckpt"))
    else:
        model = get_pytorch_model(model)
    if cuda:
        model.cuda()

    print('[bold blue] Calculating prediction uncertainty via MC-Dropout')
    print('[bold blue] Parsing data...')
    if os.path.isdir(input):
        input_list = glob.glob(os.path.join(input, "*"))
        for inputs in input_list:
            print(f'[bold yellow] Input: {inputs}')
            file_uncert(inputs, model, inputs.replace(input, output).replace(".tif", suffix), mc_dropout_it=iter,
                        ome_out=ome)
    else:
        file_uncert(input, model, output, ome_out=ome)
    if sanitize:
        os.remove(os.path.join(f'{WD}', "models", "models/model.ckpt"))

def file_uncert(input, model, output, mc_dropout_it=10, ome_out=False):
    input_data = read_input_data(input)
    pred_std = prediction_std(model, input_data, t=mc_dropout_it)

    if ome_out:
        print(f'[bold green] Output: {output}_uncert_.ome.tif')
        write_ome_out(input_data, pred_std, output + "_uncert_")
    else:
        print(f'[bold green] Output: {output}_uncert_.npy')
        write_results(pred_std, output + "_uncert_")

def prediction_std(net, img, t=10):
    net.eval()
    imgs = []
    img = img[:3, :, :].astype(np.float32)
    imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32)
    imgs = torch.from_numpy(imgs)
    pred_std = monte_carlo_dropout_proc(net, imgs, T=t)
    pred_std = pred_std.detach().cpu().numpy().astype(np.float32)

    return pred_std

def read_input_data(path_to_input_data: str):
    """
    Reads the data of an input image
    :param path_to_input_data: Path to the input data file
    """
    return tiff.imread(path_to_input_data)

def write_results(results_array: np.ndarray, path_to_write_to) -> None:
    """
    Writes the output into a file.
    :param results_array: output as a numpy array
    :param path_to_write_to: Output path
    """
    os.makedirs(pathlib.Path(path_to_write_to).parent.absolute(), exist_ok=True)
    np.save(path_to_write_to, results_array)
    pass

def write_ome_out(image, results, out_name) -> None:
    full_image = np.zeros((256, 256, 4))
    full_image[:, :, 0] = image[0, :, :]
    full_image[:, :, 1] = image[1, :, :]
    full_image[:, :, 2] = image[2, :, :]
    full_image[:, :, 3] = results
    full_image = np.transpose(full_image, (2, 0, 1))
    with tiff.TiffWriter(os.path.join(".", out_name), bigtiff=True) as tif_file:
        metadata = {"axes": "CYX",
                    'Channel': {"Name": ["red", "green", "blue", "mask"]}}
        tif_file.write(full_image, photometric="rgb", metadata=metadata)

def get_pytorch_model(path_to_pytorch_model: str):
    if path_to_pytorch_model == "models/model.ckpt":
        if not _check_exists(os.getcwd() + "/models/model.ckpt"):
            download("models/model.ckpt")
    model = Unet(len_test_set=128, hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
    model.apply(weights_init)
    state_dict = torch.load(path_to_pytorch_model, map_location="cpu")
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model.eval()
    return model

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

if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover