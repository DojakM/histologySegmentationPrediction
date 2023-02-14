import click
import numpy as np
import os
import sys
import tifffile as tiff
import torch
from rich import traceback
from model.unet_instance import Unet

WD = os.path.dirname(__file__)
@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to data file to predict.')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-suf', '--suffix', type=str, help='Path to write the output to')
@click.option('-o', '--output', default="", required=True, type=str, help='Path to write the output to')
@click.option('-h', '--ome', type=bool, default=False,
              help='human readable output (OME-TIFF format), input and output as image channels')
def main(input: str, suffix: str,  cuda: bool, output: str, ome: bool):
    model = get_pytorch_model(input)
    if cuda:
        model.cuda()
    img = read_data_to_predict("/Users/dominikmolitor/PycharmProjects/seg_predict/consep_1-0000.ome.tif")
    res = predict(img, model)
    write_ome_out(img, res, 1)

def read_data_to_predict(path_to_data_to_predict: str):
    data = tiff.imread(path_to_data_to_predict)
    data = data.transpose(2, 1, 0)
    label = data[:, :, 3]
    image = data[:, :, :3]
    return image


def write_results(predictions: np.ndarray, path_to_write_to) -> None:
    pass


def write_ome_out(image, classification, ids) -> None:
    full_image = np.zeros((256, 256, 4))
    full_image[:, :, 0] = image[:, :, 0]
    full_image[:, :, 1] = image[:, :, 1]
    full_image[:, :, 2] = image[:, :, 2]
    full_image[:, :, 3] = classification[:, :]
    full_image = np.transpose(full_image, (2, 0, 1))
    with tiff.TiffWriter(os.path.join("./OME-TIFFs/", "conic_" + str(ids) + ".ome.tif"), bigtiff=True) as tif_file:
        metadata = {"axes": "CYX",
                    'Channel': {"Name": ["red", "green", "blue", "mask"]}}
        tif_file.write(full_image, photometric="rgb", metadata=metadata)

def get_pytorch_model(path_to_pytorch_model: str):
    model = Unet.load_from_checkpoint(path_to_pytorch_model, num_classes=7, len_test_set=1, strict=False).to('cpu')
    model.eval()
    return model

def predict(data_to_predict, model):
    img = data_to_predict[0, :, :]
    img = torch.from_numpy(np.expand_dims(np.expand_dims(img, 0), 0)).float()
    logits = model(img)[0]
    prediction = torch.argmax(logits.squeeze(), dim=0).cpu().detach().numpy().squeeze()
    return prediction


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover