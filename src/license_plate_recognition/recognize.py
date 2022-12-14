from pathlib import Path
from typing import Union

import click
import cv2
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as transforms

import license_plate_recognition.data.alphabet as alphabet
import license_plate_recognition.data.utils as utils
from license_plate_recognition.trainer import OCRModule


@click.command()
@click.option(
    "--ckpt-path",
    type=click.Path(exists=True, dir_okay=False),
    default="checkpoints/best_checkpoint.ckpt",
    show_default=True,
    help="Path to checkpoint.",
)
@click.option(
    "--img-path",
    type=click.Path(exists=True, dir_okay=False),
    default="data/demo.jpg",
    show_default=True,
    help="Path to image.",
)
def recognize(ckpt_path: Path, img_path: Path) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    converter = load_label_converter()

    model = load_model(ckpt_path=ckpt_path, device=device, label_converter=converter)
    model.eval()

    # preprocess image
    img = cv2.imread(img_path)
    img = preprocess_image(img)

    # make predicition
    with torch.no_grad():
        pred = model(img.unsqueeze(0))

    click.echo(f"Predicted label: {converter.decode(pred.argmax(-1))}")


def preprocess_image(img: npt.NDArray[np.int_]) -> torch.Tensor:
    img = cv2.resize(img, (128, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    return transform(img)


def load_label_converter() -> utils.LabelConverter:
    dictionary = utils.Dictionary()
    for char in (
        alphabet.blank + alphabet.provinces + alphabet.alphabets + alphabet.numbers
    ):
        dictionary.add_char(char)

    return utils.LabelConverter(dictionary=dictionary)


def load_model(
    label_converter: utils.LabelConverter,
    ckpt_path: Union[Path, str] = "checkpoints/best_checkpoint.ckpt",
    device: Union[torch.device, str] = "cpu",
) -> OCRModule:
    checkpoint = torch.load(ckpt_path, map_location=device)
    hyper_parameters = checkpoint["hyper_parameters"]
    model = OCRModule(**hyper_parameters, label_converter=label_converter)
    model_weights = checkpoint["state_dict"]
    model.load_state_dict(model_weights)
    return model
