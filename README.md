# License Plate Recognition
This is implementation of CRNN (CNN+RNN) model proposed in ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"](https://arxiv.org/abs/1507.05717) paper for Chinese License Plate Recognition task.  

Example input: 
![](https://github.com/Bayramova/license_plate_recognition/blob/main/data/demo.jpg)  
Expected output: 闽GGL883

## Data
By default, the training script uses part of the [CCPD2019](https://github.com/detectRecog/CCPD) dataset.

## Usage
1. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Python 3.9.7 and Poetry 1.1.13).
2. Clone this repository to your machine.
3. Download data manually from [here](https://drive.google.com/drive/folders/1d34OtOA2zYlB307zpVjh_1CJnP4PJtii?usp=sharing) and save it locally (default path is *data/CCPD2019-dl1* in repository's root).
4. Install project dependencies (run this and following commands in a terminal, from the root of cloned repository):
```
poetry install --no-dev
```
5. Run train with the following command:
```
poetry run train --input-dir <directory with data> --ckpt-dir <directory to save the model file> --ckpt-name <checkpoint filename> --logs-dir <save directory for TensorBoard Logger> --exp-name <experiment name>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```
poetry run train --help
```
6. Run TensorBoard to visualize losses and metrics:
```
poetry run tensorboard --logdir <save directory for TensorBoard Logger>
```
7. Run recognize to make the prediction for input image with trained model:
```
poetry run recognize --ckpt-path <path to checkpoint> --img-path <path to image>
```
You can download pretrained model from [here](https://drive.google.com/drive/folders/1iYbb1Vvcy7AVs1zQRvh1qKaXUtMiG5rA?usp=sharing) and save it locally (default path is *checpoints/best_checkpoint.ckpt* in repository's root) for usage.

## Development

The code in this repository must be linted with flake8, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer tools.

Format your code with [black](https://github.com/psf/black) and lint it with [flake8](https://github.com/PyCQA/flake8):
```
poetry run black src
```
```
poetry run flake8 src
```
Type annotate your code, run [mypy](https://github.com/python/mypy) to ensure the types are correct:
```
poetry run mypy src
```

Use [pre-commit](https://pre-commit.com/) to run automated checks when you commit changes.
Install the [hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) by running the following command:
```
poetry run pre-commit install
```
Now pre-commit will run automatically on git commit. To trigger hooks manually for all files use the following command:
```
poetry run pre-commit run --all-files
```
