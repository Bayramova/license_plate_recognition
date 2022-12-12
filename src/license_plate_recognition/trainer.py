from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torchmetrics import CharErrorRate, ConfusionMatrix, ExactMatch

import license_plate_recognition.data.utils as utils
from license_plate_recognition.model import CRNN


class OCRModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        dictionary_size: int,
        label_converter: utils.LabelConverter,
    ) -> None:
        super().__init__()

        self.save_hyperparameters("learning_rate", "dictionary_size")  # type: ignore

        self.learning_rate = learning_rate
        self.dictionary_size = dictionary_size
        self.label_converter = label_converter

        self.model = CRNN(dictionary_size=dictionary_size)
        self.criterion = nn.CTCLoss()
        self.test_accuracy = ExactMatch(
            task="multiclass", num_classes=dictionary_size, ignore_index=0
        )
        self.test_char_error_rate = CharErrorRate()
        self.test_confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=dictionary_size, ignore_index=0
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def step(
        self, stage: str, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        images, targets = batch
        logits = self(images)
        log_probs = F.log_softmax(logits, dim=-1)

        (
            T,
            N,
            C,
        ) = (
            log_probs.size()
        )  # T - sequence length, N - batch size, C - number of classes

        input_lengths = torch.LongTensor([T for _ in range(N)])
        target_lengths = torch.LongTensor([targets.size(-1) for _ in range(N)])

        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

        return {"loss": loss, "logits": logits}

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        results = self.step("train", batch, batch_idx)
        self.log(
            "train_loss", results["loss"], on_step=False, on_epoch=True, prog_bar=True
        )
        return results["loss"]

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        results = self.step("val", batch, batch_idx)
        self.log("val_loss", results["loss"], prog_bar=True)
        return results["loss"]

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, Any]:
        results = self.step("test", batch, batch_idx)
        self.log("test_loss", results["loss"], prog_bar=True)

        batch_size = batch[0].size(0)
        targets = batch[1]
        targets_decoded = [
            self.label_converter.decode(targets[i][:], prediction=False)
            for i in range(batch_size)
        ]

        preds = results["logits"].argmax(-1)  # seq_len x batch_size
        preds = preds.permute(1, 0)  # batch_size x seq_len
        preds_decoded = [
            self.label_converter.decode(preds[i][:]) for i in range(batch_size)
        ]  # for CER

        preds_encoded = pad_sequence(
            [self.label_converter.encode(label) for label in preds_decoded],
            batch_first=True,
        ).to(
            self.device  # type: ignore
        )  # for accuracy

        if preds_encoded.size(1) != 7:
            targets = F.pad(targets, (0, preds_encoded.size(1) - 7), "constant", 0)

        # compute accuracy
        self.test_accuracy.update(preds_encoded, targets)
        self.log("test_acc", self.test_accuracy, prog_bar=True)  # type: ignore

        # compute character error rate
        self.test_char_error_rate.update(preds_decoded, targets_decoded)
        self.log("test_char_error_rate", self.test_char_error_rate, prog_bar=True)

        return {"loss": results["loss"], "outputs": preds_encoded, "targets": targets}

    def test_epoch_end(self, outs: list[dict[str, Any]]) -> None:  # type: ignore
        # compute and display confusion matrix
        # some batches have different dimensions, so we have to align them
        # in order to compute confusion matrix
        outputs, targets = [], []
        for i in range(len(outs)):
            batch_size = outs[i]["outputs"].size(0)
            for j in range(batch_size):
                outputs.append(outs[i]["outputs"][j][:])
                targets.append(outs[i]["targets"][j][:])

        outputs_padded = pad_sequence(outputs, batch_first=True).to(
            self.device  # type: ignore
        )
        targets_padded = pad_sequence(targets, batch_first=True).to(
            self.device  # type: ignore
        )

        self.test_confusion_matrix(outputs_padded, targets_padded)  # type: ignore
        computed_conf_mat = (
            self.test_confusion_matrix.compute().cpu().numpy().astype(int)
        )

        # plot confusion matrix
        df = pd.DataFrame(
            computed_conf_mat,
            index=range(self.dictionary_size),
            columns=range(self.dictionary_size),
        )

        plt.figure(figsize=(35, 35))
        fig = sns.heatmap(df, annot=True, cmap="Spectral").get_figure()
        plt.xlabel("Predicted", fontsize=18)
        plt.ylabel("Actual", fontsize=18)
        plt.close(fig)

        self.logger.experiment.add_figure("test_confusion_matrix", fig)  # type: ignore

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
