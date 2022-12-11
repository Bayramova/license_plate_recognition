from itertools import groupby
from typing import Union

import torch


class Dictionary:
    def __init__(self) -> None:
        self.char2idx: dict[str, int] = {}
        self.idx2char: list[str] = []

    def add_char(self, char: str) -> None:
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self) -> int:
        return len(self.idx2char)


class LabelConverter:
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def encode(self, label: str) -> torch.Tensor:
        """Encodes string into tensor of corresponding indices"""
        return torch.LongTensor([self.dictionary.char2idx[char] for char in label])

    def decode(
        self,
        sequence: Union[list[int], torch.Tensor],
        prediction: bool = True,
        blank: int = 0,
    ) -> str:
        """Decodes tensor of indices into string"""
        # remove repetitions
        if prediction:
            sequence = [int(k) for k, _ in groupby(sequence)]
        return "".join(
            self.dictionary.idx2char[idx] for idx in sequence if idx != blank
        )
