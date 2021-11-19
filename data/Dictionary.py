import typing

import tensorflow as tf
import numpy as np


class Dictionary:
    def __init__(self, data: typing.Union[str, list] = None, lower: bool = False) -> None:
        self.index_to_label = {}
        self.label_to_index = {}
        self.frequencies = {}
        self.lower = lower

        self.special_tokens = []

        if data is not None:
            if type(data) == str:
                self.load_from_file(data)
            else:
                self.add_special_tokens(data)

    @property
    def size(self):
        return len(self.index_to_label)

    def __len__(self):
        return self.size()

    def load_from_file(self, filename: str):
        for line in open(filename, 'r', encoding='utf-8').readlines():
            fields = line.split()

            label = fields[0]
            index = int(fields[1])

            self.add_token(label, index)

    def write_to_file(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as file:
            for i in range(self.size):
                label = self.index_to_label[i]
                file.write(f'{label} {i}\n')

    def add_token(self, label: str, index: int = None) -> int:
        if self.lower:
            label = label.lower()

        if index is not None:
            self.index_to_label[index] = label
            self.label_to_index[label] = index
        else:
            if label in self.label_to_index:
                index = self.label_to_index[label]
            else:
                index = len(self.index_to_label)

                self.index_to_label[index] = label
                self.label_to_index[label] = index

        if index not in self.frequencies:
            self.frequencies[index] = 1
        else:
            self.frequencies[index] += 1

        return index

    def add_special_token(self, label: str, index=None) -> None:
        index = self.add_token(label, index)
        self.special_tokens += [index]

    def add_special_tokens(self, labels: typing.List[str]) -> None:
        for label in labels:
            self.add_special_token(label)

    def lookup(self, key: str, default: int = None) -> typing.Union[int, None]:
        if self.lower:
            key = key.lower()

        try:
            return self.label_to_index[key]
        except KeyError:
            return default

    def reverse_lookup(self, index: int, default: str = None) -> typing.Union[str, None]:
        try:
            return self.index_to_label[index]
        except KeyError:
            return default

    def prune(self, size: int):
        if size >= self.size:
            return self

        sorted_frequencies = np.argsort(
            [self.frequencies[i] for i in range(len(self.frequencies))], axis=0)

        pruned_dictionary = Dictionary(None, self.lower)

        for i in self.special_tokens:
            pruned_dictionary.add_special_token(self.index_to_label[i])
        
        sorted_frequencies = sorted_frequencies[::-1]

        for index in sorted_frequencies[:size]:
            
            pruned_dictionary.add_token(self.index_to_label[index])

        return pruned_dictionary

    def convert_to_index(self, labels: typing.List[str], unkown_word: str, bos_word: str = None, eos_word: str = None) -> typing.List[int]:
        vec = []

        if bos_word is not None:
            vec.append(self.lookup(bos_word))

        unkown_index = self.lookup(unkown_word)
        vec += [self.lookup(label, unkown_index) for label in labels]

        if eos_word is not None:
            vec.append(self.lookup(eos_word))

        return tf.convert_to_tensor(vec)

    def convert_to_labels(self, indices: typing.List[int], stop: int) -> typing.List[str]:
        labels = []

        for index in indices:
            labels.append(self.reverse_lookup(index))
            if index == stop:
                break

        return labels
