import csv
import os
from typing import Collection
import pandas as pd


def load_alphabet(
        alphabet_file_path: str,
        include_end_start_symbols: bool = False
) -> Collection[str]:
    with open(alphabet_file_path, mode='r') as file:
        reader = csv.reader(file, delimiter=";")
        next(reader)
        letters = [
            line[1] if line[1] != " " else "_"
            for line in reader
            if line[1] != ''
        ]
        if include_end_start_symbols:
            letters.append(">")
            letters.append("<")

        return letters


def mkdir_if_not_exists(file_name: str):
    if os.path.isdir(file_name):
        dir = file_name
    else:
        dir = os.path.dirname(file_name)
    if not os.path.exists(dir):
        os.mkdir(dir)


def load_data_frame(data_file_path: str, separator=";") -> pd.DataFrame:
    return pd.read_csv(data_file_path, sep=separator)


def save_lines_to_file(file_path: str, text_lines: Collection[str], add_newlines: bool = False):
    if add_newlines:
        text_lines = [f"{line}\n" for line in text_lines]

    with open(file_path, mode="w+") as file:
        file.writelines(text_lines)
    print(f"Wrote to file {file_path}")
