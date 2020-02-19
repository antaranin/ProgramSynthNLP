from typing import List, Tuple

import pandas as pd
import numpy as np
from itertools import chain
import csv


def _load_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delimiter=";", quotechar='"')


def _save_file(file_path: str, data: pd.DataFrame):
    data.to_csv(file_path, sep=";", quotechar='"', index=False)



def revise_steps(subword_file: str, first_step_file: str, second_step_file: str,
                 revised_first_step_file: str, revised_second_step_file: str):
    subword = _load_file(subword_file)
    first_step = _load_file(first_step_file)
    second_step = _load_file(second_step_file)
    revised_first_step = _revise_first_step(subword, first_step)
    revised_second_step = _revise_second_step(subword, second_step)
    _save_file(revised_first_step_file, revised_first_step)
    _save_file(revised_second_step_file, revised_second_step)


def _revise_first_step(subword_data: pd.DataFrame, first_step_data: pd.DataFrame) -> pd.DataFrame:
    first_step_data: pd.DataFrame = first_step_data.replace(np.nan, "", regex=True)
    first_step_data["Operations"] = (
            first_step_data["Inserts"].astype(str) + "," + first_step_data["Deletes"].astype(
        str)).replace(",$", "", regex=True)

    for index, row in first_step_data.iterrows():
        matching_subword = subword_data[subword_data["Original"].isin(row["Operations"].split(","))]
        for _, subword in matching_subword.iterrows():
            op = subword["Original"]
            split = subword["Split"]
            op_columns = "Inserts" if op.startswith("INS") else "Deletes"
            row["Combined"] = row["Combined"].replace(op, split.replace(",", ""))
            row[op_columns] = row[op_columns].replace(op, split)
    first_step_data.drop("Operations", axis=1, inplace=True)
    return first_step_data


def _revise_second_step(subword_data: pd.DataFrame, second_step_data: pd.DataFrame) -> pd.DataFrame:
    def chainer(s):
        return list(chain.from_iterable(s.str.split(',')))

    second_step_data["Original"] = \
        second_step_data["Operation type"] + "(" + second_step_data["String"] + ")"
    merged = second_step_data.merge(subword_data, on="Original", how="left")
    merged["Split"] = merged["Split"].replace(np.nan, "", regex=True)
    lens = merged['Split'].str.split(',').map(len)
    revised = pd.DataFrame({
        "String": np.repeat(merged["String"], lens),
        "Operation type": np.repeat(merged["Operation type"], lens),
        "Occurrences": np.repeat(merged["Occurrences"], lens),
        "New": chainer(merged["Split"]),
    })
    revised.reset_index(inplace=True, drop=True)
    revised["New"] = _remove_ops(revised["New"])
    ops_to_apply = revised.index[revised["New"] != ""]
    revised.loc[ops_to_apply, "String"] = revised.loc[ops_to_apply, "New"]
    revised["Occurrences"] = revised.groupby(["String", "Operation type"], axis=0)[
        "Occurrences"].transform('sum')
    revised.drop("New", axis=1, inplace=True)
    revised.drop_duplicates(subset=["String", "Operation type"], inplace=True)
    return revised


def _remove_ops(operation_with_string: pd.Series) -> pd.Series:
    no_ins = operation_with_string.replace(r"INS\(", "", regex=True)
    no_del = no_ins.replace(r"DEL\(", "", regex=True)
    return no_del.replace(r"\)", "", regex=True)
