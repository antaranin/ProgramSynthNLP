import os
from typing import Collection, Tuple, Union
from enum import Flag, auto

import pandas as pd

from mln_model import Operation, Context, OpObject


class MorphOp(Operation):
    morph_features: Collection[str]

    def __init__(self, contexts: Union[Context, dict],
                 op_and_object: Union[OpObject, dict],
                 morph_features: Collection[str]) -> None:
        super().__init__(contexts, op_and_object)
        self.morph_features = morph_features


class SplitType(Flag):
    NoSplit = 0
    ContextLetters = auto()
    OpAndObject = auto()
    ObjectLetters = auto()
    OpLetters = auto()
    IncludeGrammar = auto()
    GrammarSymbols = auto()


def generate_grammar_train_file(data_file_path: str, output_file_path: str,
                                split_type_flag: SplitType):
    data = _load_csv(data_file_path)
    data = _filter_only_full_words(data)
    relevant_data = data[["Left", "Right", "Operation", "Object", "Grammar"]]
    contexts = relevant_data.apply(_row_to_operation, axis=1).tolist()
    _save_train_file(output_file_path, contexts, split_type_flag)


def _escape_spaces(string: str) -> str:
    return str(string).replace(" ", "_")


def _row_to_operation(row) -> MorphOp:
    return MorphOp(
        Context(_escape_spaces(row.Left), _escape_spaces(row.Right)),
        OpObject(_escape_spaces(row.Operation), _escape_spaces(row.Object)),
        row.Grammar.split(",")
    )


def _filter_only_full_words(data: pd.DataFrame) -> pd.DataFrame:
    full_left = data["Left"].str.contains("<")
    full_right = data["Right"].str.contains(">")
    data = data[full_left][full_right]
    return data


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def _save_train_file(path: str, data: Collection[MorphOp], split_type_flag: SplitType):
    formatted_data = [f"{_format_op(op, split_type_flag)}\n" for op in data]
    with open(path, mode="w+") as file:
        file.writelines(formatted_data)

    print(f"Saved to file: {path}")


def _format_op(op: MorphOp, split_type_flag: SplitType) -> str:
    left, right = _format_context(op.contexts, split_type_flag)
    op_object = _format_op_object(op.op_and_object, split_type_flag)
    if SplitType.IncludeGrammar in split_type_flag:
        morph_features = _format_morph_features(op.morph_features, split_type_flag)
        combined = f"^ {left} {morph_features} {op_object} {right} $"
    else:
        combined = f"^ {left} {op_object} {right} $"

    return combined


def _format_context(context: Context, split_type_flag: SplitType) -> Tuple[str, str]:
    if SplitType.ContextLetters in split_type_flag:
        return _split_chars(context.left), _split_chars(context.right)
    return context.left, context.right


def _split_chars(string: str) -> str:
    return ' '.join(list(string))


def _format_op_object(op_ob: OpObject, split_type_flag: SplitType) -> str:
    op = _split_chars(op_ob.operation) \
        if SplitType.OpLetters in split_type_flag else op_ob.operation
    op_with_parens = op + (
        " ({0}) " if SplitType.OpLetters in split_type_flag else "({0})"
    )

    object = _split_chars(op_ob.operation_object) \
        if SplitType.ObjectLetters in split_type_flag else op_ob.operation_object

    combined = op_with_parens.format(
        f" {object} " if SplitType.OpAndObject in split_type_flag else object
    )
    return combined


def _format_morph_features(morph_features: Collection[str], split_type_flag: SplitType) -> str:
    joiner = " " if SplitType.GrammarSymbols in split_type_flag else ","
    joined_features = joiner.join(morph_features)
    return "[ {0} ]".format(joined_features)


if __name__ == '__main__':
    split_type = SplitType(
        SplitType.ContextLetters | SplitType.IncludeGrammar
    )
    output_dir = f"data/processed/grammar/train_data/{split_type}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    generate_grammar_train_file(
        "data/processed/context_morph_data/asturian.csv",
        f"{output_dir}/asturian.dat",
        split_type
    )
