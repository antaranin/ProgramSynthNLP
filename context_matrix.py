from itertools import chain
from typing import Union, Iterable

import numpy as np
import pandas as pd


def get_chars_before_str(combined: str, sub_string: str, count: int, exact_count=True) -> Union[
    str, None]:
    op_start = combined.index(sub_string)
    cleaned_combined, cleaned_op_start = escape_ops(combined, op_start)
    prefix = cleaned_combined[:cleaned_op_start]
    is_shorter_prefix = len(prefix) < count
    if exact_count and is_shorter_prefix:
        return None
    adjusted_prefix = prefix if is_shorter_prefix else prefix[-count:]
    return f"L({adjusted_prefix})"


def get_chars_after_str(combined: str, sub_string: str, count: int, exact_count=True) -> Union[
    str, None]:
    op_end = combined.index(sub_string) + len(sub_string)
    cleaned_combined, cleaned_op_end = escape_ops(combined, op_end)
    postfix = cleaned_combined[cleaned_op_end:]
    is_shorter_postfix = len(postfix) < count
    if is_shorter_postfix and exact_count:
        return None
    adjusted_postfix = postfix if is_shorter_postfix else postfix[:count]
    return f"R({adjusted_postfix})"


def escape_ops(combined: str, index_to_keep_track_of: int):
    new_index = index_to_keep_track_of
    new_combined = ""
    clear_op = ""
    for index, char in enumerate(combined):
        if char == "I" or char == "D" or char == ")":
            clear_op = char
        elif clear_op == "" or clear_op == ")":
            new_combined = new_combined + char
            clear_op = ""
        elif clear_op == "D" and char == "(":
            clear_op = ")"

        if clear_op != "" and index < index_to_keep_track_of:
            new_index -= 1
    return new_combined, new_index


def load_csv_to_pandas(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name, delimiter=";", quotechar='"')


def group_character_operations(first_step: pd.DataFrame, second_step: pd.DataFrame) -> pd.DataFrame:
    all_groups = []
    for index, row in second_step.iterrows():
        op = row['Operation type']
        op_type = "Inserts" if op == "INS" else "Deletes"
        string = row["String"]
        full_op = f"{op}({string})"
        matching_rows = first_step.loc[first_step[op_type] == full_op]
        filtered_rows = matching_rows[["Combined"]]
        filtered_rows["Object"] = full_op
        all_groups.append(filtered_rows)
    return pd.concat(all_groups)


def get_surrounding_character_finder(left: bool, right: bool, count: int):
    assert not (left == right and left), "Getting characters from both sides not implemented"
    if left:
        return lambda combined, operation: get_chars_before_str(combined, operation, count)
    if right:
        return lambda combined, operation: get_chars_after_str(combined, operation, count)

    return lambda combined, operation: ""


def find_surrounding_characters_in_data(data: pd.DataFrame, left: bool, right: bool,
                                        counts: Iterable[int]) -> pd.DataFrame:
    frames = []
    surrounding_chars_column = "Surrounding chars"
    for count in counts:
        if left:
            data_copy: pd.DataFrame = data.copy()
            left_op = get_surrounding_character_finder(True, False, count)
            data_copy[surrounding_chars_column] = data.apply(
                lambda r: left_op(r["Combined"], r["Object"]), axis=1
            )
            frames.append(data_copy)
        if right:
            data_copy: pd.DataFrame = data.copy()
            right_op = get_surrounding_character_finder(False, True, count)
            data_copy[surrounding_chars_column] = data.apply(
                lambda r: right_op(r["Combined"], r["Object"]), axis=1)
            frames.append(data_copy)
    combined = pd.concat(frames, axis=0)
    return combined[combined[surrounding_chars_column].notnull()]


def create_content_matrix_from_data(data: pd.DataFrame):
    return data.groupby(["Object", "Surrounding chars"]).size().unstack(fill_value=0)


def create_context_matrix(objects, first_step_file_name: str, second_step_file_name: str,
                          char_counts: Iterable[int] = (1, 2),
                          left=True, right=True) -> pd.DataFrame:
    first_step = load_csv_to_pandas(first_step_file_name)
    second_step = load_csv_to_pandas(second_step_file_name)
    combined = group_character_operations(first_step, second_step)
    combined_with_chars = find_surrounding_characters_in_data(combined, left, right, char_counts)
    data_with_grammar = _add_grammar_rules_to_data(first_step, combined_with_chars)
    return create_content_matrix_from_data(data_with_grammar)


def _add_grammar_rules_to_data(first_step: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    def chainer(s):
        return list(chain.from_iterable(s.str.split(',')))

    relevant_data = first_step[["Combined", "Grammar"]]
    with_grammar: pd.DataFrame = data.merge(relevant_data, on="Combined")
    lens = with_grammar['Grammar'].str.split(',').map(len)
    grammar_replacing_chars = pd.DataFrame({
        "Object": np.repeat(with_grammar["Object"], lens),
        "Combined": np.repeat(with_grammar["Combined"], lens),
        "Surrounding chars": chainer(with_grammar["Grammar"]),
    })
    chars_and_grammar = pd.concat([data, grammar_replacing_chars])
    return chars_and_grammar


def save_context_matrix(context_matrix: pd.DataFrame, path: str):
    context_matrix.to_csv(path, sep=";", quotechar='"')


def create_and_save_context_matrix(output_path: str, first_step_file_name: str,
                                   second_step_file_name: str, char_counts: Iterable[int] = (1, 2, 3),
                                   left=True, right=True):
    matrix = create_context_matrix(None, first_step_file_name, second_step_file_name, char_counts,
                                   left, right)
    save_context_matrix(matrix, output_path)


def load_context_matrix(path: str, delimiter=";") -> pd.DataFrame:
    return pd.read_csv(path, sep=delimiter, quotechar='"', index_col="Object")

def rewrite_context_matrix_to_concept_lib_format(path: str, output_path: str, threshold:int):
    matrix = load_context_matrix(path)
    to_modify = matrix >= threshold
    not_to_modify = matrix < threshold
    matrix[to_modify] = "X"
    matrix[not_to_modify] = "-"
    matrix.to_csv(output_path, sep=",")

def create_grammar_context_matrix_for_concept_lib_format(concept_lib_ctx_mtx_path: str, output_path: str):
    matrix = load_context_matrix(concept_lib_ctx_mtx_path, delimiter=",")
    grammar_matrix = matrix.filter(regex=r"^(?!(L|R)\().*[^)]$")
    grammar_matrix.to_csv(output_path, sep=",")





# print(create_context_matrix(None, "data/processed/first_step/danish.csv",
#                             "data/processed/second_step/danish.csv"))

# test_data = pd.DataFrame({
#     "Object": ["INS(x)", "INS(x)", "DEL(y)", "INS(x)", "INS(y)", "INS(x)"],
#     "Combined": ["abcINS(x)def", "abcINS(x)xxx", "abDEL(y)cdef", "nINS(x)opqrs", "INS(y)tuwxyzINS(x)", "INS(y)tuwxyzINS(x)"]
# })
#
# print(find_surrounding_characters_in_data(test_data, True, True, [1,2]))
