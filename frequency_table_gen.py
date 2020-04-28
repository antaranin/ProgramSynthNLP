from __future__ import annotations

from typing import Union, Tuple, List, Iterator, Iterable, Collection, Callable, Optional
import pandas as pd
import re
from itertools import chain, combinations


# class ObjectHierarchy:
#     object: str
#     operation: str
#     parents: List[ObjectHierarchy]
#
#     def __init__(self, object: str, operation: str) -> None:
#         super().__init__()
#         self.object = object
#         self.operation = operation
#
#     def try_add_parent(self, object: str, operation: str) -> bool:


class ContextHierarchy:
    context: str
    parents: List[ContextHierarchy]
    left_parents: List[ContextHierarchy]
    right_parents: List[ContextHierarchy]

    def __init__(self, context: str) -> None:
        super().__init__()
        assert context.count("&") == 1, \
            "Context must contain both left and right context separated with &"
        self.context = context
        self.left_parents = []
        self.right_parents = []

    def try_add_parent(self, parent) -> bool:
        assert parent.count("&") == 1, \
            "Parent context must contain both left and right context separated with &"

        if len(self.context) >= len(parent):
            return False

        if not self._is_parent(parent):
            return False

        if self._is_parent_a_direct_parent(parent):
            self._add_direct_parent(parent)
            return True

        found = False
        for l_p in self.left_parents:
            if l_p.try_add_parent(parent):
                found = True
                break

        for r_p in self.right_parents:
            if r_p.try_add_parent(parent):
                found = True
                break

        assert found, "Non direct parent was not added to any of the parents"

        return True

    def get_direct_parents(self) -> Collection[ContextHierarchy]:
        return self.left_parents + self.right_parents

    def _add_direct_parent(self, parent: str):
        left, right = self.context.split("&")
        p_left, p_right = self.context.split("&")

        if len(left) < len(p_left):
            self.left_parents.append(ContextHierarchy(parent))
        else:
            self.right_parents.append(ContextHierarchy(parent))

    def _is_parent(self, parent: str):
        return self.context in parent

    def _is_parent_a_direct_parent(self, parent: str):
        return len(self.context) + 1 == len(parent)


def generate_basic_data_csv(first_step_file_path: str, output_path: str) -> None:
    data: pd.DataFrame = _load_csv_to_pandas(first_step_file_path)
    data["Operations"] = data["Combined"].apply(lambda x: _combined_to_ops(x))
    data = data[["Source", "Combined", "Operations", "Grammar"]][data["Operations"] != ""]
    data = _split_column(
        split_op=lambda d: d["Operations"].str.split(',').tolist(),
        data=data,
        new_column_name="Operation",
        counter_name="Counter"
    )
    # split_ops: pd.Series = pd.DataFrame(data["Operations"].str.split(',').tolist(),
    #                                     index=data.index).stack()
    # split_ops: pd.DataFrame = split_ops.reset_index(level=0)
    # split_ops = split_ops.reset_index()
    # split_ops.columns = ["Counter", "SplitJoiner", "Operation"]
    # data: pd.DataFrame = split_ops.merge(data, left_on="SplitJoiner", right_on=data.index)
    data = data[["Source", "Combined", "Counter", "Operation", "Grammar"]]
    # print(data)
    print("Split ops")
    data: pd.DataFrame = data.apply(
        lambda row: _generate_basic_data_row(row[0], row[1], row[2], row[3], row[4]),
        axis=1
    )
    _save_csv_to_pandas(data, output_path)


def generate_morph_op_table(
        data_file_path: str,
        output_path: str,
        threshold: float = 0.0
):
    data = _load_csv_to_pandas(data_file_path)
    data["OpObject"] = data["Operation"] + "(" + data["Object"] + ")"

    def _grammar_splitter(d: pd.DataFrame) -> Collection[Collection[str]]:
        gram_list = d["Grammar"].str.split(',').tolist()
        all_grammar_combinations = [str_powerset(_powerset(grammars)) for grammars in gram_list]
        return all_grammar_combinations

    data = _split_column(split_op=_grammar_splitter, data=data, new_column_name="MorphFeature")
    data = data[["MorphFeature", "OpObject"]]
    data = _create_support_matrix(data, "MorphFeature", "OpObject")
    data = _create_descriptive_confirmed_confidence_matrix(data)
    data = _threshold_matrix(data, threshold)
    _save_csv_to_pandas(data, output_path, keep_index=True)


def _get_subcontexts(d: pd.DataFrame) -> Collection[Collection[str]]:
    context_list = d[["Left", "Right"]].values.tolist()
    combined_context_list = [_get_all_subcontexts(contexts) for contexts in context_list]
    return combined_context_list


def generate_context_op_table(
        data_file_path: str,
        output_path: str,
        confidence_threshold: float = 0.0,
        support_threshold: float = 4
):
    data = _load_csv_to_pandas(data_file_path)
    data["OpObject"] = data["Operation"] + "(" + data["Object"] + ")"

    data = _split_column(split_op=_get_subcontexts, data=data, new_column_name="Context")
    data = data[["Context", "OpObject"]]
    data = _create_support_matrix(data, "Context", "OpObject")
    # data = _limit_support_context_matrix_to_largest_mrs(data)
    data = _trim_rows_below_threshold(data, support_threshold)
    data = _create_descriptive_confirmed_confidence_matrix(data)
    data = _threshold_matrix(data, confidence_threshold)
    _save_csv_to_pandas(data, output_path, keep_index=True)


def generate_morph_context_op_table(
        data_file_path: str,
        output_path: str,
        confidence_threshold: float = 0.5,
        support_threshold: float = 10
):
    data = _load_csv_to_pandas(data_file_path)
    data["OpObject"] = data["Operation"] + "(" + data["Object"] + ")"
    data = _split_column(split_op=_get_subcontexts, data=data, new_column_name="Context")
    data = _split_column(split_op=lambda d: d["Grammar"].str.split(',').tolist(),
                         data=data,
                         new_column_name="MorphFeature")

    data["ContextMorph"] = data["Context"] + "&" + data["MorphFeature"]
    data = data[["ContextMorph", "OpObject"]]

    data = _create_support_matrix(data, "ContextMorph", "OpObject")
    data = _trim_rows_below_threshold(data, support_threshold)
    data = _create_descriptive_confirmed_confidence_matrix(data)
    data = _threshold_matrix(data, confidence_threshold)
    _save_csv_to_pandas(data, output_path, keep_index=True)


def _split_column(
        *,
        split_op: Callable[[pd.DataFrame], Collection[Collection[str]]],
        data: pd.DataFrame,
        new_column_name: str,
        counter_name: Optional[str] = None
) -> pd.DataFrame:
    split_frame = pd.DataFrame(split_op(data), index=data.index).stack()
    split_frame = split_frame.reset_index(level=0)
    if counter_name is None:
        split_frame.columns = ["SplitJoiner", new_column_name]
    else:
        split_frame = split_frame.reset_index()
        split_frame.columns = [counter_name, "SplitJoiner", new_column_name]
    merged_data: pd.DataFrame = split_frame.merge(data, left_on="SplitJoiner", right_on=data.index)

    return merged_data


def _create_support_matrix(data: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
    return data.groupby([column1, column2]).size().unstack(fill_value=0)


def _create_descriptive_confirmed_confidence_matrix(support_matrix: pd.DataFrame) -> pd.DataFrame:
    x_counts = support_matrix.sum(axis=1)
    confidence_matrix = support_matrix.divide(x_counts, axis=0)
    negative_confidence_matrix = (confidence_matrix - 1) * -1
    return confidence_matrix - negative_confidence_matrix


def _limit_support_context_matrix_to_largest_mrs(support_matrix: pd.DataFrame) -> pd.DataFrame:
    contexts = support_matrix.index.tolist()
    sorted_contexts = sorted(contexts, key=lambda x: len(x))
    root = ContextHierarchy('&')
    for context in sorted_contexts:
        res = root.try_add_parent(context)
        assert res, f"{res} could not be added"
        # TODO finish this by using the ContextHierarchy to find the largest mrs
        # Go through them recursively keeping the lower ones if they are found on their own, otherwise removing them
    standalone_contexts = _get_standalone_contexts_from_parents_of_hierarchy(root, support_matrix)
    return support_matrix.loc[standalone_contexts]


def _get_standalone_contexts_from_parents_of_hierarchy(
        hierarchy: ContextHierarchy,
        support_matrix: pd.DataFrame
) -> Collection[str]:
    return [context for parent in hierarchy.get_direct_parents() for context in
            _get_standalone_contexts_in_hierarchy(parent, support_matrix)]


def _get_standalone_contexts_in_hierarchy(hierarchy: ContextHierarchy,
                                          support_matrix: pd.DataFrame) -> Collection[str]:
    parent_names = [parent.context for parent in hierarchy.get_direct_parents()]
    appearances = support_matrix.loc[hierarchy.context]
    parent_appearances = support_matrix.loc[parent_names]
    total_parent_appearances = parent_appearances.sum(axis=0)
    difference = appearances - total_parent_appearances
    if difference.sum() < 0:
        print(f"Context: {hierarchy.context}")
        print(f"Appearances")
        print(appearances[appearances != 0])
        print(f"Direct Parents: {parent_names}")
        print("Total Parent appearances")
        print(total_parent_appearances[total_parent_appearances != 0])
        assert False
    return []


def _threshold_matrix(matrix: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    return _trim_columns_below_threshold(_trim_rows_below_threshold(matrix, threshold), threshold)


def _trim_columns_below_threshold(matrix: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    return matrix.loc[:, (matrix > threshold).any(axis=0)]


def _trim_rows_below_threshold(matrix: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    return matrix[(matrix > threshold).any(axis=1)]


def _powerset(l: List):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) Does not include empty set"
    return chain.from_iterable(combinations(l, r) for r in range(1, len(l) + 1))


def str_powerset(pow: Iterable[Collection[str]]) -> List[str]:
    return [",".join(sorted(items)) for items in pow]


def _combined_to_ops(combined: str) -> str:
    return ",".join(re.findall(r"[A-Z]{3}\(\w+\)", combined))


def _generate_basic_data_row(
        source: str,
        combined: str,
        counter: int,
        operation: str,
        grammar: str
) -> pd.Series:
    op = operation[:3]
    object = operation[4:-1]
    left, right = _get_left_right_of_op(combined, counter)
    return pd.Series({
        "Source": source,
        "Left": left,
        "Right": right,
        "Operation": op,
        "Object": object,
        "Grammar": grammar
    })


def _get_left_right_of_op(combined: str, counter: int) -> Tuple[str, str]:
    combined = f"<{combined}>"
    operations = re.finditer(r"[A-Z]{3}\(\w+\)", combined)
    this_op: re.Match = next(op for i, op in enumerate(operations) if i == counter)
    op_left, op_right = this_op.span()
    left = combined[:op_left]
    right = combined[op_right:]
    left = _remove_deletes(_remove_inserts(left))
    right = _remove_deletes(_remove_inserts(right))
    return left, right


def _remove_deletes(combined: str) -> str:
    return combined.replace("DEL(", "").replace(")", "")


def _remove_inserts(combined: str) -> str:
    return re.sub(r"INS\([^\)]+\)", "", combined)


def _load_csv_to_pandas(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name, delimiter=";", quotechar='"')


def _save_csv_to_pandas(data_frame: pd.DataFrame, path: str, delimiter=";", keep_index=False):
    data_frame.to_csv(path, sep=delimiter, index=keep_index)
    print(f"Saved to {path}")


def _get_all_subcontexts(contexts: List[str]) -> List[str]:
    left = _get_all_left_subcontexts(contexts[0])
    right = _get_all_right_subcontexts(contexts[1])
    return [f"{l}&{r}" for l in left for r in right if l != r or l != ""]


def _get_all_left_subcontexts(context: str) -> List[str]:
    return [context[i:] for i in range(len(context) + 1)]


def _get_all_right_subcontexts(context: str) -> List[str]:
    return [context[:i] for i in range(len(context) + 1)]

# generate_basic_data_csv("data/processed/first_step/asturian.csv",
#                         "data/processed/context_morph_data/asturian.csv")

# generate_morph_op_table("data/processed/context_morph_data/asturian.csv",
#                         "data/processed/morph_matrix/asturian.csv")

# generate_context_op_table("data/processed/context_morph_data/asturian.csv",
#                           "data/processed/context_matrix/asturian.csv")

# generate_morph_context_op_table("data/processed/context_morph_data/asturian.csv",
#                                 "data/processed/morph_context_matrix/asturian.csv")
