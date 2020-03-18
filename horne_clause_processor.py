from typing import List, Tuple

import pandas as pd
from mln_model import *
import csv


def save_as_operations(input_file_path: str, output_file_path: str):
    data = pd.read_csv(input_file_path)
    relevant_data = data[["conComb", "objF"]]
    operations = _format_as_operations(relevant_data)
    _save_operations(output_file_path, operations)


def _save_operations(file_path: str, operations: Collection[Operation]):
    with open(file_path, mode="w+") as file:
        writer = csv.writer(file)
        writer.writerow(["Left", "Right", "Operation", "Object"])
        for operation in operations:
            writer.writerow([operation.contexts.left,
                              operation.contexts.right,
                              operation.op_and_object.operation,
                              operation.op_and_object.operation_object])



def _format_as_operations(clauses: pd.DataFrame) -> List[Operation]:
    combined_and_object: List = clauses.values.tolist()
    return [_format_as_operation(co) for co in combined_and_object]


def _format_as_operation(combined_and_operation: Tuple[str, str]) -> Operation:
    items, object = combined_and_operation
    items = items.lstrip("(").rstrip(")").split(",")
    items = [item.strip().strip("'") for item in items]
    left, right, op = items
    object = object.lstrip("(").rstrip(")")
    op = op.upper()
    object_and_op = OpObject(op, object)
    context = Context(left, right)
    return Operation(context, object_and_op)


path = "data/horne_clauses/insHorns.csv"
# print(load_horne_clauses(path))
save_as_operations(path, "data/processed/mln/operations/asturian.csv")

