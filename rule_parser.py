from typing import Collection
import data_readers as dr
import regex as re

from mln_model import WeightedOperation, Operation, Context, OpObject


def parse_rules(rule_file_path: str) -> Collection[WeightedOperation]:
    with open(rule_file_path, mode="r") as file:
        return [_parse_line_to_rule(line) for line in file]

def _parse_line_to_rule(text_line: str) -> WeightedOperation:
    groups = tuple(re.findall(r"\[[^\]]+\]", text_line))
    assert len(groups) == 3, f"Expected 3 groups, but was {groups}"
    left, right, morph_features = groups
    left = left.rstrip("]").lstrip("[")
    right = right.rstrip("]").lstrip("[")
    morph_features = tuple(morph_features.rstrip("]").lstrip("[").split(","))
    operations_str = text_line.split("->")[1]
    operations = tuple(re.findall(r"[A-Z]{3}\([^\)]+\)", operations_str))
    assert len(operations) == 1, f"Expected 1 group but was {operations}"
    only_op = operations[0]
    op = only_op[:3]
    object = only_op[4:-1]
    context = Context(left, right)
    op_object = OpObject(op, object)
    return WeightedOperation(1.0, Operation(context, op_object), morph_features)

if __name__ == "__main__":
    rules = parse_rules("data/processed/models/results/asturian.p")
    for r in rules:
        print(f"{r}")


