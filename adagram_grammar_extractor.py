#from __future__ import annotations

from typing import Collection, Iterable, List
from mln_model import WeightedOperation, Operation, Context, OpObject
import regex as re
import os
import csv


class AdaGramRule:
    weight: float
    operation_name: str
    operation_value: str

    def __init__(self, weight: float, operation_name: str, operation_value: str) -> None:
        super().__init__()
        self.weight = weight
        self.operation_name = operation_name
        self.operation_value = operation_value

    def __str__(self) -> str:
        return f"{self.weight}, {self.operation_name} -> {self.operation_value}"


class AdaGram:
    rules: Collection[AdaGramRule]
    operation_name: str

    def __init__(self, rules: List[AdaGramRule]) -> None:
        super().__init__()
        self.rules = rules
        operation_name = rules[0].operation_name
        assert all(r.operation_name == operation_name for r in rules)
        self.operation_name = operation_name

    @classmethod
    def load_from_text(cls, text_lines: Iterable[str]):
        rules = []
        for line in text_lines:
            if line.startswith("True"):
                rules.append(AdaGram._create_rule_from_data_line(line))

        return cls(rules)

    @staticmethod
    def _create_rule_from_data_line(data_line: str) -> AdaGramRule:
        split_data = data_line.split("\t")
        weight = float(split_data[1].strip())
        str_data = split_data[2].strip()
        relevant_data = re.match(r".+ -> .+ \(", str_data).group(0).strip("(").strip()
        operation_name, operation_value = relevant_data.split("->")
        operation_name = operation_name.strip()
        operation_value = operation_value.strip()
        return AdaGramRule(weight, operation_name, operation_value)


def process_grammar_file(grammar_file_path: str) -> Collection[WeightedOperation]:
    adagram = _load_grammar_file(grammar_file_path)
    return _process_both_sides_grammar(adagram)


def save_grammar_file(grammar_file_path: str, output_file_path: str):
    ops = process_grammar_file(grammar_file_path)
    _write_ops(output_file_path, ops)


def _write_ops(file_path: str, operations: Collection[WeightedOperation]):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(file_path, mode="w+") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Left", "Right", "Operation", "Object", "MorphFeatures", "Weight"])
        for op in operations:
            left = op.operation.contexts.left
            right = op.operation.contexts.right
            operation = op.operation.op_and_object.operation
            object = op.operation.op_and_object.operation_object
            weight = op.weight
            morph_features = ",".join(op.morph_features)
            writer.writerow([left, right, operation, object, morph_features, weight])
    print(f"Wrote grammar to {file_path}")


def _load_grammar_file(grammar_file_path: str):
    with open(grammar_file_path, mode="r") as file:
        return AdaGram.load_from_text(file)


def _process_both_sides_grammar(adagram: AdaGram) -> Collection[WeightedOperation]:
    return [_adagram_rule_both_to_weighted_operation(rule) for rule in adagram.rules]


def _adagram_rule_both_to_weighted_operation(rule: AdaGramRule) -> WeightedOperation:
    op_value = rule.operation_value
    morph_features_match = re.search(r"\[ .+ \]", op_value)
    if morph_features_match is not None:
        morph_features_str: str = morph_features_match.group(0)
        op_value = op_value.replace(morph_features_str + " ", "")
        morph_features_str = morph_features_str.lstrip("[ ").rstrip(" ]")
        morph_features = morph_features_str.split(",") \
            if "," in morph_features_str else morph_features_str.split(" ")
        morph_features = tuple(morph_features)
    else:
        morph_features = tuple()

    op_object = re.search(r"((INS)|(DEL))\(.+\)", op_value).group(0)
    op = op_object[:3]
    object = op_object[4:-1]
    values = op_value.split(" ")
    op_index = values.index(op_object)
    left_context = "".join(values[:op_index])
    left_context = _unescape_spaces(left_context)
    right_context = "".join(values[op_index + 1:])
    right_context = _unescape_spaces(right_context)
    operation = Operation(Context(left_context, right_context), OpObject(op, object))
    return WeightedOperation(rule.weight, operation, morph_features)


def _unescape_spaces(string: str) -> str:
    return string.replace("_", " ")


if __name__ == '__main__':
    language = "livonian"
    save_grammar_file(
       f"data/processed/grammar/adagram/both/{language}.grammar",
       f"data/processed/grammar/adagram/both/{language}.csv"
    )
    #operations = process_grammar_file("data/processed/grammar/adagram/both/asturian.grammar")
    #print(operations)
