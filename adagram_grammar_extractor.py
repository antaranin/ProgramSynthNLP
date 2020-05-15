from __future__ import annotations

from typing import Collection, Iterable, List
from mln_model import WeightedOperation, Operation, Context, OpObject
import regex as re


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
    def load_from_text(cls, text_lines: Iterable[str]) -> AdaGram:
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


def _load_grammar_file(grammar_file_path: str) -> AdaGram:
    with open(grammar_file_path, mode="r") as file:
        return AdaGram.load_from_text(file)


def _process_both_sides_grammar(adagram: AdaGram) -> Collection[WeightedOperation]:
    return [_adagram_rule_both_to_weighted_operation(rule) for rule in adagram.rules]


def _adagram_rule_both_to_weighted_operation(rule: AdaGramRule) -> WeightedOperation:
    op_object = re.search(r"((INS)|(DEL))\(.+\)", rule.operation_value).group(0)
    op = op_object[:3]
    object = op_object[4:-1]
    values = rule.operation_value.split(" ")
    op_index = values.index(op_object)
    left_context = "".join(values[:op_index])
    right_context = "".join(values[op_index + 1:])
    operation = Operation(Context(left_context, right_context), OpObject(op, object))
    return WeightedOperation(rule.weight, operation)


if __name__ == '__main__':
    operations = process_grammar_file("data/processed/grammar/adagram/both/asturian.grammar")
    print(operations)
