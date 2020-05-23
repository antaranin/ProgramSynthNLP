from typing import Collection
import data_readers as dr
import regex as re

from mln_model import WordContext


def parse_rule(rule_file_path: str) -> Collection[WordContext]:
    with open(rule_file_path, mode="r") as file:
        return [_parse_line_to_rule(line) for line in file]

def _parse_line_to_rule(text_line: str) -> WordContext:
    groups = tuple(re.findall(r"\[.+\]", text_line))
    assert len(groups) == 3
    left, right, morph_features = groups
    operations_str = text_line.split("->")[1]
    operations = re.findall()

