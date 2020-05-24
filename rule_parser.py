from typing import Collection, Optional, Tuple, List, Set
import data_readers as dr
import regex as re
import os
import json

from mln_model import WeightedOperation, Operation, Context, OpObject


def parse_rules(rule_file_path: str) -> Tuple[dict, Collection[WeightedOperation]]:
    with open(rule_file_path, mode="r") as file:
        # skip the stats
        stat_line = next(file).replace("'", '"')
        stats = json.loads(stat_line)
        return stats, [_parse_line_to_rule(line) for line in file]


def parse_combine_rules(
        rule_dir: str,
        base_file_name: str,
        top_quality_perc: Optional[float] = None
) -> Set[WeightedOperation]:
    files = [file for file in os.listdir(rule_dir) if bool(re.fullmatch(rf"{base_file_name}_\d+", file))]
    scores_and_rules: List[Tuple[int, Collection[WeightedOperation]]] = []
    for file in files:
        file_stats, file_rules = parse_rules(os.path.join(rule_dir, file))
        scores_and_rules.append((file_stats["query_score"], file_rules))
    if top_quality_perc is None:
        return {rule for _, rules in scores_and_rules for rule in rules}

    scores_and_rules = sorted(scores_and_rules, key=lambda r_s: r_s[0], reverse=True)
    print([score for score, _ in scores_and_rules])
    high_score = scores_and_rules[0][0]
    min_score = high_score * top_quality_perc
    return {rule for score, rules in scores_and_rules if score >= min_score for rule in rules}


def _parse_line_to_rule(text_line: str) -> WeightedOperation:
    groups = tuple(re.findall(r"\[[^\]]*\]", text_line))
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
    # stats, res_rules = parse_rules("data/processed/models/results/asturian_1")
    # print(stats)
    # for r in res_rules:
    #     print(f"{r}")
    ops = parse_combine_rules("data/processed/models/results", "asturian", 1)
    print(ops)
