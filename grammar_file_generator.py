import csv
import os
from enum import Enum
from typing import Collection, List

import pandas as pd

from grammar_train_file_generator import SplitType
from mln_model import OpObject
import data_readers as util


class TrainingType(Enum):
    Left = 1
    Right = 2
    Both = 3


ROOT = "Word"
LEFT_SIDE = "LeftSide"
LEFT_OP = "LeftOp"

RIGHT_SIDE = "RightSide"
RIGHT_OP = "RightOp"
RIGHT_MORPH_OP = "RightMOp"
MORPH_AND_OP = "MOp"
LEFT_MORPH_OP = "LeftMOp"

BOTH_TOP = "BothTop"
BOTH_OP = "BothOp"
START = "Start"
END = "End"

CHAR = "Char"
LEFT = "Left"
RIGHT = "Right"
CHARS = "Chars"
START_CHAR = "StartChar"
END_CHAR = "EndChar"
OP = "Op"
# represents the end of string, can be in context
STRING_END = ">"
# represents the beginning of string, can be in context
STRING_START = "<"
# positioned at the beginning of a string, but not actually ever a part of a context
START_CHAR_VAL = "^"
# positioned at the end of a string, but not actually ever a part of a context
END_CHAR_VAL = "$"
LEFT_BRACKET_VAL = "["
RIGHT_BRACKET_VAL = "]"
IMP = "->"
TERM_HEADER = "% terminals\n"
ADAPTED_HEADER = "% adapted non-terminals\n"
NON_TERM_HEADER = "% non-terminals\n"
EMPTY = "\n"

WRAPPED_MORPHS = "WrappedMorphs"
RIGHT_WRAPPED_MORPHS = "RightWrappedMorphs"
MORPH_GROUP = "Morphs"
MORPH = "Morph"
LEFT_BRACKET = "LeftBracket"
RIGHT_BRACKET = "RightBracket"


def generate_grammar_file(alphabet_file_path: str, data_file_path: str, output_path: str,
                          split_type: SplitType, training_type: TrainingType):
    alphabet = util.load_alphabet(alphabet_file_path)
    data = util.load_data_frame(data_file_path)
    op_objects = data[["Operation", "Object"]].apply(_row_to_op_object, axis=1).tolist()
    unique_combined_morph_features = data["Grammar"].unique().tolist()
    morph_features = [combined.split(",") for combined in unique_combined_morph_features]
    grammar = _generate_grammar(alphabet, set(op_objects), morph_features, split_type,
                                training_type)
    util.save_lines_to_file(output_path, grammar)


def _row_to_op_object(row) -> OpObject:
    return OpObject(row["Operation"], row["Object"])


def _generate_grammar(
        alphabet: Collection[str],
        ops_and_objects: Collection[OpObject],
        morph_features: Collection[Collection[str]],
        split_type: SplitType,
        training_type: TrainingType
) -> Collection[str]:
    rules = [NON_TERM_HEADER]
    rules.append(EMPTY)
    rules.extend(_generate_non_terminals(split_type, training_type))
    rules.append(EMPTY)
    rules.append(ADAPTED_HEADER)
    rules.append(EMPTY)
    rules.extend(_generate_adapted_non_terminals(training_type))
    rules.append(EMPTY)
    rules.append(TERM_HEADER)
    rules.append(EMPTY)
    rules.extend(_generate_start_end_terminals())
    rules.append(EMPTY)
    rules.extend(_generate_terminal_ops(ops_and_objects))
    rules.append(EMPTY)
    rules.extend(_generate_char_terminals(alphabet))
    if SplitType.IncludeGrammar in split_type:
        rules.append(EMPTY)
        rules.extend(_generate_morph_feature_terminals(morph_features, split_type))
    return rules


def _generate_non_terminals(split_type: SplitType, training_type: TrainingType) -> Collection[str]:
    rule_gen = {
        TrainingType.Left: _generate_left_train_non_terminals,
        TrainingType.Right: _generate_right_train_non_terminals,
        TrainingType.Both: _generate_both_train_non_terminals
    }
    rules = rule_gen[training_type](split_type)

    rules.append(_generate_rule(START, START_CHAR, CHARS))
    rules.append(_generate_rule(START, START_CHAR))
    rules.append(_generate_rule(END, END_CHAR))
    rules.append(_generate_rule(END, CHARS, END_CHAR))
    rules.append(_generate_rule(LEFT, CHARS))
    rules.append(_generate_rule(RIGHT, CHARS))
    rules.append(_generate_rule(CHARS, CHAR, CHARS))
    rules.append(_generate_rule(CHARS, CHAR))

    return rules


def _generate_both_train_non_terminals(split_type: SplitType) -> List[str]:
    rules = [
        (ROOT, START, BOTH_TOP),
        (BOTH_TOP, BOTH_OP, END)
    ]
    if SplitType.IncludeGrammar in split_type:
        rules.append((BOTH_OP, LEFT, RIGHT_MORPH_OP))
        rules.append((RIGHT_MORPH_OP, MORPH_AND_OP, RIGHT))
        rules.append((MORPH_AND_OP, WRAPPED_MORPHS, OP))
        rules.append((WRAPPED_MORPHS, LEFT_BRACKET, RIGHT_WRAPPED_MORPHS))
        rules.append((RIGHT_WRAPPED_MORPHS, MORPH_GROUP, RIGHT_BRACKET))
        if SplitType.GrammarSymbols in split_type:
            rules.append((MORPH_GROUP, MORPH, MORPH_GROUP))
            rules.append((MORPH_GROUP, MORPH))
    else:
        rules.append((BOTH_OP, LEFT, RIGHT_OP))
        rules.append((RIGHT_OP, OP, RIGHT))
    return [_generate_rule(*rule) for rule in rules]


def _generate_right_train_non_terminals(split_type: SplitType) -> List[str]:
    if SplitType.IncludeGrammar in split_type:
        raise NotImplementedError
    rules = [
        _generate_rule(ROOT, LEFT_SIDE, RIGHT_SIDE),
        _generate_rule(LEFT_SIDE, START, LEFT),
        _generate_rule(RIGHT_SIDE, RIGHT_OP, END),
        _generate_rule(RIGHT_OP, OP, RIGHT)
    ]
    return rules


def _generate_left_train_non_terminals(split_type: SplitType) -> List[str]:
    if SplitType.IncludeGrammar in split_type:
        raise NotImplementedError
    rules = [
        _generate_rule(ROOT, LEFT_SIDE, RIGHT_SIDE),
        _generate_rule(RIGHT_SIDE, RIGHT, END),
        _generate_rule(LEFT_SIDE, START, LEFT_OP),
        _generate_rule(LEFT_OP, LEFT, OP)
    ]
    return rules


def _generate_adapted_non_terminals(training_type: TrainingType) -> Collection[str]:
    rules_to_adapt = {
        TrainingType.Left: LEFT_OP,
        TrainingType.Right: RIGHT_OP,
        TrainingType.Both: BOTH_OP
    }
    return [
        _generate_adapted_rule(rules_to_adapt[training_type], 2000, 100, 0)
    ]


def _generate_rule(lhs: str, *rhs: str) -> str:
    return f"{lhs} {IMP} {' '.join(rhs)}\n"


def _generate_terminal_rule(lhs: str, terminal_value: str) -> str:
    return f'{lhs} {IMP} "{terminal_value}"\n'


def _generate_adapted_rule(item: str, trunc_level: int, alpha: int, beta: int) -> str:
    return f"@ {item} {trunc_level} {alpha} {beta}\n"


def _generate_start_end_terminals() -> Collection[str]:
    rules = [
        _generate_terminal_rule(START_CHAR, START_CHAR_VAL),
        _generate_terminal_rule(END_CHAR, END_CHAR_VAL)
    ]
    return rules


def _generate_char_terminals(alphabet: Collection[str]) -> Collection[str]:
    rules = [_generate_terminal_rule(CHAR, STRING_START), _generate_terminal_rule(CHAR, STRING_END)]

    for c in alphabet:
        rules.append(_generate_terminal_rule(CHAR, c))

    return rules


def _generate_morph_feature_terminals(
        morph_features: Collection[Collection[str]],
        split_type: SplitType
) -> Collection[str]:
    bracket_terminals = [
        _generate_terminal_rule(LEFT_BRACKET, LEFT_BRACKET_VAL),
        _generate_terminal_rule(RIGHT_BRACKET, RIGHT_BRACKET_VAL)
    ]
    if SplitType.GrammarSymbols in split_type:
        terminal_to_use = MORPH
        formatted_features = set(feature for features in morph_features for feature in features)
    else:
        terminal_to_use = MORPH_GROUP
        formatted_features = set(",".join(features) for features in morph_features)
    return bracket_terminals + \
           [_generate_terminal_rule(terminal_to_use, feature) for feature in formatted_features]


def _generate_terminal_ops(ops_and_objects: Collection[OpObject]) -> Collection[str]:
    return [_generate_terminal_rule(OP, str(oo)) for oo in ops_and_objects]


def assert_grammar_and_train_match(grammar_file_path: str, train_file_path: str):
    terminals = _get_terminals_from_grammar(grammar_file_path)
    _assert_train_file_composed_of_terminals(train_file_path, terminals)


def _get_terminals_from_grammar(grammar_file_path: str) -> Collection[str]:
    import regex as re
    all_terminals = []
    with open(grammar_file_path, mode="r") as file:
        for line in file:
            terminals = re.findall('".+"', line)
            all_terminals.extend(t.strip('"') for t in terminals)
    assert len(set(all_terminals)) == len(all_terminals)
    return all_terminals


def _assert_train_file_composed_of_terminals(train_file_path: str, terminals: Collection[str]):
    with open(train_file_path, mode="r") as file:
        for line in file:
            chars = line.strip().split(" ")
            for char in chars:
                if char not in terminals:
                    print(f"Character: {char} from line {line} not in terminals")
                    print(f"Terminals:")
                    for i in range(len(terminals) // 10):
                        print(terminals[i * 10: (i + 1) * 10])
                    raise Exception
#Breaker
#['^', '<', 'E', 'U', '-', 'c', 's', 'a', 't', 'l', 'a', 'k', 'o', 'z', 'á', 's', '[', 'N,DAT,PL', ']', 'INS(oknak)', '>', '$']

if __name__ == '__main__':
    # train_type = TrainingType.Both
    # out_names = {TrainingType.Left: "left", TrainingType.Right: "right", TrainingType.Both: "both"}
    #
    # type_of_split = SplitType(
    #     SplitType.ContextLetters | SplitType.IncludeGrammar
    # )
    # output_dir = f"data/processed/grammar/{out_names[train_type]}"
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # generate_grammar_file(
    #     "data/processed/alphabet/asturian.csv",
    #     "data/processed/context_morph_data/asturian.csv",
    #     f"{output_dir}/asturian.unigram",
    #     type_of_split,
    #     train_type
    # )

    gram_path = "data/processed/grammar/both/hungarian.unigram"
    train_path = "data/processed/grammar/train_data/SplitType_IncludeGrammar_ContextLetters/hungarian.dat"
    assert_grammar_and_train_match(gram_path, train_path)
