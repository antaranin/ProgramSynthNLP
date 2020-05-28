import collections
import csv
import subprocess
from typing import List, Tuple, Collection

import more_itertools as mit
import pandas as pd
import regex as re
import Bio.pairwise2 as pair

from data_model import Transformation, Operation, Operations

FREQUENCY_THRESHOLD = 10


# def read_base_expected_words_and_morph_features(
#         first_step_file_path: str
# ) -> Collection[Tuple[str, str, Tuple[str, ...]]]:
#     data = pd.read_csv(first_step_file_path, sep=";")
#     data["Grammar"] = data["Grammar"].str.split(",")
#     data_tuples = data[["Source", "Target", "Grammar"]].values.tolist()
#
#     return tuple((source, target, tuple(grammar)) for source, target, grammar in data_tuples)


def read_base_expected_words_and_morph_features(
        base_data_file_path: str
) -> Collection[Tuple[str, str, Tuple[str, ...]]]:
    data = pd.read_csv(base_data_file_path, sep="\t")
    data.columns = ["Source", "Target", "Grammar"]
    data["Grammar"] = data["Grammar"].str.split(";")
    data_tuples = data[["Source", "Target", "Grammar"]].values.tolist()

    return tuple((source, target, tuple(grammar)) for source, target, grammar in data_tuples if " " not in source and " " not in target)


def read_file_data(file_name: str) -> List[Transformation]:
    lines = open(file_name, mode="r", encoding="utf-8")
    data = []
    for l in lines:
        lemma, form, rule_str = l.split(u'\t')
        rules = rule_str.strip().split(';')
        data.append(Transformation(lemma, form, rules))

    return data


# Inserts gaps in either the Lemma or Inflection to determine the psitions where we have to perform INS or DEL operations
def find_gap_positions_and_matching_characters(gap_string: str, character_string: str) -> List[
    Operation]:
    assert len(gap_string) == len(character_string)
    positions = [m.start() for m in re.finditer('<', gap_string)]
    grouped_positions = group_consecutive_indexes(positions)
    operations = []
    for group in grouped_positions:
        if len(group) == 0:
            continue
        characters = ""
        for index in group:
            characters += character_string[index]
        operations.append(Operation(group[0], characters))
    return operations


# Unused code
# def transform_index_to_directional_index(index: int, string: str):
#     if index > len(string) / 2:
#         return index - len(string)
#     return index


# Grouping gaps to later bundle single-character operations into multi-character operations
def group_consecutive_indexes(indexes: List[int]) -> List[List[int]]:
    return [list(group) for group in mit.consecutive_groups(indexes)]


def get_operations(transformations: List[Transformation]) -> List[Operations]:
    return [find_operations(transformation) for transformation in transformations]


def find_operations(transformation: Transformation) -> Operations:
    # Large penalty to make sure no mismatching characters appear
    penalty = -1000000
    alignements = pair.align.globalms(transformation.lemma, transformation.inflection, 1, penalty,
                                      -1, -0.1, force_generic=True, one_alignment_only=True,
                                      gap_char="<")
    first_alig = alignements[0]
    inserts = find_gap_positions_and_matching_characters(first_alig[0], first_alig[1])
    deletes = find_gap_positions_and_matching_characters(first_alig[1], first_alig[0])
    return Operations(transformation, inserts, deletes)


def group_inserts(operations: List[Operations]):
    inserts = [insert for operation in operations for insert in operation.inserts]
    return group_operations(inserts)


def group_inserts_by_characters_only(operations: List[Operations]):
    inserts = [insert for operation in operations for insert in operation.inserts]
    return group_operation_characters(inserts)


def group_deletes_by_characters_only(operations: List[Operations]):
    deletes = [delete for operation in operations for delete in operation.deletes]
    return group_operation_characters(deletes)


def group_deletes(operations: List[Operations]):
    deletes = [delete for operation in operations for delete in operation.deletes]
    return group_operations(deletes)


def group_operations(operations: List[Operation]):
    counter = collections.Counter(operations)
    return counter


def group_operation_characters(operations: List[Operation]):
    counter = collections.Counter([op.letters for op in operations])
    return counter


def write_first_step(file, operations: List[Operations]):
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Source", "Target", "Combined", "Inserts", "Deletes", "Grammar"])
    for op in operations:
        lemma = op.transformation.lemma
        inflection = op.transformation.inflection
        combined = op.combined()
        ins_steps = ",".join([f"INS({ins.letters})" for ins in op.inserts])
        del_steps = ",".join([f"DEL({delete.letters})" for delete in op.deletes])
        grammar = ",".join(op.transformation.rules)

        writer.writerow([lemma, inflection, combined, ins_steps, del_steps, grammar])


def write_second_step(file, operations: List[Operations]):
    counted_inserts = group_inserts_by_characters_only(operations)
    counted_deletes = group_deletes_by_characters_only(operations)
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["String", "Operation type", "Occurrences"])
    for characters in counted_inserts:
        writer.writerow([characters, "INS", counted_inserts[characters]])
    for characters in counted_deletes:
        writer.writerow([characters, "DEL", counted_deletes[characters]])


def extract_alphabet(words: List[str]):
    alphabet = set()
    for word in words:
        print(word)
        for letter in word:
            if len(letter) > 0:
                alphabet.add(letter.lower())
    return alphabet


def write_alphabet(output_file: str, first_step_file_name: str):
    first_step = pd.read_csv(first_step_file_name, delimiter=";", quotechar='"')
    words = first_step["Source"] + first_step["Target"]
    alphabet = extract_alphabet(words)
    pd.DataFrame(alphabet, columns=["Letter"]).to_csv(output_file, ";")


def write_third_step_ins(file, operations: List[Operations]):
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for op in operations:
        for ins in op.inserts:
            writer.writerow([ins.letters])


def write_third_step_del(file, operations: List[Operations]):
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for op in operations:
        for d in op.deletes:
            writer.writerow([d.letters])


def group_bellow_threshold_inserts(operations: List[Operations]):
    counted_inserts = group_inserts_by_characters_only(operations)
    bellow_threshold_inserts = []
    for characters in counted_inserts:
        if (counted_inserts[characters] <= FREQUENCY_THRESHOLD):
            bellow_threshold_inserts.append(characters)
    # print(f'Counted Inserts: {counted_inserts}')
    return bellow_threshold_inserts


def group_bellow_threshold_deletes(operations: List[Operations]):
    counted_deletes = group_deletes_by_characters_only(operations)
    bellow_threshold_deletes = []
    for characters in counted_deletes:
        if (counted_deletes[characters] <= FREQUENCY_THRESHOLD):
            bellow_threshold_deletes.append(characters)

    return bellow_threshold_deletes


def read_subword_file(file_name: str) -> List[List[str]]:
    lines = open(file_name, mode="r", encoding="utf-8")
    data = []
    for l in lines:
        if (l.find('#') == -1):
            s = l.split(u'<')[0]
            splits = s.strip().split(u' ')
            data.append(splits)

    return data


def prepare_fifth_step(operation: str, operation_splits: List[List[str]],
                       bellow_threshold_operations: List[str]) -> List[List[str]]:
    operation_result = []
    for op in bellow_threshold_operations:
        matches = []
        for splits in operation_splits:
            if op == ''.join(splits):
                matches.append(splits)

        original = f'{operation}({op})'
        if len(matches) == 0:
            operation_result.append([original, ','.join([f'{operation}({c})' for c in op])])
        if len(matches) == 1:
            operation_result.append([original, ','.join([f'{operation}({s})' for s in matches[0]])])
        if len(matches) > 1:
            max_value_index = 0
            max_split_value = 0
            for i, splits in enumerate(matches):
                split_value = 0
                for s in splits:
                    split_value += len(s) ** 2

                if (split_value > max_split_value):
                    max_split_value = split_value
                    max_value_index = i
            operation_result.append(
                [original, ','.join([f'{operation}({s})' for s in matches[max_value_index]])])

        # print(f'Matches found for {op}: {matches}')

    return operation_result


def write_fifth_step(file, operations_result: List[List[str]]):
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Original", "Split"])
    for op in operations_result:
        # print(f'Writing operation: {op}')
        writer.writerow(op)


def process_data_file(file_name, language, perform_step_one=False, perform_step_two=False,
                      perform_step_three=False, perform_step_four=False, perform_step_five=False):
    trans_data = read_file_data(file_name)
    operations = get_operations(trans_data)
    if perform_step_one:
        with open(f'data/processed/first_step/{language}.csv', mode='w+') as file:
            write_first_step(file, operations)
    if perform_step_two:
        with open(f'data/processed/second_step/{language}.csv', mode='w+') as file:
            write_second_step(file, operations)
    if perform_step_three:
        with open(f'data/processed/third_step/ins_{language}.txt', mode='w+') as file:
            write_third_step_ins(file, operations)
        with open(f'data/processed/third_step/del_{language}.txt', mode='w+') as file:
            write_third_step_del(file, operations)
    if perform_step_four:
        subprocess.call(
            f'subword-nmt learn-bpe -s 30 < ./data/processed/third_step/ins_{language}.txt > ./data/processed/fourth_step/ins_{language}.txt',
            shell=True)
        subprocess.call(
            f'subword-nmt learn-bpe -s 30 < ./data/processed/third_step/del_{language}.txt > ./data/processed/fourth_step/del_{language}.txt',
            shell=True)
    if perform_step_five:
        print(f'Processing Language: {language}')
        insert_splits = read_subword_file(f'./data/processed/fourth_step/ins_{language}.txt')
        delete_splits = read_subword_file(f'./data/processed/fourth_step/del_{language}.txt')
        bellow_threshold_inserts = group_bellow_threshold_inserts(operations)
        bellow_threshold_deletes = group_bellow_threshold_deletes(operations)
        # print(f'Insert Splits: {insert_splits}')
        # print(f'Delete Splits: {delete_splits}')
        # print(f'Insert Thresholds: {bellow_threshold_inserts}')
        # print(f'Delete Thresholds: {bellow_threshold_deletes}')
        with open(f'data/processed/subword/{language}.csv', mode='w+') as file:
            prepared_inserts = prepare_fifth_step('INS', insert_splits, bellow_threshold_inserts)
            prepared_deletes = prepare_fifth_step('DEL', delete_splits, bellow_threshold_deletes)
            write_fifth_step(file, prepared_inserts + prepared_deletes)
