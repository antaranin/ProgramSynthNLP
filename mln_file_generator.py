import csv
import re
from timeit import default_timer as timer
from typing import Set

import pandas as pd

from key_binder import KeyBinder
from mln_model import *

CONTEXT_FUNC = "AppC"
OBJECT_FUNC = "AppO"


def get_words_and_ops(first_step_file_path: str) -> Collection[WordAndOps]:
    combined = pd.read_csv(first_step_file_path, delimiter=";")[["Source", "Combined"]] \
        .values.tolist()
    return [_combined_string_to_word_and_ops(*c) for c in combined]


def _combined_string_to_word_and_ops(initial_word: str, combined: str) -> WordAndOps:
    inserts = re.finditer(r"INS\(\w+\)", combined)
    deletes = re.finditer(r"DEL\(\w+\)", combined)
    insert_ops = [(OpObject("INS", m[0].lstrip("INS(").rstrip(")")), m.start(0)) for m in inserts]
    delete_ops = [(OpObject("DEL", m[0].lstrip("DEL(").rstrip(")")), m.start(0)) for m in deletes]

    return WordAndOps(f"<{initial_word}>", insert_ops + delete_ops)


def get_operations(operation_file_path: str) -> Collection[Operation]:
    with open(operation_file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        return [Operation(Context(line[0], line[1]), OpObject(line[2], line[3])) for line in reader]


def generate_mln_files(
        precondition_file_path: str,
        evidence_file_path: str,
        context_file_path: str,
        object_file_path: str,
        word_file_path: str,
        combined_words: Collection[WordAndOps],
        operations: Collection[Operation]
):
    context_binder = _bind_items([op.contexts for op in operations])
    print("Bound contexts")
    all_objects = [op.op_and_object for op in operations] + \
                  [op[0] for word in combined_words for op in word.objects_and_positions]
    object_binder = _bind_items(all_objects)
    print("Bound objects")
    word_binder = _bind_items(combined_words)
    print("Bound words")
    declarations = _generate_precondition_declarations()
    preconditions = _generate_preconditions(context_binder, object_binder, operations)
    print("Generated preconditions")

    context_binder.to_csv(context_file_path)
    object_binder.to_csv(object_file_path)
    word_binder.to_csv(word_file_path)

    _save_mln_file(declarations, preconditions, precondition_file_path)
    del declarations, preconditions
    evidence = _generate_evidence(context_binder, object_binder, word_binder, combined_words,
                                  operations)
    print("Generated evidence")

    _save_db_file(evidence_file_path, evidence)


def _generate_evidence(
        context_binder: KeyBinder,
        object_binder: KeyBinder,
        word_binder: KeyBinder,
        combined_words: Collection[WordAndOps],
        operations: Collection[Operation]
) -> Collection[Evidence]:
    evidences = []
    word_count = len(combined_words)
    op_count = len(operations)
    gen_count = word_count * op_count
    perc = gen_count / 100
    i = 0
    print(f"Generating evidence from {word_count} words and {op_count} operations:")
    for word in combined_words:
        print(f"Progress: {i / perc}%, total {gen_count}")
        object_evidence = _generate_object_evidence_for_word(object_binder, word_binder, word)
        evidences.extend(object_evidence)
        start = timer()
        for operation in operations:
            i += 1
            context_evidence = _generate_context_evidence_for_word(
                context_binder,
                word_binder,
                word,
                operation.contexts
            )
            if context_evidence is not None:
                evidences.append(context_evidence)
        end = timer()
        print(f"Word took {end - start} seconds, there is {len(evidences)} now")

    return evidences


def _generate_evidence_for_word_and_operation(
        context_binder: KeyBinder,
        object_binder: KeyBinder,
        word_binder: KeyBinder,
        word: WordAndOps,
        operation: Operation,
        set_to_update: Set[Evidence]
) -> Collection[Evidence]:
    context = operation.contexts
    object = operation.op_and_object
    _add_if_not_none(set_to_update,
                     _generate_context_evidence_for_word(context_binder, word_binder, word,
                                                         context))
    _add_if_not_none(set_to_update,
                     _generate_object_evidence_for_word(object_binder, word_binder, word, object))
    return set_to_update


def _add_if_not_none(coll: Set, item):
    if item is not None:
        coll.add(item)


def _generate_context_evidence_for_word(
        context_binder: KeyBinder,
        word_binder: KeyBinder,
        word: WordAndOps,
        context: Context
) -> Union[Evidence, None]:
    if (context.left + context.right) not in word.word:
        return None
    word_key = word_binder.get_key(word)
    context_key = context_binder.get_key(context)
    return Evidence(True, CONTEXT_FUNC, word_key, context_key)


def _generate_object_evidence_for_word(
        object_binder: KeyBinder,
        word_binder: KeyBinder,
        word: WordAndOps,
) -> Collection[Evidence]:
    word_key = word_binder.get_key(word)
    return [
        Evidence(True, OBJECT_FUNC, word_key, object_binder.get_key(o[0]))
        for o in word.objects_and_positions
    ]


def _generate_preconditions(
        context_binder: KeyBinder,
        object_binder: KeyBinder,
        operations: Collection[Operation]
) -> Collection[Precondition]:
    preconditions = []
    for op in operations:
        context_func = Func(CONTEXT_FUNC, "w", context_binder.get_key(op.contexts))
        object_func = Func(OBJECT_FUNC, "w", object_binder.get_key(op.op_and_object))
        preconditions.append(ImpPrecondition(context_func, object_func))
    return preconditions


def _generate_precondition_declarations() -> Collection[Func]:
    context_declaration = Func(CONTEXT_FUNC, "word", "context")
    object_declaration = Func(OBJECT_FUNC, "word", "object")
    return context_declaration, object_declaration


def _save_mln_file(
        precondition_declarations: Collection[Func],
        preconditions: Collection[Precondition],
        mln_file_path: str
) -> None:
    with open(mln_file_path, mode="w+") as file:
        file.writelines([f"{declaration}\n" for declaration in precondition_declarations])
        file.writelines([f"{precondition}\n" for precondition in preconditions])


def _save_db_file(
        db_file_path: str,
        evidence: Collection[Evidence]
) -> None:
    with open(db_file_path, mode="w+") as file:
        file.writelines([f"{e}\n" for e in evidence])


def _bind_items(items: Collection[JsonSerializable]) -> KeyBinder:
    binder = KeyBinder()
    for item in items:
        binder.bind(item)
    return binder


# w = Operation(Context("pp", "y>"), OpObject("INS", "ill"))
#
# jsoninsed = json.dumps(w, default=lambda o: o.__dict__)
# print(jsoninsed)
# print(Operation.from_json(jsoninsed))
#
# c1 = Context("A", "B")
# c2 = Context("C", "D")
# c3 = Context("E", "F")
#
# csv_path = "data/processed/mln/contexts/test.csv"
# binder = KeyBinder()
# binder.bind(c1)
# binder.bind(c2)
# binder.bind(c3)
# binder.to_csv(csv_path)
# binder.from_csv(csv_path, Context)
#
# print(Evidence(True, "AC", "Happy", "xx"))

words_and_ops = get_words_and_ops("data/processed/first_step/asturian.csv")[:1000]
mln_dir = "data/processed/mln"
operations = get_operations(f"{mln_dir}/operations/asturian.csv")

generate_mln_files(
    f"{mln_dir}/precondition/asturian.mln",
    f"{mln_dir}/evidence/asturian2.db",
    f"{mln_dir}/contexts/asturian.csv",
    f"{mln_dir}/objects/asturian.csv",
    f"{mln_dir}/words/asturian.csv",
    words_and_ops,
    operations
)
