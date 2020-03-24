from __future__ import annotations

from collections import defaultdict
from typing import Collection, Tuple, DefaultDict, List, Union

from mln_model import WeightedOperation, Context, WordAndOps, OpObject


class Prediction:
    word: str
    prediction: str
    actual: str

    def __init__(self, word: str, prediction: str, actual: str) -> None:
        super().__init__()
        self.word = word
        self.prediction = prediction
        self.actual = actual


class WordContext:
    context: Context
    operation: str
    _hash: int

    def __init__(self, context: Context, operation: str) -> None:
        super().__init__()
        self.context = context
        self.operation = operation
        self._hash = hash((context, operation))

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"({self.context}, {self.operation})"

    def __repr__(self) -> str:
        return str(self)

    def applies(self, word: str) -> bool:
        left = self.context.left
        right = self.context.right
        if self.operation == 'INS':
            return (left + right) in word
        left_start = word.find(left)
        if left_start == -1:
            return False
        right_start = word.find(right, left_start + len(left))
        print(f"Right start: {right_start}")
        return right_start != -1

    def is_same_as(self, other: WordContext) -> bool:
        if self is other:
            return True

        if self.operation != other.operation:
            return False

        are_left_same = self.context.left.endswith(other.context.left) \
                        or other.context.left.endswith(self.context.left)
        are_right_same = self.context.right.startswith(other.context.right) \
                         or other.context.right.startswith(self.context.right)
        return are_left_same and are_right_same


class Interpreter:
    contexts: Collection[WordContext]
    operations: DefaultDict[WordContext, Collection[WeightedOperation]]

    def __init__(self, operations: Collection[WeightedOperation]) -> None:
        super().__init__()
        contexts = set()
        dict_operations = defaultdict(list)
        for operation in operations:
            context = operation.operation.contexts
            word_context = WordContext(context, operation.operation.op_and_object.operation)
            dict_operations[word_context].append(operation)
            contexts.add(word_context)

        self.operations = dict_operations
        self.contexts = contexts

    def find_word_operations(self, word: str) -> WordAndOps:
        surrounded_word = f"<{word}>"
        contexts = self._find_contexts_for_word(surrounded_word)
        applicable_operations = self._find_applicable_operations(contexts)
        return self._create_word_and_ops(word, applicable_operations)

    def _find_contexts_for_word(self, word: str) -> Collection[WordContext]:
        return [context for context in self.contexts if context.applies(word)]

    def _find_applicable_operations(self, contexts: Collection[WordContext]
                                    ) -> Collection[WeightedOperation]:
        combined_contexts = Interpreter._combine_same_contexts(contexts)

        applicable_operation_groups = [
            [op for context in combined for op in self.operations[context]]
            for combined in combined_contexts
        ]
        return [max(ops, key=lambda op: op.weight) for ops in applicable_operation_groups]

    def _create_word_and_ops(self, word: str, applicable_operations: Collection[WeightedOperation]
                             ) -> WordAndOps:
        ops = [
            Interpreter._weighted_op_to_op_and_position(word, op) for op in applicable_operations
        ]
        return WordAndOps(word, ops)

    @staticmethod
    def _weighted_op_to_op_and_position(word: str, weighted_op: WeightedOperation
                                        ) -> Union[Tuple[OpObject, int], None]:
        context = weighted_op.operation.contexts
        op_and_object = weighted_op.operation.op_and_object
        if weighted_op.operation.op_and_object.operation == "INS":
            position = word.find(context.left + context.right)
        else:
            position = Interpreter._get_delete_op_position_in_word(word, context, op_and_object)

        return weighted_op.operation.op_and_object, position

    @staticmethod
    def _get_delete_op_position_in_word(word: str, context: Context, operation: OpObject) -> int:
        assert operation.operation == "DEL"

        left = context.left
        right = context.right

        left_position = word.find(left)
        if left_position == -1:
            return -1

        position = left_position + len(left)

        right_position = word.find(right, position, len(word))
        if right_position == -1:
            return -1

        if word[position:right_position] != operation.operation_object:
            return -1
        return position

    @staticmethod
    def _combine_same_contexts(ctxts: Collection[WordContext]):
        combined: List[List[WordContext]] = []
        for ctx in ctxts:
            was_combined = False
            for comb in combined:
                if ctx.is_same_as(comb[0]):
                    comb.append(ctx)
                    was_combined = True
                    break
            if not was_combined:
                combined.append([ctx])
        return combined


def run_predictions(
        base_and_expected_words: Collection[Tuple[str, str]],
        predicates: Collection[WeightedOperation]
) -> Collection[Prediction]:
    interpreter = Interpreter(predicates)
    print("Initialized interpreter")
    predictions = []
    total_count = len(base_and_expected_words)
    perc = total_count / 100
    count = 0
    for base, expected in base_and_expected_words:
        if count % perc == 0:
            print(f"Done {count} words, perc: {count / perc}")
        word_and_ops = interpreter.find_word_operations(base)
        predicted = word_and_ops.apply()
        predictions.append(Prediction(base, predicted, expected))
        count += 1
    return predictions
