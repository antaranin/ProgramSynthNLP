from __future__ import annotations
import numpy as np

from collections import defaultdict
from typing import Collection, Tuple, DefaultDict, List, Union

from mln_model import WeightedOperation, Context, WordAndOps, OpObject, WordContext, Strictness


class Prediction:
    word: str
    prediction: str
    actual: str

    def __init__(self, word: str, prediction: str, actual: str) -> None:
        super().__init__()
        self.word = word
        self.prediction = prediction
        self.actual = actual


class Interpreter:
    contexts: Collection[WordContext]
    operations: DefaultDict[WordContext, Collection[WeightedOperation]]
    probabilistic: bool

    def __init__(self, operations: Collection[WeightedOperation], is_probabilistic: bool) -> None:
        super().__init__()
        self.probabilistic = is_probabilistic
        contexts = set()
        dict_operations = defaultdict(list)
        for operation in operations:
            assert not is_probabilistic or (0 <= operation.weight)
            context = operation.operation.contexts
            word_context = WordContext(context, operation.operation.op_and_object.operation)
            dict_operations[word_context].append(operation)
            contexts.add(word_context)

        self.operations = dict_operations
        self.contexts = contexts

    def find_word_operations(
            self,
            word: str,
            morphological_features: Tuple[str, ...],
            morphological_feature_comparison_strictness: Strictness
    ) -> WordAndOps:
        surrounded_word = f"<{word}>"
        contexts = self._find_contexts_for_word(surrounded_word)
        applicable_operations = self._find_applicable_operations(
            contexts,
            morphological_features,
            morphological_feature_comparison_strictness
        )
        return self._create_word_and_ops(word, applicable_operations)

    def _find_contexts_for_word(self, word: str) -> Collection[WordContext]:
        return [context for context in self.contexts if context.applies(word)]

    def _find_applicable_operations(
            self,
            contexts: Collection[WordContext],
            morph_features: Tuple[str, ...],
            morph_feature_comparison_strictness: Strictness
    ) -> Collection[WeightedOperation]:
        combined_contexts = Interpreter._combine_same_contexts(contexts)

        applicable_operation_groups = [
            [op for context in combined for op in self.operations[context]]
            for combined in combined_contexts
        ]

        morph_viable_groups = []
        for group in applicable_operation_groups:
            filtered_group = Interpreter._filter_morphologically_viable_operations(
                group,
                morph_features,
                morph_feature_comparison_strictness
            )
            if len(filtered_group) > 0:
                morph_viable_groups.append(filtered_group)

        return [self._choose_applicable_operation(ops) for ops in morph_viable_groups]

    @staticmethod
    def _filter_morphologically_viable_operations(
            operations: Collection[WeightedOperation],
            morph_features: Tuple[str, ...],
            morph_feature_comparison_strictness: Strictness
    ) -> Collection[WeightedOperation]:
        compare = lambda op: \
            morph_feature_comparison_strictness.compare(morph_features, op.morph_features)
        return [operation for operation in operations if compare(operation)]

    def _choose_applicable_operation(
            self,
            operations: Collection[WeightedOperation]
    ) -> WeightedOperation:
        if self.probabilistic:
            return self._choose_operation_probabilistically(operations)
        else:
            return max(operations, key=lambda op: op.weight)

    @staticmethod
    def _choose_operation_probabilistically(
            operations: Collection[WeightedOperation]
    ) -> WeightedOperation:
        weights = [op.weight for op in operations]
        probabilistic_weights = Interpreter._adjust_weights_to_probabilistic(weights)
        return np.random.choice(operations, 1, p=probabilistic_weights)[0]

    @staticmethod
    def _adjust_weights_to_probabilistic(weights: Collection[float]) -> Collection[float]:
        total_weight = sum(weights)
        probabilistic_weights = [w / total_weight for w in weights]
        return probabilistic_weights

    @staticmethod
    def _create_word_and_ops(
            word: str, applicable_operations: Collection[WeightedOperation]
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
        base_expected_words_and_morph_features: Collection[Tuple[str, str, Tuple[str, ...]]],
        predicates: Collection[WeightedOperation],
        are_weights_probabilistic: bool,
        morphological_comparison_strictness: Strictness = Strictness.Ignore
) -> Collection[Prediction]:
    interpreter = Interpreter(predicates, are_weights_probabilistic)
    print("Initialized interpreter")
    predictions = []
    total_count = len(base_expected_words_and_morph_features)
    perc = total_count / 100
    count = 0
    for base, expected, morph_features in base_expected_words_and_morph_features:
        if count % perc == 0:
            print(f"Done {count} words, perc: {count / perc}")
        word_and_ops = interpreter.find_word_operations(
            base,
            morph_features,
            morphological_comparison_strictness
        )
        predicted = word_and_ops.apply()
        predictions.append(Prediction(base, predicted, expected))
        count += 1
    return predictions


if __name__ == "__main__":
    pass
