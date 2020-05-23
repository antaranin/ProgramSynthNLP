# from __future__ import annotations

import json
from enum import Enum
from typing import Collection, Tuple, Union
import abc

from logic_model import LogicClause, Predicate, Imp, And


class JsonSerializable(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **data) -> None:
        pass

    def to_json(self) -> json:
        return json.dumps(self, default=lambda o: JsonSerializable._skip_underscore_dict(o))

    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))

    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    @abc.abstractmethod
    def __eq__(self, o: object) -> bool:
        pass

    @staticmethod
    def _skip_underscore_dict(o: object):
        d = o.__dict__
        return {k: v for k, v in d.items() if k[:1] != "_"}


class Context(JsonSerializable):
    left: str
    right: str
    _hash: int

    def __init__(self, left: str, right: str) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self._hash = hash((self.left, self.right))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Context):
            return False

        return self.left == o.left and self.right == o.right

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"{self.left}, {self.right}"

    def __repr__(self) -> str:
        return str(self)


class OpObject(JsonSerializable):
    operation: str
    operation_object: str
    _hash: int

    def __init__(self, operation: str, operation_object: str) -> None:
        super().__init__()
        self.operation = operation
        self.operation_object = operation_object
        self._hash = hash((self.operation, self.operation_object))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, OpObject):
            return False
        return self.operation == o.operation and self.operation_object == o.operation_object

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"{self.operation}({self.operation_object})"

    def __repr__(self) -> str:
        return str(self)


class Operation(JsonSerializable):
    contexts: Context
    op_and_object: OpObject
    _hash: int

    def __init__(
            self,
            contexts: Union[Context, dict],
            op_and_object: Union[OpObject, dict]
    ) -> None:
        super().__init__()

        self.contexts = contexts if isinstance(contexts, Context) else Context(**contexts)
        self.op_and_object = op_and_object if isinstance(op_and_object, OpObject) else OpObject(
            **op_and_object)
        self._hash = hash((self.contexts, self.op_and_object))

    def to_horne_clause(self) -> LogicClause:
        left = Predicate("L", self.contexts.left)
        right = Predicate("R", self.contexts.right)
        op = Predicate(self.op_and_object.operation, self.op_and_object.operation_object)
        return Imp(And(left, right), op)

    def __str__(self) -> str:
        return f"{self.contexts}, {self.op_and_object}"

    def __repr__(self) -> str:
        return f"[{str(self)}]"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Operation):
            return False
        return self.contexts == o.contexts and self.op_and_object == o.op_and_object

    def __hash__(self) -> int:
        return self._hash


class WeightedOperation:
    weight: float
    operation: Operation
    morph_features: Tuple[str, ...]

    def __init__(self, weight: float, operation: Operation,
                 morph_features: Tuple[str, ...] = ()) -> None:
        super().__init__()
        self.weight = weight
        self.operation = operation
        self.morph_features = morph_features

    def __str__(self) -> str:
        return f"{self.weight}, {self.operation}, {self.morph_features}"

    def __repr__(self) -> str:
        return f"({str(self)})"


class WordAndOps(JsonSerializable):
    word: str
    objects_and_positions: Tuple[Tuple[OpObject, int], ...]

    def __init__(self, word: str,
                 objects_and_positions: Collection[Tuple[Union[OpObject, dict], int]]) -> None:
        super().__init__()
        self.word = word
        temp = []
        for o_p in objects_and_positions:

            if isinstance(o_p[0], OpObject):
                temp.append(o_p)
            else:
                temp.append((OpObject(**o_p[0]), o_p[1]))
        self.objects_and_positions = tuple(temp)

    def __str__(self) -> str:
        return f"{self.word}, {self.objects_and_positions}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.word, self.objects_and_positions))

    def __eq__(self, other) -> bool:
        if not isinstance(other, WordAndOps):
            return False
        return self.objects_and_positions == other.objects_and_positions and self.word == other.word

    def apply(self) -> str:
        ops = sorted(self.objects_and_positions, key=lambda el: (el[1], el[0].operation == "DEL"))
        added = 0
        new_word = self.word
        for op in ops:
            operation, object = op[0].operation, op[0].operation_object
            position = op[1] + added

            if operation == "INS":
                added += len(object)
                new_word = new_word[:position] + object + new_word[position:]
            elif new_word[position:position + len(object)] == object:
                added -= len(object)
                new_word = new_word[:position] + new_word[position + len(object):]
        return new_word


class WordAndApplicable:
    word: str
    applicable_operation: Collection[Operation]


class Evidence:
    is_true: bool
    precondition: str
    items: Tuple[str, ...]
    _hash: int

    def __init__(self, is_true: bool, precondition: str, *items: str) -> None:
        super().__init__()
        self.is_true = is_true
        self.precondition = precondition
        self.items = items
        self._hash = hash((self.is_true, self.precondition, self.items))

    def __str__(self) -> str:
        bool_val = Evidence._format_bool(self.is_true)
        return f"{bool_val}{self.precondition}({str.join(',', self.items)})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return self._hash

    @staticmethod
    def _format_bool(value: bool):
        return "" if value else "!"


class Func:
    func_name: str
    items: Collection[str]

    def __init__(self, func_name: str, *items: str) -> None:
        super().__init__()
        self.name = func_name
        self.items = items

    def __str__(self) -> str:
        return f"{self.name}({str.join(',', self.items)})"

    def __repr__(self) -> str:
        return str(self)


class Precondition:
    weight: float

    def __init__(self, weight: float = 0.0) -> None:
        super().__init__()
        self.weight = weight

    def __str__(self) -> str:
        return f"{self.weight}"

    def __repr__(self):
        return str(self)


class ImpPrecondition(Precondition):
    premise: Func
    consequence: Func

    def __init__(self, premise: Func, consequence: Func, weight: float = 0.0) -> None:
        super().__init__(weight)
        self.premise = premise
        self.consequence = consequence

    def __str__(self) -> str:
        return f"{super().__str__()} {self.premise} => {self.consequence}"

    def __repr__(self):
        return str(self)


class OpMorph:
    morph_feature: str
    operation: OpObject
    count: int

    def __init__(self, morph_feature: str, operation: OpObject, count: int = 0) -> None:
        super().__init__()
        self.morph_feature = morph_feature
        self.operation = operation
        self.count = count

    def __str__(self) -> str:
        return f"(MF: {self.morph_feature}, OP: {self.operation}, count: {self.count})"

    def __repr__(self) -> str:
        return str(self)


class Strictness(Enum):
    Ignore = 0
    AtLeastOne = 1
    MajorityLeftInRight = 2
    MajorityRightInLeft = 3
    AllLeftInRight = 4
    AllRightInLeft = 5
    All = 6

    def compare(self, left: Collection, right: Collection) -> bool:
        if self == Strictness.Ignore:
            return True
        if self == Strictness.AtLeastOne:
            return any(item in right for item in left)
        if self == Strictness.MajorityLeftInRight:
            return Strictness._majority_left_in_right(left, right)
        if self == Strictness.MajorityRightInLeft:
            return Strictness._majority_left_in_right(right, left)
        if self == Strictness.AllLeftInRight:
            return Strictness._all_left_in_right(left, right)
        if self == Strictness.AllRightInLeft:
            return Strictness._all_left_in_right(right, left)
        if self == Strictness.All:
            return set(left) == set(right)

        raise NotImplementedError

    @staticmethod
    def _majority_left_in_right(left: Collection, right: Collection) -> bool:
        return sum(1 for item in left if item in right) > (len(left) / 2)

    @staticmethod
    def _all_left_in_right(left: Collection, right: Collection) -> bool:
        return all(item in right for item in left)


class WordContext(JsonSerializable):
    context: Context
    operation: str
    _hash: int

    def __init__(self, context: Union[Context, dict], operation: str) -> None:
        super().__init__()

        self.context = context if isinstance(context, Context) else Context(**context)
        self.operation = operation
        self._hash = hash((self.context, self.operation))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, WordContext):
            return False
        return self.operation == o.operation and self.context == o.context

    def __str__(self) -> str:
        return f"({self.context}, {self.operation})"

    def __repr__(self) -> str:
        return str(self)

    def applies(
            self,
            word: str,
    ) -> bool:
        return self._context_applies(word)

    def _context_applies(self, word: str) -> bool:
        left = self.context.left
        right = self.context.right
        if self.operation == 'INS':
            return (left + right) in word
        left_start = word.find(left)
        if left_start == -1:
            return False
        right_start = word.find(right, left_start + len(left))
        return right_start != -1

    def is_same_as(self, other) -> bool:
        if self is other:
            return True

        if self.operation != other.operation:
            return False

        are_left_same = self.context.left.endswith(other.context.left) \
                        or other.context.left.endswith(self.context.left)
        are_right_same = self.context.right.startswith(other.context.right) \
                         or other.context.right.startswith(self.context.right)
        return are_left_same and are_right_same
