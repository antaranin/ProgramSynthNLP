from typing import Union, Collection, Callable
import abc
import uuid


class LogicClause(abc.ABC):

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass


class EmptyLogicClause(LogicClause):
    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return ""


class UniLogicClause(LogicClause):
    _value: Union[LogicClause, str]

    def __init__(self, value: Union[LogicClause, str]) -> None:
        super().__init__()
        self._value = value

    def __str__(self) -> str:
        return f"{self._value}"

    def __repr__(self) -> str:
        return f"{self._value}"


class Not(UniLogicClause):
    def __str__(self) -> str:
        return f"!{self._value}"

    def __repr__(self) -> str:
        return f"!{self._value}"


class HardConstraint(UniLogicClause):
    def __str__(self) -> str:
        return f"{self._value}."

    def __repr__(self) -> str:
        return f"{self._value}."


class Par(UniLogicClause):
    def __str__(self) -> str:
        return f"({self._value})"

    def __repr__(self) -> str:
        return f"({self._value})"


class BiLogicClause(LogicClause):
    _left: Union[LogicClause, str]
    _right: Union[LogicClause, str]

    def __init__(self, left: Union[LogicClause, str], right: Union[LogicClause, str]) -> None:
        super().__init__()
        self._left = left
        self._right = right

    @staticmethod
    def _from_many(clauses: Collection[Union[LogicClause, str]],
                   creator: Callable[[LogicClause, LogicClause], LogicClause]) -> LogicClause:
        clause_count = len(clauses)
        if clause_count == 0:
            return EmptyLogicClause()

        iterator = iter(clauses)
        clause = next(iterator)
        for item in iterator:
            clause = creator(clause, item)
        return clause


class Weight(BiLogicClause):
    def __init__(self, weight: Union[float, str], clause: Union[LogicClause, str]) -> None:
        super().__init__(weight, clause)

    def __str__(self) -> str:
        return f"{self._left} {self._right}"

    def __repr__(self) -> str:
        return f"{self._left} {self._right}"


class Predicate(BiLogicClause):
    def __init__(self, predicate: str, clause: Union[LogicClause, str]):
        super().__init__(predicate, Par(clause))

    def __str__(self) -> str:
        return f"{self._left}{self._right}"

    def __repr__(self) -> str:
        return str(self)


class Imp(BiLogicClause):
    def __str__(self) -> str:
        return f"{self._left} => {self._right}"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def from_many(clauses: Collection[Union[LogicClause, str]]) -> LogicClause:
        return BiLogicClause._from_many(clauses, lambda l, r: Imp(l, r))


class And(BiLogicClause):
    def __str__(self) -> str:
        return f"{self._left} ^ {self._right}"

    def __repr__(self) -> str:
        return f"{self._left} ^ {self._right}"

    @staticmethod
    def from_many(clauses: Collection[Union[LogicClause, str]]) -> LogicClause:
        return BiLogicClause._from_many(clauses, lambda l, r: And(l, r))


class Or(BiLogicClause):
    def __str__(self) -> str:
        return f"{self._left} v {self._right}"

    def __repr__(self) -> str:
        return f"{self._left} v {self._right}"

    @staticmethod
    def from_many(clauses: Collection[Union[LogicClause, str]]) -> LogicClause:
        return BiLogicClause._from_many(clauses, lambda l, r: Or(l, r))


class Eq(BiLogicClause):
    def __str__(self) -> str:
        return f"{self._left} = {self._right}"

    def __repr__(self) -> str:
        return f"{self._left} = {self._right}"

    @staticmethod
    def from_many(clauses: Collection[Union[LogicClause, str]]) -> LogicClause:
        return BiLogicClause._from_many(clauses, lambda l, r: Eq(l, r))


class InEq(BiLogicClause):
    def __str__(self) -> str:
        return f"{self._left} =/= {self._right}"

    def __repr__(self) -> str:
        return f"{self._left} =/= {self._right}"

    @staticmethod
    def from_many(clauses: Collection[Union[LogicClause, str]]) -> LogicClause:
        return BiLogicClause._from_many(clauses, lambda l, r: InEq(l, r))


class BiImp(BiLogicClause):
    def __str__(self) -> str:
        return f"{self._left} <=> {self._right}"

    def __repr__(self) -> str:
        return f"{self._left} <=> {self._right}"

    @staticmethod
    def from_many(clauses: Collection[Union[LogicClause, str]]) -> LogicClause:
        return BiLogicClause._from_many(clauses, lambda l, r: BiImp(l, r))


class DomainClause(LogicClause):
    _values: Collection[str]
    _str_value: str

    def __init__(self, values: Collection[str]) -> None:
        super().__init__()
        self._values = values
        self._str_value = self._create_str()

    def _create_str(self) -> str:
        value = "{"

        if len(self._values) > 0:
            iterator = iter(self._values)
            value += next(iterator)
            for v in iterator:
                value += f", {v}"
        return value + "}"

    def __str__(self) -> str:
        return self._str_value

    def __repr__(self) -> str:
        return self._str_value


class DeclarationClause(BiLogicClause):

    def __init__(self, variable_name: str, assignment: DomainClause) -> None:
        super().__init__(variable_name, assignment)

    def __str__(self) -> str:
        return f"{self._left} = {self._right}"

    def __repr__(self) -> str:
        return f"{self._left} = {self._right}"



