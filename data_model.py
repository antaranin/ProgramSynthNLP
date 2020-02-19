from typing import List


class Transformation:
    inflection: str
    lemma: str
    rules: List[str]

    def __str__(self) -> str:
        return f"(lemma: {self.lemma}, inflection: {self.inflection}, rules: {self.rules})"

    def __repr__(self) -> str:
        return f"(lemma: {self.lemma}, inflection: {self.inflection}, rules: {self.rules})"

    def __init__(self, lemma: str, inflection: str, rules: list) -> None:
        super().__init__()
        self.inflection = inflection
        self.lemma = lemma
        self.rules = rules


class Operation:
    index: int
    letters: str

    def __init__(self, index: int, letters: str) -> None:
        super().__init__()
        self.index = index
        self.letters = letters

    def __str__(self) -> str:
        return f"(index: {self.index}, letters: {self.letters})"

    def __repr__(self) -> str:
        return f"(index: {self.index}, letters: {self.letters})"

    def __eq__(self, o: object) -> bool:
        return self.index == o.index and self.letters == o.letters

    def __hash__(self) -> int:
        return hash((self.index, self.letters))


class Operations:
    transformation: Transformation
    inserts: List[Operation]
    deletes: List[Operation]

    def __init__(self, transformation: Transformation, inserts: List[Operation],
                 deletes: List[Operation]) -> None:
        super().__init__()
        self.transformation = transformation
        self.inserts = inserts
        self.deletes = deletes

    def __str__(self) -> str:
        return f"(transformation: {self.transformation}, inserts: {self.inserts}, deletes: {self.deletes})"

    def __repr__(self) -> str:
        return f"(transformation: {self.transformation}, inserts: {self.inserts}, deletes: {self.deletes})"

    def combined(self) -> str:
        total = self.transformation.lemma
        ops = [(True, ins) for ins in self.inserts] + [(False, delete) for delete in self.deletes]
        ops.sort(key=lambda x: x[1].index)
        char_added = 0
        for op in ops:
            if op[0]:
                ins = op[1]
                insert_index = ins.index + char_added
                total = total[:insert_index] + f"INS({ins.letters})" + total[insert_index:]
                char_added += 5
            else:
                delete = op[1]
                delete_index = delete.index + char_added
                total = total[:delete_index] + f"DEL({delete.letters})" + total[delete_index + len(
                    delete.letters):]
                char_added += 5

        return total
