import csv
from typing import Dict, Tuple, List, TypeVar, Generic, Type
from uuid import uuid4

from mln_model import JsonSerializable


class KeyBinder():
    _binding: Dict[JsonSerializable, str]
    _reverse_binding: Dict[str, JsonSerializable]

    def __init__(self) -> None:
        super().__init__()
        self._binding = {}
        self._reverse_binding = {}

    def bind(self, item: JsonSerializable) -> str:
        if item in self._binding:
            return self._binding[item]

        key = KeyBinder._gen_key()
        self._binding[item] = key
        self._reverse_binding[key] = item
        return key

    def get_item(self, key: str) -> JsonSerializable:
        return self._reverse_binding[key]

    def get_key(self, item: JsonSerializable) -> str:
        return self._binding[item]

    def to_list(self) -> List[Tuple[str, JsonSerializable]]:
        return [(key, item) for key, item in self._reverse_binding]

    def to_csv(self, file_path: str) -> None:
        with open(file_path, mode="w+") as file:
            writer = csv.writer(file, delimiter=';', quotechar='&', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Key", "Item"])
            for key, item in self._reverse_binding.items():
                writer.writerow([key, item.to_json()])

    @classmethod
    def from_csv(cls, csv_file_path: str, deserialization_type: Type[JsonSerializable]):
        with open(csv_file_path) as file:
            reader = csv.reader(file, delimiter=";", quotechar="&", quoting=csv.QUOTE_MINIMAL)
            next(reader)
            keys_and_items = {row[0]: deserialization_type.from_json(row[1]) for row in reader}
            binder = KeyBinder()
            binder._reverse_binding = keys_and_items
            binder._binding = {item: key for key, item in keys_and_items.items()}

    @staticmethod
    def _gen_key() -> str:
        return f"x{uuid4()}".replace("-", "").upper()
