from typing import Collection, Tuple

import concepts as c

contexts = c.Context.fromfile("data/processed/grammar_context_matrix/danish.csv", "csv")
# print(contexts['DEL(a)',])
for extent, intent in contexts.lattice:
    print(f"Intent: {intent}")
    for object in extent:
        print(object)


def generate_contexts(
        context_matrix_path: str
) -> Collection[Tuple[Collection[str], Collection[str]]]:
    context = _load_context(context_matrix_path)
    return [(intent, extent), context.lattice]


def _load_context(context_matrix_path: str):
    return c.Context.fromfile(context_matrix_path, "csv")
