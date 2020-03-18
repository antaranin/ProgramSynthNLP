from typing import Tuple

from logic_model import *

INSERT_VAR = "iop"
DELETE_VAR = "dop"
LEFT_VAR = "lop"
RIGHT_VAR = "rop"
GRAMMAR_VAR = "gctx"


def generate_grammar_clauses(
        all_grammar: Collection[str],
        rules_to_ops: Collection[Tuple[Collection[str], Collection[str]]]
) -> Collection[LogicClause]:
    return [_generate_grammar_clause(all_grammar, rule_to_ops) for rule_to_ops in rules_to_ops]


def generate_operation_domains(ops: Collection[str]) -> Collection[DeclarationClause]:
    inserts = [op.lstrip("INS(").rstrip(")") for op in ops if op.startswith("INS")]
    deletes = [op.lstrip("DEL(").rstrip(")") for op in ops if op.startswith("DEL")]
    insert_declaration = DeclarationClause(INSERT_VAR, DomainClause(inserts))
    delete_declaration = DeclarationClause(DELETE_VAR, DomainClause(deletes))
    return insert_declaration, delete_declaration


def generate_grammar_domain(grammar_rules: Collection[str]) -> DeclarationClause:
    rules = [rule.lstrip("G(").rstrip(")") for rule in grammar_rules]
    return DeclarationClause(GRAMMAR_VAR, DomainClause(rules))


def _generate_grammar_clause(
        all_grammar: Collection[str],
        rules_to_ops: Tuple[Collection[str], Collection[str]]
) -> LogicClause:
    used_rules, objects = rules_to_ops
    unused_rules = [rule for rule in all_grammar if rule not in used_rules]
    not_unused = Not(Par(Or.from_many(unused_rules)))
    any_of_the_used = Par(Or.from_many(used_rules))
    any_of_the_operations = Par(Or.from_many(objects))
    any_of_used_none_of_unused = Par(And(not_unused, any_of_the_used))
    implication = Imp(any_of_used_none_of_unused, any_of_the_operations)
    return Weight(1.0, implication)


gramm = ["P", "C", "A", "B", "D", "E", "F"]
gramm = [f"G({g})" for g in gramm]
gramm_used = [gramm[0], gramm[1], gramm[4]]
objects = ["INS(a)", "DEL(b)", "INS(x)"]
op_domains = generate_operation_domains(objects)
print(op_domains[0])
print(op_domains[1])
print(generate_grammar_domain(gramm))
print(_generate_grammar_clause(gramm, (gramm_used, objects)))
