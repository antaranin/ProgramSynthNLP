from copy import deepcopy
from typing import Collection, Tuple, Set, List, Union, Optional
import pandas as pd
import regex as re
import random as rand
import data_readers as dr
from rulesynthesis.agent import Example, State, ParseError

from rulesynthesis.util import Lang, make_hashable, build_sample, build_padded_var
from rulesynthesis.model import Model

MAX_ATTEMPTS = 10
LEFT = "L"
RIGHT = "R"
INSERT = "INS"
DELETE = "DEL"
MORPH_FEATURE = "MORPH"
IGNORE = "IGNORE"
L_PAREN = "["
R_PAREN = "]"


class NLPState:
    examples: Set[Example]
    partial_examples: Set[Example]
    rules: List
    score: int

    @classmethod
    def new(cls, examples: Collection[Example]):
        rules = []
        return cls(set(examples), set(), rules)

    def __init__(self, examples: Set[Example], partial_examples: Set[Example], rules):
        self.examples = examples
        self.partial_examples = partial_examples
        self.rules = rules
        self.score = 0

    def has_examples(self) -> bool:
        return len(self.examples) + len(self.partial_examples) > 0

    def all_examples(self) -> Set[Example]:
        return self.examples.union(self.partial_examples)


class NLPExample:
    input_tokens: Collection[str]
    output_operations: Collection[str]

    def __init__(self, input_tokens: Collection[str], output_operations: Collection[str]) -> None:
        super().__init__()
        self.input_tokens = input_tokens
        self.output_operations = output_operations

    def __str__(self) -> str:
        merged_inputs = "".join(self.input_tokens)
        merged_outputs = " ".join(self.output_operations)
        return f"{merged_inputs} -> {merged_outputs}"


class NLPRule:
    # left-hand-side
    left_context: Collection[str]
    right_context: Collection[str]
    morph_features: Collection[str]
    operation: str
    object: str
    lhs_regex: str

    def __init__(self, left_context: Collection[str], right_context: Collection[str],
                 morph_features: Collection[str], operation: str, object: str):

        # TODO think about using tokens
        self.left_context = left_context
        self.right_context = right_context
        self.morph_features = morph_features
        self.operation = operation
        self.object = object
        self.lhs_regex = self._create_rule_regex()

    def to_action(self) -> List[str]:
        action = []
        action.append(L_PAREN)
        action.extend(self.left_context)
        action.append(R_PAREN)
        action.append(L_PAREN)
        action.extend(self.right_context)
        action.append(R_PAREN)
        action.append(L_PAREN)
        action.append(",".join(self.morph_features))
        action.append(R_PAREN)
        action.append("->")
        action.append(f"{self.operation}({self.object})")
        return action

    @staticmethod
    def can_parse_action_into_rule(action: List[str]):
        general_rule_regex = r"(\[ (. )*\] ){2}\[ \S+ \] ->( ((INS)|(DEL))\([^\)]+\))+"
        return bool(re.fullmatch(general_rule_regex, " ".join(action)))

    @classmethod
    def from_action(cls, action: List[str]):
        try:
            end_left_index = action.index(R_PAREN)
            left = action[1:end_left_index]
            action = action[end_left_index + 1:]
            end_right_index = action.index(R_PAREN)
            right = action[1:end_right_index]
            action = action[end_right_index + 1:]
            morph_features = action[1].split(",")
            op_object = action[len(action) - 1]
            operation = op_object[:3]
            object = op_object[4: -1]
            return cls(left, right, morph_features, operation, object)
        except Exception as e:
            raise e

    def _create_rule_regex(self) -> str:
        left = " ".join(self.left_context)
        right = " ".join(self.right_context)
        morphs = ",".join(self.morph_features)
        if self.operation == INSERT:
            result = rf".*{left} {right}.* {L_PAREN} {morphs} {R_PAREN}"
        elif self.operation == DELETE:
            object = ' '.join(list(self.object))
            result = rf".*{left} {object} {right}.* {L_PAREN} {morphs} {R_PAREN}"
        else:
            raise NotImplementedError(f"Only {INSERT} and {DELETE} operations supported")
        result = result.replace("[", r"\[").replace("]", r"\]")
        return result

    # TODO this might need to be re-written to follow the LHS and RHS style
    def __getstate__(self):
        return (
            self.left_context,
            self.right_context,
            self.morph_features,
            self.operation,
            self.object
        )

    def __setstate__(self, state):
        if type(state) == dict:
            pass

        else:
            left, right, morph, op, object = state
            self.__init__(left, right, morph, op, object)

    def applies(self, rule_input: Union[str, Collection[str]]):
        # return True if the re-write rule applies to this input (string or collection of tokens)
        if isinstance(rule_input, Collection):
            rule_input = " ".join(rule_input)
        try:
            return bool(re.fullmatch(self.lhs_regex, rule_input))
        except Exception as e:
            print(e)
            raise e

    def apply(self, rule_input: Union[str, Collection[str]]):
        # apply rule to input (string, or collection of tokens)
        assert self.applies(rule_input)

        return f"{self.operation}({self.object})"

    def __str__(self):
        return f"{' '.join(self.left_context)} {' '.join(self.right_context)} " + \
               f"{L_PAREN} {','.join(self.morph_features)} {R_PAREN} -> " + \
               f"{self.operation}({self.object})"

    @staticmethod
    def attempt_fix_action(action) -> Optional[str]:
        return None


class NLPGrammar:
    rules: List[NLPRule]

    def __init__(self, rules: List[NLPRule]) -> None:
        super().__init__()

        self.rules = deepcopy(rules)

    def apply(self, input_to_apply_grammar_on: Union[str, Collection[str]]) -> Collection[str]:
        if len(input_to_apply_grammar_on) == 0:
            return []
        ops = []
        for rule in self.rules:
            # TODO Use probability as well?
            if rule.applies(input_to_apply_grammar_on):
                ops.append(rule.apply(input_to_apply_grammar_on))

        result = sorted(ops)
        # This will result in a collection o operations ordered alphabetically
        # A better way would be ordering them by order of appearance,
        # but that might be too complicated for the network to handle
        return result

    def get_useful_rules(self, examples: Set[Example]) -> Collection[NLPRule]:
        useful_rules = set()
        for rule in self.rules:
            applicable_examples = [ex for ex in examples if rule.applies(ex.current)]
            if len(applicable_examples) == 0:
                continue
            score = 0
            for ex in applicable_examples:
                res = rule.apply(ex.current)
                score += 1 if res in ex.target else -1
            if score >= 0:
                useful_rules.add(rule)

        return useful_rules

    def __str__(self):
        s = ''
        for r in self.rules:
            s += str(r) + '\n'
        return s


class TestDataSampler:
    data: pd.DataFrame

    def __init__(self, data_path: str) -> None:
        super().__init__()
        data = pd.read_csv(data_path, sep=";")
        data["Operations"] = data["Combined"].apply(lambda x: TestDataSampler._combined_to_ops(x))
        self.data = data[["Source", "Operations", "Grammar"]][data["Operations"] != ""]

    @staticmethod
    def _combined_to_ops(combined: str) -> str:
        return ",".join(re.findall(r"[A-Z]{3}\(\w+\)", combined))

    def generate_episode(
            self,
            support_set_size: int,
            query_size: int,
            input_language: Lang,
            output_language: Lang,
            program_language: Lang,
            already_generated_episodes: Set[str]
    ):
        for attempt in range(MAX_ATTEMPTS):
            sample_data: pd.DataFrame = self.data.sample(n=support_set_size + query_size, axis=0)
            examples: List[NLPExample] = sample_data.apply(TestDataSampler._data_row_to_example,
                                                           axis=1).tolist()
            data_hash = TestDataSampler._make_example_hash(examples)
            if data_hash in already_generated_episodes:
                continue

            x_total = [e.input_tokens for e in examples]
            y_total = [e.output_operations for e in examples]
            x_support = x_total[:support_set_size]
            y_support = y_total[:support_set_size]
            x_query = x_total
            y_query = y_total
            return build_sample(
                x_support,
                y_support,
                x_query,
                y_query,
                input_language,
                output_language,
                program_language,
                data_hash
            )

    @staticmethod
    def _make_example_hash(examples: Collection[NLPExample]) -> str:
        str_examples = [str(e) for e in examples]
        str_examples = sorted(str_examples)
        return "\n".join(str_examples)

    @staticmethod
    def _data_row_to_example(row) -> NLPExample:
        word = list(row.Source)
        morphs = [L_PAREN, row.Grammar, R_PAREN]
        input_tokens = ["<"] + word + [">"] + morphs
        output_operations = sorted(row.Operations.split(","))
        return NLPExample(input_tokens, output_operations)


class DataSampler:
    data: pd.DataFrame
    alphabet: Collection[str]

    def __init__(self, data_path: str, alphabet: Collection[str]) -> None:
        super().__init__()
        self.alphabet = alphabet
        data = pd.read_csv(data_path, sep=";")
        self.data = data[
            ["Left", "Right", "Operation", "Object", "MorphFeatures"]
        ]

    def generate_episode(
            self,
            support_set_size: int,
            query_size: int,
            rule_count: int,
            input_language: Lang,
            output_language: Lang,
            program_language: Lang,
            already_generated_episodes: Set[str]
    ):
        for attempt in range(MAX_ATTEMPTS):
            sample_data: pd.DataFrame = self.data.sample(n=rule_count, axis=0)
            rules = sample_data.apply(self._data_row_to_grammar_rule, axis=1).tolist()
            grammar = NLPGrammar(rules)
            grammar_hash = make_hashable(grammar, sort=True)
            if grammar_hash in already_generated_episodes:
                continue

            examples = DataSampler._generate_examples_from_grammar(support_set_size + query_size,
                                                                   grammar, self.alphabet)

            x_total = [e.input_tokens for e in examples]
            y_total = [e.output_operations for e in examples]
            x_support = x_total[:support_set_size]
            y_support = y_total[:support_set_size]
            x_query = x_total
            y_query = y_total
            return build_sample(
                x_support,
                y_support,
                x_query,
                y_query,
                input_language,
                output_language,
                program_language,
                grammar_hash,
                grammar
            )

        raise Exception("Couldn't generate a unique new grammar in the given number of attempts")

    @staticmethod
    def _generate_examples_from_grammar(
            example_count: int,
            grammar: NLPGrammar,
            alphabet: Collection[str]
    ) -> Collection[NLPExample]:
        examples = []
        rules = grammar.rules
        rule_count = len(rules)
        assert example_count >= rule_count, "At least one example per rule must be made"
        for i in range(example_count):
            rule_index = i % rule_count
            rule = rules[rule_index]
            input = DataSampler._generate_input_tokens_for_rule(rule, alphabet)
            output = grammar.apply(input)
            assert len(output) > 0
            examples.append(NLPExample(input, output))
        return examples

    @staticmethod
    def _generate_input_tokens_for_rule(rule: NLPRule, characters: Collection[str]) -> Collection[
        str]:
        extra_chars_on_left = rand.randint(0, 5)
        extra_chars_on_right = rand.randint(0, 5)
        left_side = rand.choices(characters, k=extra_chars_on_left) + rule.left_context
        right_side = rule.right_context + rand.choices(characters, k=extra_chars_on_right)
        morphs = [L_PAREN, ",".join(rule.morph_features), R_PAREN]
        if rule.operation == INSERT:
            result = left_side + right_side + morphs
        elif rule.operation == DELETE:
            object = list(rule.object)
            result = left_side + object + right_side + morphs
        else:
            raise NotImplementedError("This operation is not implemented")

        assert rule.applies(result), f"Generated input {result} cannot be used by rule: {rule}"
        return result

    @staticmethod
    def _data_row_to_grammar_rule(row) -> NLPRule:
        left = list(row.Left)
        right = list(row.Right)
        operation = row.Operation
        object = row.Object
        morph_features = row.MorphFeatures.split(",")
        return NLPRule(left, right, morph_features, operation, object)


class NLPLanguage:
    alphabet: Collection[str]
    op_objects: Collection[str]
    morph_features: Collection[str]
    train_data: DataSampler
    test_data: TestDataSampler
    support_set_count: int
    query_set_count: int
    rule_count: int

    # test_data: DataSampler

    def __init__(self, alphabet_file_path: str, data_file: str, train_grammar_path: str,
                 test_data_path: str,
                 support_set_count: int, query_set_count: int, rule_count: int) -> None:
        super().__init__()
        self.alphabet = dr.load_alphabet(alphabet_file_path, include_end_start_symbols=True)
        data = dr.load_data_frame(data_file)
        data["OpObjects"] = data["Operation"] + "(" + data["Object"] + ")"
        self.op_objects = data.drop_duplicates(["OpObjects"])["OpObjects"].tolist()
        self.morph_features = data.drop_duplicates(["Grammar"])["Grammar"].tolist()
        self.train_data = DataSampler(train_grammar_path, self.alphabet)
        self.test_data = TestDataSampler(test_data_path)
        self.support_set_count = support_set_count
        self.query_set_count = query_set_count
        self.rule_count = rule_count

    def get_episode_generator(self):
        input_language = Lang(self._get_input_tokens())
        output_language = Lang(self._get_output_tokens())
        program_language = Lang(self._get_program_tokens())
        # TODO make this code less shit
        generate_episode_from_sampler = \
            lambda data_sampler, already_generated_episodes: data_sampler.generate_episode(
                self.support_set_count,
                self.query_set_count,
                self.rule_count,
                input_language,
                output_language,
                program_language,
                already_generated_episodes
            )
        train_episode_gen = lambda already_generated_episodes: generate_episode_from_sampler(
            self.train_data,
            already_generated_episodes
        )
        # TODO make the test episode use dev data?
        test_episode_gen = lambda already_generated_episodes: self.test_data.generate_episode(
            self.support_set_count,
            self.query_set_count,
            input_language,
            output_language,
            program_language,
            already_generated_episodes
        )

        return train_episode_gen, test_episode_gen, input_language, output_language, program_language

    def _get_input_tokens(self) -> Collection[str]:
        tokens = [L_PAREN, R_PAREN]
        tokens.extend(self.alphabet)
        tokens.extend(self.morph_features)
        return tokens

    def _get_output_tokens(self) -> Collection[str]:
        tokens = [INSERT, DELETE]
        tokens.extend(self.op_objects)
        return tokens

    def _get_program_tokens(self) -> Collection[str]:
        tokens = ["->", "\n"]
        tokens.extend(self._get_input_tokens())
        tokens.extend(self._get_output_tokens())
        return tokens


class NLPModel(Model):
    def sample_to_statelist(self, sample):
        grammar: NLPGrammar = sample['grammar']
        rules_as_actions = [r.to_action() for r in grammar.rules]

        inputs_and_outputs = zip(sample['xs'], sample['ys'])

        examples = {Example(cur, tgt) for cur, tgt in inputs_and_outputs}
        initial_state = State.new(examples)

        states = [initial_state]
        executed_actions = [rules_as_actions]
        return states, executed_actions

    def REPL(self, state: NLPState, action: Collection[str]) -> NLPState:
        if action is not None:
            new_rule_grammar = NLPModel._parse_grammar_from_actions(action)
            useful_new_rules = new_rule_grammar.get_useful_rules(state.all_examples())
            useful_new_action = [r.to_action() for r in useful_new_rules]
            rules = state.rules + useful_new_action
        else:
            rules = state.rules
        grammar = NLPModel._parse_grammar_from_actions(rules)

        new_examples = set()
        partial_new_examples = set()
        score = state.score
        for example in state.examples:
            grammar_output = grammar.apply(example.current)
            if tuple(grammar_output) == example.target:
                score += 2
            elif any(gram_out in example.target for gram_out in grammar_output):
                score += 1
                partial_new_examples.add(example)
            else:
                new_examples.add(example)

        for partial_example in state.partial_examples:
            grammar_output = grammar.apply(partial_example.current)
            if tuple(grammar_output) == partial_example.target:
                score += 1
            else:
                partial_new_examples.add(partial_example)

        new_state = NLPState(new_examples, partial_new_examples, rules)
        new_state.score = score
        return new_state

    @staticmethod
    def _parse_grammar_from_actions(actions: Collection[Collection[str]]) -> NLPGrammar:
        rules = [NLPRule.from_action(action) for action in actions]
        return NLPGrammar(rules)

    def state_rule_to_sample(self, state: State, actions: Collection[Collection[str]]):
        sample = {}

        tokenized_rules = self.tokenize_target_rule(actions)

        sample['grammar'] = tokenized_rules
        sample['identifier'] = "N/A"
        sample['g_padded'], sample['g_length'] = build_padded_var([tokenized_rules], self.prog_lang)

        sample['g_sos_padded'], sample['g_sos_length'] = build_padded_var(
            [tokenized_rules],
            self.prog_lang,
            add_eos=False,
            add_sos=True
        )

        r_support = [self.tokenize_target_rule([past_r]) for past_r in state.rules]  # past_rules ]
        sample['rs'] = r_support
        if r_support:
            sample['rs_padded'], sample['rs_lengths'] = build_padded_var(r_support, self.prog_lang)
        else:
            sample['rs_padded'], sample['rs_lengths'] = [], []

        x_support = []
        y_support = []
        for ex in state.examples:
            x_support.append(list(ex.current))
            y_support.append(list(ex.target))

        sample['xs'] = x_support  # support
        sample['ys'] = y_support
        sample['xs_padded'], sample['xs_lengths'] = build_padded_var(x_support,
                                                                     self.input_lang)  # (ns x max_length)
        sample['ys_padded'], sample['ys_lengths'] = build_padded_var(y_support,
                                                                     self.output_lang)  # (ns x max_length)

        return sample

    def tokenize_target_rule(self, actions: Collection[Collection[str]]):  # ONLY FOR MINISCAN
        tokenized_rules = []
        action_count = len(actions)
        for i, r in enumerate(actions):
            tokenized_rules.extend(r)
            # don't add new line after last rule
            if i + 1 != action_count:
                tokenized_rules.append('\n')
        return tokenized_rules

    def detokenize_action(self, action):
        # split tokens into rule tokens
        rules = []
        rule = []
        for token in action:
            if token == '\n':
                rules.append(rule)
                rule = []
                continue
            else:
                rule.append(token)
        if len(rule) > 0:
            rules.append(rule)
        final_rules = []
        for rule in rules:
            if NLPRule.can_parse_action_into_rule(rule):
                final_rules.append(rule)
            else:
                fixed_rule = NLPRule.attempt_fix_action(rule)
                if fixed_rule is not None:
                    final_rules.append(fixed_rule)

        return final_rules

    def GroundTruthModel(self, state, action):
        if action is None:
            rules = state.rules
        else:
            rules = state.rules + action
        grammar = NLPModel._parse_grammar_from_actions(rules)

        new_examples = []

        for example in state.examples:
            new_example = Example(grammar.apply(example.current), example.target)
            new_examples.append(new_example)

        return State(new_examples, rules)


if __name__ == '__main__':
    # alphabet_file_path = "../data/processed/alphabet/asturian.csv"
    # data_file_path = "../data/processed/context_morph_data/asturian.csv"
    # train_data_file_path = "../data/processed/grammar/adagram/both/asturian.csv"
    # test_data_file_path = "../data/processed/first_step/asturian.csv"
    # sampler = TestDataSampler(test_data_file_path)
    # nlp_lang = NLPLanguage(alphabet_file_path, data_file_path, train_data_file_path,
    #                        test_data_file_path, 5, 2, 0)
    # train_gen, test_gen, input_lang, output_lang, prog_lang = nlp_lang.get_episode_generator()
    # sample = test_gen(set())
    # sample = train_gen(set())
    # rule = NLPRule(7.6, ("a", "b", "c"), ("d", "e", "f"), ("P", "V", "SV"), "INS", "123")
    # action = rule.to_action()
    # print(action)
    # back_to_rule = NLPRule.from_action(action)
    # print(back_to_rule)
    # print(f"Done")
    # sampler = DataSampler("../data/processed/context_morph_data/asturian.csv")
    # sampler.generate_episode(30, 10, 10, None, None, None, set())
    pass
