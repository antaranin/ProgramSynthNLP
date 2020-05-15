from copy import deepcopy
from typing import Collection, Tuple, Set, List
import pandas as pd
import regex as re

from rulesynthesis.util import Lang

LEFT = "L"
RIGHT = "R"
INSERT = "INS"
DELETE = "DEL"
MORPH_FEATURE = "MORPH"
IGNORE = "IGNORE"


class NLPRule:
    # left-hand-side
    LHS_str: str = ''
    LHS_list: List[str] = []
    LHS_regexp: str = ''

    # right-hand-side
    RHS_str: str = ''
    RHS_list: List[str] = []

    def __init__(self, LHS, RHS):
        # LHS : string with variables (no interpretation symbols [ or ] )
        # RHS : string with variables (can have interpretation symbols for recursive computation)

        self.LHS_str = LHS
        self.LHS_list = LHS.split()
        self.RHS_str = RHS
        self.RHS_list = RHS.split()

        self.RHS_exp = self._detokenize(self.RHS_list)

        # self.RHS_list = split_special(RHS)

    def __getstate__(self):
        return (self.LHS_str, self.RHS_str)

    def __setstate__(self, state):
        if type(state) == dict:
            pass

        else:
            LHS, RHS = state
            self.__init__(LHS, RHS)

    #TODO make it work with actual tokens instead of letters
    def _detokenize(self, RHS) -> str:
        return ' '.join(RHS)

    def _is_ignore(self, var: str) -> bool:
        return var == IGNORE

    def set_primitives(self, list_prims):
        # list_prims : list of the primitive symbols
        self.list_prims = list_prims
        self.ignore_regex = '([(' + '|'.join(self.list_prims + [' ']) + ')]+)'

        # get list of all variables in LHS
        self.vars = [v for v in self.LHS_list if self._is_ignore(v)]

        # Compute the regexp for checking whether the rule is active
        mylist = deepcopy(self.LHS_list)

        self.LHS_regexp = ''
        for i, x in enumerate(mylist):
            if self._is_ignore(x):
                mylist[i] = self.ignore_regex

            if self.LHS_regexp == '':
                self.LHS_regexp += mylist[i]
            else:
                self.LHS_regexp += ' ' + mylist[i]


    def applies(self, s):
        # return True if the re-write rule applies to this string
        return  bool(re.fullmatch(self.LHS_regexp, s))

    def apply(self, string_to_apply_the_rule_on):
        # apply rule to string s
        assert self.applies(string_to_apply_the_rule_on)

        return self._detokenize(self.RHS_list)

    def __str__(self):
        return str(self.LHS_str) + ' -> ' + str(self.RHS_str)

class NLPGrammar:
    rules: List[NLPRule]

    def __init__(self, rules: List[NLPRule], primitives: List[str]) -> None:
        super().__init__()


class DataSampler:
    data: pd.DataFrame

    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data = pd.read_csv(data_path, sep=";")

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
        sample_data: pd.DataFrame = self.data.sample(n=rule_count, axis=0)
        rules = sample_data.apply(self._data_row_to_grammar_rule)

    print(sample_data)
    pass


def _data_row_to_grammar_rule(self, left: str, right: str, operation: str, object: str,
                              grammar: str) -> Rule:
    pass


class NLPLanguage:
    alphabet: Collection[str]
    train_data: DataSampler
    test_data: DataSampler
    objects: Collection[str]

    def get_episode_generator(self):
        input_languae = Lang(self._get_input_tokens())
        output_languae = Lang(self._get_output_tokens())
        program_languae = Lang(self._get_program_tokens())

    def _get_input_tokens(self) -> Collection[str]:
        tokens = [LEFT, RIGHT]
        tokens.extend(self.alphabet)
        return tokens

    def _get_output_tokens(self) -> Collection[str]:
        tokens = [INSERT, DELETE]
        tokens.extend(self.objects)
        return tokens

    def _get_program_tokens(self) -> Collection[str]:
        tokens = ['->', '\n', 'x1', 'x2', 'u1', 'u2']
        tokens.extend(self._get_input_tokens())
        tokens.extend(self._get_output_tokens())
        return tokens


if __name__ == '__main__':
    sampler = DataSampler("../data/processed/context_morph_data/asturian.csv")
    sampler.generate_episode(30, 10, 10, None, None, None, set())
