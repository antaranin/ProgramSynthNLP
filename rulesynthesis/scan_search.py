# scan_search.py


# evaluate.py

import torch
import argparse
import os
import time

import math

import dill
from scipy import stats as scistats

from rulesynthesis import util
from rulesynthesis.agent import Example, State
from rulesynthesis.model import MiniscanRBBaseline, WordToNumber
from rulesynthesis.nlp import NLPModel
from rulesynthesis.util import get_episode_generator, timeSince, tabu_update, cuda_a_dict, \
    generate_batchsize_of_samples
from rulesynthesis.train import gen_samples, train_batched_step, eval_ll
from rulesynthesis.test import batched_test_with_sampling
from rulesynthesis.generate_episode import exact_perm_doubled_rules
from rulesynthesis.interpret_grammar import Grammar

from collections import namedtuple

SearchResult = namedtuple("SearchResult", "hit solution stats")


# 50R
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length_eval', type=int, default=15,
                        help='maximum generated sequence length when evaluating the network')
    parser.add_argument('--max_num_rules', type=int, default=12, help='maximum generated num rules')
    parser.add_argument('--episode_type', type=str, default='auto')
    parser.add_argument('--fn_out_model', type=str, default='',
                        help='filename for saving the model')
    parser.add_argument('--dir_model', type=str, default='out_models',
                        help='directory for saving model files')
    parser.add_argument('--gpu', type=int, default=0, help='set which GPU we want to use')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--mode', type=str, default='sample', choices=['smc', 'sample'])  # beam,
    parser.add_argument('--type', type=str, default="miniscanRBbase")
    parser.add_argument('--savefile', type=str, default="results/smcREPL.p")
    parser.add_argument('--use_large_support', action='store_true')
    parser.add_argument('--use_rules_hard', action='store_true')
    parser.add_argument('--use_scan_large_s', action='store_true')
    parser.add_argument('--new_test_ep', type=str, default='')
    parser.add_argument('--load_data', type=str, default='')
    parser.add_argument('--val_ll', action='store_true')
    parser.add_argument('--val_ll_only', action='store_true')
    parser.add_argument('--n_test', type=int, default=20)
    parser.add_argument('--duplicate_test', action='store_true')
    parser.add_argument('--positional', action='store_true',
                        default=True)  # positional rule encodings
    parser.add_argument('--hack_gt_g', action='store_true')
    parser.add_argument('--nosearch', action='store_true')
    parser.add_argument('--partial_credit', action='store_true', default=True)
    parser.add_argument('--n_runs', type=int, default=20)
    parser.add_argument('--alphabet_file_path', type=str)
    parser.add_argument('--test_data_file_path', type=str)
    parser.add_argument('--data_file_path', type=str)
    parser.add_argument('--grammar_file_path', type=str)
    parser.add_argument('--rule_count', type=int, default=100)
    parser.add_argument('--support_set_count', type=int, default=200)
    parser.add_argument('--query_set_count', type=int, default=100)
    res_args = parser.parse_args(args)
    if res_args.val_ll_only:
        res_args.val_ll = True
    return res_args


def _search(args):
    path = os.path.join(args.dir_model, args.fn_out_model)
    util.alphabet_path = args.alphabet_file_path
    util.data_file_path = args.data_file_path
    util.grammar_path = args.grammar_file_path
    util.test_data_file_path = args.test_data_file_path
    util.rule_count = args.rule_count
    util.support_set_count = args.support_set_count
    util.query_set_count = args.query_set_count

    # load model
    if args.type == 'miniscanRBbase':
        model = MiniscanRBBaseline.load(path)
    elif args.type == 'WordToNumber':
        model = WordToNumber.load(path)
    elif args.type == "NLP":
        model = NLPModel.load(path)
    else:
        assert False, "not implemented yet"

    time_list = []
    n_nodes_list = []
    n_ex_list = []
    for i in range(args.n_runs):
        times, n_nodes, n_ex = run_search(model, args)
        time_list.append(times)
        n_nodes_list.append(n_nodes)
        n_ex_list.append(n_ex)

    avg_time = sum(time_list) / len(time_list)
    error_time = scistats.sem(time_list)

    avg_n_nodes = sum(n_nodes_list) / len(n_nodes_list)
    error_n_nodes = scistats.sem(n_nodes_list)

    avg_n_ex = sum(n_ex_list) / len(n_ex_list)
    error_n_ex = scistats.sem(n_ex_list)

    print(f"time: avg: {avg_time}, error: {error_time}")
    print(f"n_nodes: avg: {avg_n_nodes}, error: {error_n_nodes}")
    print(f"n_ex: avg: {avg_n_ex}, error: {error_n_ex}")


def compute_val_ll(model, n_test_new: int, samples_val=None):
    if samples_val == None:
        samples_val = model.samples_val[:n_test_new]
    model.val_states = []
    for s in model.samples_val[:n_test_new]:
        states, rules = model.sample_to_statelist(s)
        for i in range(len(rules)):
            model.val_states.append(model.state_rule_to_sample(states[i], rules[i]))

    val_loss = eval_ll(model.val_states, model)
    return val_loss


def run_search(model, args):
    ex_queries = []
    if args.new_test_ep:
        print("generating new test examples")
        _, generate_episode_test, input_lang, output_lang, prog_lang = get_episode_generator(
            args.new_test_ep, model_in_lang=model.input_lang,
            model_out_lang=model.output_lang,
            model_prog_lang=model.prog_lang)
        # model.tabu_episodes = set([])
        # Rafal
        model.samples_val = []
        if args.hack_gt_g:
            for i in range(args.n_test):
                sample = generate_episode_test(model.tabu_episodes)
                if args.hack_gt_g:
                    sample['grammar'] = Grammar(exact_perm_doubled_rules(),
                                                model.input_lang.symbols)
                model.samples_val.append(sample)
                if not args.duplicate_test:
                    model.tabu_episodes = tabu_update(model.tabu_episodes, sample['identifier'])
        else:
            for i in range(args.n_test):
                sample = generate_episode_test(model.tabu_episodes)
                model.tabu_episodes = tabu_update(model.tabu_episodes, sample['identifier'])
                inputs_and_outputs = zip(sample['xs'], sample['ys'])
                examples = {Example(cur, tgt) for cur, tgt in inputs_and_outputs}
                inputs_and_outputs_query = zip(sample['xq'], sample['yq'])
                query_examples = {Example(cur, tgt) for cur, tgt in inputs_and_outputs_query}
                ex_queries.append((examples, query_examples))
                model.samples_val.append(sample)

        model.input_lang = input_lang
        model.output_lang = output_lang
        if not args.val_ll_only:
            model.prog_lang = prog_lang

    if args.val_ll:
        val_ll = compute_val_ll(model, n_test_new=args.n_test)
        print("val ll:", val_ll)

    if args.val_ll_only:
        assert False

    print(f"testing using {args.mode}")
    # print("batchsize:", args.batchsize)
    count = 0
    results = []
    frac_exs_hits = []

    tot_time = 0.
    tot_nodes = 0
    all_examples = set()

    for j, sample in enumerate(model.samples_val):
        print()
        print(f"Task {j + 1} out of {len(model.samples_val)}")
        # print("ground truth grammar:")
        # print(sample['identifier'])
        examples, query_examples = ex_queries[j]

        hit, solution, stats = batched_test_with_sampling(sample, model,
                                                          examples=examples,
                                                          query_examples=query_examples,
                                                          max_len=10,
                                                          timeout=args.timeout,
                                                          verbose=True,
                                                          min_len=0,
                                                          batch_size=args.batchsize,
                                                          nosearch=args.nosearch,
                                                          partial_credit=args.partial_credit,
                                                          max_rule_size=200)

        print(stats)
        result_path = f"{args.savefile}_{j + 1}"
        _write_solution_to_file(result_path, solution, stats)
        tot_time += time.time() - stats['start_time']
        tot_nodes += stats['nodes_expanded']
        for ex in sample['xs']:
            all_examples.add(tuple(ex))

        if hit:
            print('done with one of the runs')
            return tot_time, tot_nodes, len(all_examples)

    else:
        assert 0, "didn't hit after 20 examples, that's dumb!"


def _write_solution_to_file(result_file_path: str, solution: State, stats: dict):
    print("Writing lines")
    lines = ["".join(rule) + "\n" for rule in solution.rules]
    print(f"Lines: {lines}")
    with open(result_file_path, mode="w+") as file:
        file.write(str(stats))
        file.write("\n")
        file.writelines(lines)


def main(args=None):
    parsed_args = parse_args(args)
    _search(parsed_args)


if __name__ == '__main__':
    main()
