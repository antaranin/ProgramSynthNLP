import os
from typing import Callable, List, Tuple

from functools import partial
from context_matrix import create_and_save_context_matrix, load_context_matrix, \
    rewrite_context_matrix_to_concept_lib_format, \
    create_grammar_context_matrix_for_concept_lib_format
from main import process_data_file, write_alphabet
import pandas as pd
import operation_revisor as rev
import baseline


def _run_steps(directory_name: str, filename: str, generate_step_1: bool, generate_step_2: bool,
               generate_step_3: bool, generate_step_4: bool, generate_step_5: bool) -> None:
    file_data = filename.split("-")
    language = file_data[0]
    operation = file_data[1]
    if operation == "dev":
        process_data_file(f"{directory_name}/{filename}", language, generate_step_1,
                          generate_step_2, generate_step_3, generate_step_4, generate_step_5)


def _iterate_directory(directory_name, operation: Callable[[str, str], None]) -> None:
    for filename in os.listdir(directory_name):
        operation(directory_name, filename)


def _write_alphabet_for_file(directory_name: str, filename: str):
    language = filename.split(".")[0]
    print(f"Writing alphabet for {language}")
    write_alphabet(f"data/processed/alphabet/{filename}", f"{directory_name}/{filename}")
    print(f"Finished alphabet for {language}")


def write_alphabets():
    dir_name = "data/processed/first_step"
    _iterate_directory(dir_name, _write_alphabet_for_file)


def write_steps(generate_step_1=False, generate_step_2=False, generate_step_3=False,
                generate_step_4=False, generate_step_5=False):
    directory_name = "data/latin_alphabet"
    op = partial(_run_steps, generate_step_1=generate_step_1, generate_step_2=generate_step_2,
                 generate_step_3=generate_step_3, generate_step_4=generate_step_4,
                 generate_step_5=generate_step_5)
    _iterate_directory(directory_name, op)


def write_context_matrices():
    directory_name = "data/processed"
    for filename in os.listdir(f"{directory_name}/first_step"):
        file_data = filename.split(r".")
        language = file_data[0]
        print(f"Starting work on {language}")
        first_step_file = f"{directory_name}/first_step_revised/{filename}"
        second_step_file = f"{directory_name}/second_step_revised/{filename}"
        output_path = f"{directory_name}/context_matrix/{filename}"
        create_and_save_context_matrix(output_path, first_step_file, second_step_file)

        print(f"Finished work on {language}")



def write_first_second_step_revision():
    directory_name = "data/processed"
    for filename in os.listdir(f"{directory_name}/subword"):
        language = filename.split(r".")[0]
        print(f"Starting work on {language}")
        subword_file = f"{directory_name}/subword/{filename}"
        first_step_file = f"{directory_name}/first_step/{filename}"
        second_step_file = f"{directory_name}/second_step/{filename}"
        revised_first_step_file = f"{directory_name}/first_step_revised/{filename}"
        revised_second_step_file = f"{directory_name}/second_step_revised/{filename}"
        rev.revise_steps(subword_file, first_step_file, second_step_file, revised_first_step_file,
                         revised_second_step_file)
        print(f"Finished work on {language}")




def rewrite_context_matrices():
    directory_name = "data/processed"
    for filename in os.listdir(f"{directory_name}/context_matrix"):
        language = filename.split(r".")[0]
        print(f"Starting work on {language}")
        context_matrix_file = f"{directory_name}/context_matrix/{filename}"
        output_file = f"{directory_name}/context_matrix_revised/{filename}"
        rewrite_context_matrix_to_concept_lib_format(context_matrix_file, output_file, 1)
        print(f"Finished work on {language}")


def create_grammar_context_matrices():
    directory_name = "data/processed"
    # context_matrix_file = f"{directory_name}/context_matrix_revised/danish.csv"
    # output_file = f"{directory_name}/grammar_context_matrix/danish.csv"
    # create_grammar_context_matrix_for_concept_lib_format(context_matrix_file, output_file)
    for filename in os.listdir(f"{directory_name}/context_matrix_revised"):
        language = filename.split(r".")[0]
        print(f"Starting work on {language}")
        context_matrix_file = f"{directory_name}/context_matrix_revised/{filename}"
        output_file = f"{directory_name}/grammar_context_matrix/{filename}"
        create_grammar_context_matrix_for_concept_lib_format(context_matrix_file, output_file)
        print(f"Finished work on {language}")



def reformat_sigmorphon_predictions():
    directory_name = "data/baseline_res"
    for filename in os.listdir(f"{directory_name}"):
        language, type, _ = filename.split(r"-")
        if type != "high":
            continue
        print(f"Starting work on {language}")
        sig_file = f"{directory_name}/{filename}"
        data_file = f"data/latin_alphabet/{language}-test"
        out_file = f"data/processed/predictions/sigmorphon/{language}.csv"
        baseline.format_and_save_sigmorphon_predictions(sig_file, data_file, out_file)
        print(f"Finished work on {language}")


def write_sigmorphon_baseline_cost():
    _write_baseline_cost("sigmorphon", "sigmorphon_cost")


def _write_baseline_cost(input_file_dir: str, output_file_dir: str):
    directory_name = "data/processed/predictions"
    for filename in os.listdir(f"{directory_name}/{input_file_dir}"):
        language = filename.split(r".")[0]
        print(f"Starting work on {language}")
        word_and_prediction_file = f"{directory_name}/{input_file_dir}/{filename}"
        output_file = f"{directory_name}/{output_file_dir}/{filename}"
        baseline.calculate_and_save_cost_baseline(word_and_prediction_file, output_file)
        print(f"Finished work on {language}")


def write_baseline_cost():
    _write_baseline_cost("base", "base_cost")


def _get_mean_and_standard_devs_for_languages() -> List[Tuple[str, float, float]]:
    directory_name = "data/latin_alphabet"
    res = []
    for filename in os.listdir(f"{directory_name}"):
        sp = filename.split(r"-")
        language = sp[0]
        type = sp[1]
        if type != "test":
            continue
        data_file = f"{directory_name}/{filename}"
        mean, stdev = baseline.get_means_and_stdev_for_language(data_file)
        res.append((language, mean, stdev))
    return res


def _get_average_baseline_cost(cost_dir: str) -> List[Tuple[str, float]]:
    directory_name = f"data/processed/predictions/{cost_dir}"
    language_cost = []
    for filename in os.listdir(f"{directory_name}"):
        language = filename.split(r".")[0]
        cost_file = f"{directory_name}/{filename}"
        cost = baseline.calculate_average_cost(cost_file)
        language_cost.append((language, cost))
    return language_cost


def get_average_baseline_cost():
    _get_average_baseline_cost("base_cost")


def get_average_sigmorphon_cost():
    _get_average_baseline_cost("sigmorphon_cost")


def compare_base_to_sigmorphon():
    base = _get_average_baseline_cost("base_cost")
    sig = _get_average_baseline_cost("sigmorphon_cost")
    means_stdev = _get_mean_and_standard_devs_for_languages()
    zipped = zip(base, sig, means_stdev)
    for item in zipped:
        print(f"Language: {item[0][0]}")
        print(f"Base: {item[0][1]} - Sig: {item[1][1]} ")
        print(f"Mean: {item[2][1]}, Stdev: {item[2][2]}")





# write_steps(True, True, True, True, True)
# write_alphabets()
# write_first_second_step_revision()
# write_context_matrices()
# write_concepts()
# draw_trees()
# _test_memorizing_lattice()
# _test_decision_tree()
# predict_words()
# write_baseline_cost()
# predict_words()
# write_baseline_cost()
# reformat_sigmorphon_predictions()
# write_sigmorphon_baseline_cost()
# get_average_sigmorphon_cost()
# compare_base_to_sigmorphon()
# write_steps(True, True, True, True, True)
# write_first_second_step_revision()
# write_context_matrices()
# write_concepts()
# predict_words()
# write_baseline_cost()
# compare_base_to_sigmorphon()
# draw_trees()
create_grammar_context_matrices()
