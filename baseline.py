import csv
from statistics import mean, stdev
from typing import List, Tuple
import editdistance


def _read_predictions_and_actual_words(prediction_file: str) -> List[Tuple[str, str, str]]:
    res = []
    with open(prediction_file) as file:
        r = csv.reader(file, delimiter=";")
        next(r)
        for row in r:
            res.append((row[0], row[1], row[2]))
    return res

def _read_lemma_and_inflection(lemma_inf_file: str, delimiter="\t") -> List[Tuple[str, str]]:
    res = []
    with open(lemma_inf_file) as file:
        r = csv.reader(file, delimiter=delimiter)
        for row in r:
            res.append((row[0], row[1]))
    return res


def calculate_and_save_cost_baseline(prediction_file: str, output_file: str):
    predictions_and_actual = _read_predictions_and_actual_words(prediction_file)
    words_and_cost = []
    for original, prediction, actual in predictions_and_actual:
        score = editdistance.eval(prediction, actual)
        words_and_cost.append((original, score))
    _save_cost_baseline(words_and_cost, output_file)


def format_and_save_sigmorphon_predictions(sigmorphon_file: str, data_file: str, output_file: str):
    sigmorphon_data = _read_lemma_and_inflection(sigmorphon_file)
    actual_data = _read_lemma_and_inflection(data_file)
    assert len(sigmorphon_data) == len(actual_data)
    predictions = []
    for i in range(len(sigmorphon_data)):
        predictions.append((sigmorphon_data[i][0], sigmorphon_data[i][1], actual_data[i][1]))
    _save_predictions(predictions, output_file)


def _save_predictions(predictions: List[Tuple[str,str,str]], output_file: str):
    with open(output_file, mode='w+') as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Word", "Prediction", "Actual"])
        for prediction in predictions:
            writer.writerow(prediction)




def calculate_average_cost(word_cost_file: str) -> float:
    res = []
    with open(word_cost_file) as file:
        r = csv.reader(file, delimiter=";")
        next(r)
        for row in r:
            res.append(float(row[1]))
    return mean(res)


def _save_cost_baseline(words_and_cost: List[Tuple[str, str]], output_file: str):
    with open(output_file, mode='w+') as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Word", "Cost"])
        for word_and_cost in words_and_cost:
            writer.writerow(word_and_cost)

def get_means_and_stdev_for_language(data_file: str) -> Tuple[float, float]:
    lemma_inflections = _read_lemma_and_inflection(data_file)
    inflection_lengths = [len(lemma_inf[1]) for lemma_inf in lemma_inflections]
    std = stdev(inflection_lengths)
    m = mean(inflection_lengths)
    return m, std
