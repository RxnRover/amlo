from optimizer import optimizer
import pandas as pd
import argparse


def main():

    args = parse_args()

    training_dataset_path = args.training_file
    training_combo_path = args.training_combo_file
    parameters = args.parameters
    yield_val = float(args.yield_val)
    itr = args.iteration

    parameters = [float(x) for x in parameters]

    next_parameters = generate_training_data(
        training_dataset_path, training_combo_path, parameters, yield_val, itr)

    print('Next experimental parameters for training set', next_parameters)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument("training_file", help="Training dataset file path")
    parser.add_argument("training_combo_file", help="Training combo file path")
    parser.add_argument(
        "parameters", help="List of previous or initial parameters")
    parser.add_argument(
        "yield_val", help="Previous experimental value or initial zero")
    parser.add_argument(
        "iteration", help="Experiment step number, starting with zero", type=int)

    return parser.parse_args()


def generate_training_data(training_dataset_path, training_combo_path, parameters=[], yield_val=0, itr=0):
    """Generating traning dataset for the ML model. This function is reading reaction parameters from 
    tranining combination file and return next  reaction parameter combination. After the first experiment, 
    previous parameter set and experimental yield are writting to the traning set file.

    :param training_dataset_path: Path to the traning dataset file.
    :type training_dataset_path: File
    :param training_combo_path: path to the traning combination file.
    :type training_combo_path: File
    :param parameters: parameter set from previous experiment or initial parameters, defaults to [].
    :type parameters: List, optional
    :param yield_val: experimental yield from previous experiment, defaults to 0.
    :type yield_val: int, optional
    :param itr: Experiment iteration, starting from 0, defaults to 0.
    :type itr: int, optional
    :return: parameter set for next experiment.
    :rtype: List
    """
    prev_parameters = ','.join([str(elem)
                               for elem in parameters]) + ',' + str(yield_val)

    if(len(parameters) != 0):
        optimizer.write_data_to_training(
            training_dataset_path, prev_parameters)
        print('writting')
    data = load_training_combo_file(training_combo_path)

    # print(data[itr])
    return data[itr]


def load_training_combo_file(training_combo):
    """loading the training combination file as pandas data frame and return combination data as list.

    :param training_combo: training parameter combination file path
    :type training_combo: File
    :return: training parameter combination list
    :rtype: List
    """
    data = pd.read_csv(training_combo, skiprows=1)
    return data.values.tolist()


if __name__ == "__main__":
    main()
