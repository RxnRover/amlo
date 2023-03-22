from amlro.optimizer import optimizer
import argparse
from typing import List, Dict


def main():
    args = parse_args()

    training_dataset_path = args.training_file
    full_combo_path = args.full_combo_file
    parameters = args.parameters
    yield_val = float(args.yield_val)

    parameters = [float(x) for x in parameters]

    best_combo = get_optimized_parameters(training_dataset_path,
                                          full_combo_path, parameters,
                                          yield_val)
    print('Optimized parameters ', best_combo)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument("training_file", help="Training dataset file path")
    parser.add_argument("full_combo_file", help="Full combination file path")
    parser.add_argument("parameters",
                        help="List of previous or initial parameters")
    parser.add_argument("yield_val",
                        help="Previous experimental value or initial zero")

    return parser.parse_args()


def get_optimized_parameters(training_dataset_path: str,
                             full_combo_path: str,
                             parameters=[],
                             yield_val=0) -> List[float]:
    """getting the best parameter set from the maching learning optimizer. 
    Initialy experimental yield and previous parameter set should be empty.

    :param training_dataset_path: file path to the training set file.
    :type training_dataset_path: str
    :param full_combo_path: path to the combination file.
    :type full_combo_path: str
    :param parameters:  previous best parameter set, defaults to [].
    :type parameters: List[float], optional
    :param yield_val: previous experimental yield, defaults to 0.
    :type yield_val: int, optional
    :return: best parameter combination for next experiment
    :rtype: List[float]
    """
    prev_parameters = ','.join([str(elem)
                                for elem in parameters]) + ',' + str(yield_val)

    if (len(parameters) != 0):
        optimizer.write_data_to_training(training_dataset_path,
                                         prev_parameters)
        print('writting')
    ####################################################
    x_train, y_train, data = optimizer.load_data(training_dataset_path,
                                                 full_combo_path)
    print('Data Loading for Machine Learning Model...')
    

    print('Training ML model...')
    regr = optimizer.model_training(x_train, y_train)

    best_combo = optimizer.predict_next_parameters(regr, data)
    print('Searching for best reaction parameters...')
    print('Best parameter combination...', best_combo[:1].values.tolist())
    #####################################################

    print('Sending optmized parameters to Rxn Rover...')

    return best_combo[:1].values.tolist()


if __name__ == "__main__":
    main()
