from optimizer import optimizer
import pandas as pd


def main():
    training_dataset_path = '../../training.csv.txt'
    training_combo_path = '../../training.csv.txt'
    parameters = []
    yield_val = 0
    itr = 0

    next_parameters = generate_training_data(
        training_dataset_path, training_combo_path, parameters, yield_val, itr)
    print('Next experimental parameters for training set', next_parameters)


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
