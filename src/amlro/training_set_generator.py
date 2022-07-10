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

    Args:
        training_dataset_path (_type_): Path to the traning dataset file.
        training_combo_path (_type_): path to the traning combination file.
        parameters (list, optional): parameter set from previous experiment or initial parameters.
        yield_val (int, optional): experimental yield from previous experiment. Defaults to 0.
        itr (int, optional): Experiment iteration, starting from 0 . Defaults to 0.

    Returns:
        parameters (list) : parameter set for next experiment.
    """
    prev_parameters =  ','.join([str(elem) for elem in parameters]) + ',' + str(yield_val)

    if(len(parameters) != 0):
        optimizer.write_data_to_training(
            training_dataset_path, prev_parameters)
        print('writting')
    data = load_training_combo_file(training_combo_path)
    
    #print(data[itr])
    return data[itr]


def load_training_combo_file(training_combo):
    """ loading the training combination file as pandas data frame and return combination data as list.

    Args:
        training_combo (string): training parameter combination file path

    Returns:
       training combos (list): training parameter combination list
    """
    data = pd.read_csv(training_combo, skiprows=1)
    return data.values.tolist()


if __name__ == "__main__":
    main()
