from optimizer import optimizer


def main():
    training_dataset_path = '../../training.csv.txt'
    full_combo_path = '../../training.csv.txt'
    parameters = []
    yield_val = 0
    best_combo = get_optimized_parameters(
        training_dataset_path, full_combo_path, parameters, yield_val)
    print('Optimized parameters ', best_combo)


def get_optimized_parameters(training_dataset_path, full_combo_path, parameters=[], yield_val=0):
    """getting the best parameter set from the maching learning optimizer. 
    Initialy experimental yield and previous parameter set should be empty.

    Args:
        training_dataset_path (file): file path to the training set file
        full_combo_path ( file): file path to the combination file.
        parameters (list, optional): previous best parameter set. Defaults to [].
        yield_val (int, optional): previous experimental yield_. Defaults to 0.

    Returns:
        List : best parameter combination for next experiment
    """
    prev_parameters = '2022/04/23 11:46:00' + ',' + \
        str(yield_val) + ',' + ','.join([str(elem) for elem in parameters])
    #prev_parameters = prev_parameters+','+str(yield_val)
    if(len(parameters) != 0):
        optimizer.write_data_to_training(
            training_dataset_path, prev_parameters)
        print('writting')
    ####################################################
    x_train, y_train, data = optimizer.load_data(
        training_dataset_path,  full_combo_path)
    print('Data Loading for Machine Learning Model...')
    print(data)
    print('Training ML model...')
    regr = optimizer.model_training(x_train, y_train)
    best_combo, data = optimizer.predict_next_parameters(regr, data)
    print('Searching for best reaction parameters...')
    print('Best parameter combination...', best_combo[:1].values.tolist())
    #####################################################
    print('Sending optmized parameters to Rxn Rover...')

    return best_combo[:1].values.tolist()


if __name__ == "__main__":
    main()
