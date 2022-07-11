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

    :param training_dataset_path: file path to the training set file.
    :type training_dataset_path: File
    :param full_combo_path: path to the combination file.
    :type full_combo_path: File
    :param parameters:  previous best parameter set, defaults to [].
    :type parameters: list, optional
    :param yield_val: previous experimental yield, defaults to 0.
    :type yield_val: int, optional
    :return: best parameter combination for next experiment
    :rtype: List
    """
    prev_parameters = ','.join([str(elem)
                               for elem in parameters]) + ',' + str(yield_val)

    if(len(parameters) != 0):
        optimizer.write_data_to_training(
            training_dataset_path, prev_parameters)
        print('writting')
    ####################################################
    x_train, y_train, data = optimizer.load_data(
        training_dataset_path, full_combo_path)
    print('Data Loading for Machine Learning Model...')
    # print(data)

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
