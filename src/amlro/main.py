from optimizer import optimizer
def main(training_dataset_path,full_combo_path,parameters=[], yield_val=0):
    prev_parameters = '2022/04/23 11:46:00'+','+str(yield_val)+','+','.join([str(elem) for elem in parameters])
    #prev_parameters = prev_parameters+','+str(yield_val)
    if(len(parameters) != 0):
        optimizer.write_data_to_training('../../training.csv.txt',prev_parameters)
        print('writting')
    ####################################################
    x_train,y_train,data = optimizer.load_data('../../training.csv.txt','../../combinations.csv.txt')
    print('Data Loading for Machine Learning Model...')
    print(data)
    print('Training ML model...')
    regr = optimizer.model_training(x_train,y_train)
    best_combo,data = optimizer.predict_next_parameters(regr,data)
    print('Searching for best reaction parameters...')
    print('Best parameter combination...',best_combo[:1].values.tolist())
    #####################################################
    print('Sending optmized parameters to Rxn Rover...')
    
    return best_combo[:1].values.tolist()

if __name__ == "__main__":
    main('../../training.csv.txt','../../training.csv.txt', [0.3,0.175,0.125,24.0],3.813667)
