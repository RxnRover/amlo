from optimizer import optimizer
import pandas as pd
def main(training_dataset_path,training_combo_path,parameters=[], yield_val=0,itr=0):
    prev_parameters = '2022/04/23 11:46:00'+','+str(yield_val)+','+','.join([str(elem) for elem in parameters])
    
    if(len(parameters) != 0):
        optimizer.write_data_to_training('../../training.csv.txt',prev_parameters)
        print('writting')
    data = load_training_combo_file(training_combo_path)
    print(data[itr])
    return data[itr]
def load_training_combo_file(training_combo):
    data = pd.read_csv(training_combo,skiprows=1)
    return data.values.tolist()

if __name__ == "__main__":
    main('../../training.csv.txt','../../training.csv.txt',itr=-1)