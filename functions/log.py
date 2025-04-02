import csv
import time
import os
import random

def log_parameters(parameters: dict):

    # Check if log.csv exists and get the last ID
    if os.path.exists('modelling/outputs/log.csv'):
        with open('modelling/outputs/log.csv', mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) > 1:
                last_id = int(rows[-1][0])
            else:
                last_id = 0
    else:
        last_id = 0
    
    # generate random number with 10 digits
    current_id = random.randint(1000000000, 9999999999)

    # Increment the ID for the current run
    #current_id = last_id + 1

    with open('outputs/log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is empty
        if last_id == 0:
            writer.writerow([
                'id', 
                'time', 
                'preprocess_data', 
                'train_model', 
                'num_topics', 
                'what_data', 
                'nmf_random_state', 
                'nmf_max_features', 
                'nmf_n_top_words', 
                'train_specter2', 
                'train_lda',
                'train_nmf'
            ])
            
        # Write the current ID, time, and parameters
        writer.writerow([
            current_id, 
            time.ctime(), 
            parameters['preprocess_data'], 
            parameters['train_model'], 
            parameters['num_topics'], 
            parameters['what_data'], 
            parameters['random_state'], 
            parameters['max_features'], 
            parameters['n_top_words'], 
            parameters['train_specter2'], 
            parameters['train_lda'],
            parameters['train_nmf']
        ])
        print('Parameters are saved to log.csv')
    
    return current_id, time.ctime()