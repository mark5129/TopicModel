# Import libraries
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import yaml

# Load parameters from YAML file
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Define functions
def MiniLM12(text_data, current_id, doc_type):
    
    # English model
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    #Generate embeddings
    embeddings = model.encode(text_data, show_progress_bar=True, batch_size=32)

    # Save embeddings to CSV
    output_dir = 'outputs/modelling/'

    # Convert the embeddings to a DataFrame
    df_embeddings = pd.DataFrame(embeddings)

    # Save the DataFrame to a CSV file
    df_embeddings.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_MiniLm12_embeddings.csv'), index=False)
    print(f'MiniLm12 embeddings saved for ID {current_id}')

    return df_embeddings

def Specter2model(text_column, current_id, doc_type):

    # Load the Specter2 model
    model = SentenceTransformer("allenai/specter2_base")

    # Generate embeddings
    embeddings_list = model.encode(text_column.tolist(), show_progress_bar=True)

    # Save embeddings to CSV
    output_dir = 'outputs/modelling/'

    # Convert the embeddings to a DataFrame
    df_embeddings = pd.DataFrame(embeddings_list)

    # Save the DataFrame to a CSV file
    df_embeddings.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_Specter2_embeddings.csv'), index=False)
    print(f'Specter2 embeddings saved for ID {current_id}')

    return df_embeddings

def XLM_Roberta_model(text_column, current_id, doc_type):

    # Load the XLM-Roberta model
    model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    # Generate embeddings
    embeddings = model.encode(text_column, show_progress_bar=True, batch_size=32)

    # Save the topics to a CSV file
    output_dir = 'outputs/modelling/'

    # Convert the embeddings to a DataFrame
    embeddings_df = pd.DataFrame(embeddings)

    # Save the DataFrame to a CSV file
    embeddings_df.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_XLM_Roberta_embeddings.csv'), index=False)
    print('XLM-Roberta Topics saved')


    return embeddings_df