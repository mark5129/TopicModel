# Import libraries
import yaml
import pandas as pd
import os

# Import custom functions
from functions.preprocessing import *
from functions.evaluating import *
from functions.modelling import *
from functions.plotting import *
from functions.log import *

# Load parameters from YAML file
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Save the parameters to a CSV file
current_id, time = log_parameters(parameters)

# Preprocessing
file_names = parameters['filenames']
languages = parameters['languages']
sources = parameters['source_names']

for i in range(len(file_names)):

    file_name = file_names[i]
    language = languages[i]
    source = sources[i]

    # Check if the file exists
    if os.path.exists(f'data/{file_name}'):
        print(f"{file_name} exists.")
    else:
        print(f"{file_name} does not exist.")
    
    text_df = pd.read_csv(f'data/{file_name}')

    # Check if the file contains column "Content"
    if 'Content' in text_df.columns:
        print(f"{file_name} contains the column 'Content'.")
    else:
        print(f"{file_name} does not contain the column 'Content'.")

    # Cleans the text_df
    text_df = clean_rows(text_df) # Clean the rows of the dataframe

    if language == 'da':
        text_df['Content'] = text_df['Content'].apply(lambda x: asyncio.run(translate_text(x)))
    
    # Remove stopwords from the text
    text_df['Content'] = text_df['Content'].apply(lambda x: remove_stopwords(x, 'english'))

    # Stem the text
    text_df['Content'] = text_df['Content'].apply(lambda x: stemming(x, 'english'))

    # Save the DataFrame to a CSV file
    text_df.to_csv(f'outputs/preprocessing/{current_id}_{source}_preprocessed.csv', index=False)
    print(f'{source} preprocessed and saved.')

# Merge all preprocessed files
preprocessed_files = []
for source in sources:
    df = pd.read_csv(f'outputs/preprocessing/{current_id}_{source}_preprocessed.csv')
    preprocessed_files.append(df)

merged_df = pd.concat(preprocessed_files, ignore_index=True)
merged_df.to_csv(f'outputs/preprocessing/{current_id}_merged_preprocessed.csv', index=False)
print('Preprocessed files merged and saved.')

# Modelling
for source in sources:

    text_df = pd.read_csv(f'outputs/preprocessing/{current_id}_{source}_preprocessed.csv')

    text_column = text_df['Content']

    MiniLm12_embeddings = MiniLM12(text_column, current_id, source)
    print(f'{source} MiniLM12 embeddings generated')

    XLM_Roberta_embeddings = XLM_Roberta_model(text_column, current_id, source)
    print(f'{source} XLM_Roberta embeddings generated')

    Specter2_embeddings = Specter2model(text_column, current_id, source)
    print(f'{source} Specter2 embeddings generated')

#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2', 'MiniLm12']

embeddings = []
for model in which_model:
    for source in sources:

        df = pd.read_csv(f'outputs/modelling/{current_id}_{source}_{model}_embeddings.csv')

        embeddings.append(df)

    merged_df = pd.concat(embeddings)

    # Save merged embeddings
    merged_output_path = f'outputs/modelling/{current_id}_merged_{model}_embeddings.csv'
    merged_df.to_csv(merged_output_path, index=False)
print('Embeddings are merged')

# Clustering and naming
# Read documents used for naming of clusters
df = pd.read_csv(f'outputs/preprocessing/{current_id}_merged_preprocessed.csv')

for model in which_model:
    # Load the merged embeddings for each model distinctly to perform clustering
    embeddings = pd.read_csv(f'outputs/modelling/{current_id}_merged_{model}_embeddings.csv')
    TFIDF_clustering(embeddings, df, current_id, 'merged', model)

print('Clustering is done')

for model in which_model:
    embeddings = pd.read_csv(f'outputs/modelling/{current_id}_merged_{model}_embeddings.csv')
    kmeans = pd.read_csv(f'outputs/modelling/{current_id}_merged_{model}_Kmeans.csv')

    # data_mapplot_with_naming(kmeans, embeddings, df, current_id, 'merged', model)
    # topic_source_plot(kmeans, current_id, 'merged', model)
    # create_bokeh_plot(kmeans, embeddings, df, current_id, 'merged', model)
    # outline_plot(kmeans, embeddings, df, current_id, 'merged', model)
    # cluster_plot(kmeans, embeddings, df, current_id, 'merged', model)