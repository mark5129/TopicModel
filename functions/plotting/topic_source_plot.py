import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def topic_source_plot(data, current_id, doc_type, model_name, palette):

    # Count the instances
    count_data = data.groupby(['main_topic_name', 'Source']).size().reset_index(name='Count')

    # Calculate the proportion
    source_counts = data['Source'].value_counts().to_dict()
    count_data['Proportion'] = count_data.apply(lambda row: row['Count'] / source_counts[row['Source']], axis=1)

    # Sort the data by topic names
    count_data = count_data.sort_values(by='main_topic_name')

    # Plot the data
    plt.figure(figsize=(12, 8))
    sns.barplot(x='main_topic_name', y='Proportion', hue='Source', data=count_data, palette=palette)
    plt.ylabel('Proportion')
    plt.title(f'Proportion of Topic Names by Source ({model_name})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to make room for the rotated x labels
    
    # Save Visualization to CSV
    output_dir = 'outputs/modelling/'

    plt.savefig(f"{output_dir}{current_id}_{doc_type}_{model_name}_sourceplot.png")
    print(f"{model_name} with {doc_type}: Source plot saved successfully for {current_id}")


