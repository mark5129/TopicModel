# Import libraries
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import yaml

# Load parameters from YAML file
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Define functions
def TFIDF_cluster_topic(cluster_texts, language='english', n_terms=5):
    # Define stop words for different languages
    stop_words = {
        'english': stopwords.words('english')
    }
    cluster_texts = [text for text in cluster_texts if isinstance(text, str) and text.strip()]
    if not cluster_texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=stop_words[language],
        max_features=1000,
        ngram_range=(2, 3)  # Use bigrams and trigrams
    )
    try:
        X = vectorizer.fit_transform(cluster_texts)
        if X.shape[1] == 0:
            return []

        tf_idf_sum = X.sum(axis=0).A1  # Sum TF-IDF scores across all documents
        terms = vectorizer.get_feature_names_out()

        top_indices = tf_idf_sum.argsort()[::-1]
        top_terms = [terms[i] for i in top_indices]

        return top_terms
    except Exception as e:
        print(f"Error in get_cluster_topic: {e}")
    return []

def TFIDF_clustering(embeddings, df, current_id, doc_type, model_name):

    # Ensure embeddings are in NumPy array format
    embeddings = np.array(embeddings)

    # Number of samples
    n_samples = embeddings.shape[0]

    # Adjust perplexity based on the number of samples
    perplexity = min(30, (n_samples - 1) // 3)

    # Step 1: Create a data map using t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    data_map = tsne.fit_transform(embeddings)

    # Step 2: Perform hierarchical clustering
    n_clusters_list = [parameters['num_topics']]  # Adjust these numbers for your desired hierarchy levels
    labels_layers = []
    topics_names = pd.DataFrame(columns=['topic_int', 'topic_names', 'labels_layer'])

    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=parameters['random_state'])
        labels_int = kmeans.fit_predict(data_map)

        used_topics = set()

        # Generate topic names for each cluster
        label_topic_map = {}
        for label in range(n_clusters):
            indices = np.where(labels_int == label)[0]
            if len(indices) == 0:
                label_topic_map[label] = f"{label}: No data"
                continue
            cluster_texts = df['Content'].iloc[indices].astype(str).tolist()
            top_terms = TFIDF_cluster_topic(cluster_texts, language='english', n_terms=5)

            # Select the first unused term as the topic name
            topic_name = None
            for term in top_terms:
                if term not in used_topics:
                    topic_name = term
                    used_topics.add(term)
                    break

            if topic_name is None:
                # All terms have been used; default to the highest scoring term with cluster label
                topic_name = f"{top_terms[0]} {label}" if top_terms else f"Cluster {label}"

            label_topic_map[label] = f"{label}: {topic_name}"

            # Save topic names for each cluster
            topics_names.loc[label, 'topic_int'] = label
            
            topics_names.loc[label, 'topic_names'] = topic_name

            topics_names.loc[label, 'labels_layer'] = f'{label}: {topic_name}'


    # Save embeddings to CSV
    output_dir = 'outputs/evaluations/'

    df_labels_int = pd.DataFrame(labels_int)
    df_labels_int.columns = ['topic_int']

    df_labels_int = df_labels_int.merge(topics_names, on='topic_int', how='left')

    sources = df['Source']

    df_labels_int['Source'] = sources
    
    df_labels_int.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_{model_name}_Kmeans.csv'), index=False)
    print(f'{model_name} k-means klustering and naming saved for ID {current_id}')

    return labels_layers