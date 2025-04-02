# Import libraries
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import yaml

# Load parameters from YAML file
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)


# Define functions

# Save the topics to a CSV file
output_dir = 'outputs/modelling/'