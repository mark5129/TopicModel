# Import libraries
import pandas as pd
import yaml
from nltk.stem.snowball import DanishStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import asyncio
from googletrans import Translator
import pandas as pd

# Load parameters from YAML file
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Define functions
def clean_rows(df):
    
    # remove all rows where the string in the content column contains more than 15000 letters
    # This is the limit for the translator in googletrans
    df = df[df['Content'].apply(lambda x: len(x) < 15000)]
    
    return df

async def translate_text(text: str, language: str):
    translator = Translator()
    
    translated = await translator.translate(text, src=language, dest='en')
    
    return str(translated.text).strip()

def remove_stopwords(text: str,language: str):
    """
    Removes stopwords from a text.

    Parameters:
    text: The text to remove stopwords from.
    language (str): What language to use.

    Returns:
    text (str): The text with stopwords removed.
    """


    if language == 'danish':
        stop_words = set(stopwords.words('danish'))
    elif language == 'english':
        stop_words = set(stopwords.words('english'))
    else:
        raise ValueError('Language not supported')
    
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def stemming(text: str, language: str):
    """
    Stems words in a text.

    Parameters:
    text: The text to stem.
    language (str): What language to use.

    Returns:
    text (str): The text with words stemmed.
    """

    if language == 'danish':
        stemmer = DanishStemmer()
    elif language == 'english':
        stemmer = PorterStemmer()
    else:
        raise ValueError('Language not supported')

    word_tokens = text.split()
    stemmed_text = [stemmer.stem(word) for word in word_tokens]
    return ' '.join(stemmed_text)