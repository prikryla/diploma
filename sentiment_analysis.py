import pandas as pd
from textblob import TextBlob
# Load the dataset
df = pd.read_csv('test_bez_upravy.csv', delimiter=';')

# Define a function to get the polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Define a function to get the subjectivity
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Apply the functions to the description column
df['polarity'] = df['description'].apply(get_polarity)
df['subjectivity'] = df['description'].apply(get_subjectivity)

df.to_csv('enriched.csv', index=False)
