import pandas as pd
import requests
import time

def translate_text(text, source_lang='en', target_lang='cz'):
    url = "https://libretranslate.com/translate"
    payload = {
        'q': text,
        'source': source_lang,
        'target': target_lang,
        'format': 'text'
    }
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    try:
        response = requests.post(url, data=payload, headers=headers)
        return response.json().get('translatedText', text)  # Return original text if translation fails
    except requests.exceptions.RequestException as e:  # Catch network errors
        print(f"Error translating text: {e}")
        return text

# Load the CSV file
df = pd.read_csv('test_bez_upravy.csv', delimiter=';')

# Translate the selected column with a pause between requests to avoid API overload
for i, row in df.iterrows():
    original_text = "Hello world"
    translated_text = translate_text(original_text)
    df.at[i, 'description'] = translated_text
    print(f"Original: {original_text} -> Translated: {translated_text}")  # Debugging output
    time.sleep(1)  # Pause for 1 second between requests

# Save the translated dataset with the same delimiter as the input
df.to_csv('translated_dataset.csv', index=False, sep=';')
