# WARNING: This file is for development and testing purposes only.
# It is NOT intended for use in production environments.
# The script cleans a CSV file to the proper format.

import pandas as pd
import unicodedata
import re

def clean_text(text):
    # Remove non-UTF-8 characters and normalize
    cleaned_text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-printable characters
    cleaned_text = re.sub(r'[^\x20-\x7E]', '', cleaned_text)
    # Remove extra spaces before and after
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def clean_description(text):
    # Check if the value is NaN
    if pd.isna(text):
        return ''
    # Remove quotes at the beginning and end, then apply general cleaning
    return clean_text(str(text).strip('"'))

def clean_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, encoding='utf-8', on_bad_lines='skip')
    
    # Apply cleaning function to all string columns
    for column in df.select_dtypes(include=['object']):
        if column == 'Description':
            df[column] = df[column].apply(clean_description)
        else:
            df[column] = df[column].apply(clean_text)
    
    # Remove rows with empty or NaN 'Description' or 'Title' columns
    df = df.dropna(subset=['Description', 'Book'])
    df = df[df['Description'] != '']
    df = df[df['Book'] != '']
    
    # Write the cleaned data to a new CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')

def main():
    input_file = 'goodreads_data.csv'
    output_file = 'cleaned_' + input_file
    
    clean_csv(input_file, output_file)
    print(f"Cleaned data has been saved to {output_file}")

if __name__ == "__main__":
    main()