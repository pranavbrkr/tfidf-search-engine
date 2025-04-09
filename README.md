# Project-1-Group-18

# Accelerating Search with a TF-IDF Server and DynamoDB Storage Backend

This project uses the Term Frequency-Inverse Document Frequency (TF-IDF) model in conjunction with AWS DynamoDB to create a scalable and effective search solution. It offers a Flask-built RESTful API that lets users input data, setup databases, and do relevance-based searches.

## Group Members
- Harry DeCecco
- Nicholas Toland
- Pranav Borikar
- Race Musgrave
- Sahil Garg
- Vedant Yatin Pimpley

## Features

- **Database Initialization:** Upload a CSV file to populate the DynamoDB table and build a TF-IDF index.
- **Incremental Insertion:** Add additional data to the database while maintaining the TF-IDF index.
- **Relevance-Based Search:** Search for books using keywords, ranked by their TF-IDF score.
- **Dynamic TF-IDF Initialization:** Automatically rebuild the TF-IDF index if the server is restarted.

## Prerequisites
1. Python 3.7+
2. Libraries
   - flask
   - boto3
   - pandas
   - scikit-learn
3. AWS configuration:
   Configure your aws credentials using
   ```aws configure```


# Download CSV File
cleaned_goodreads_data.csv
# How to run API
```python api.py```

# Note:
The file api.py, data_cleaning.py, and tfidf_index.py are for development and testing purposes only.
They are NOT intended for use in production environments.
Ensure use of the proper python file, api-3.py