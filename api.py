from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import boto3
import csv
import os
from botocore.exceptions import ClientError
import time
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

dynamodb = boto3.resource('dynamodb', region_name='us-west-1')
table_name = "Books"

def create_db_table():
    try:
        table = dynamodb.Table(table_name)
        table.load()  # Check if the table exists
        print(f"Table '{table_name}' already exists. Deleting...")
        table.delete()
        while True:
            try:
                table.load()
                print("Waiting for table deletion...")
                time.sleep(5)
            except ClientError:
                print(f"Table '{table_name}' deleted.")
                break
    except ClientError:
        print(f"Table '{table_name}' does not exist. Creating...")

    # Create the table
    params = {
        "TableName": table_name,
        "KeySchema": [
            {"AttributeName": "BookID", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "BookID", "AttributeType": "S"}],
        "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
    }

    table = dynamodb.create_table(**params)
    print(f"Creating table '{table_name}'...")
    table.wait_until_exists()
    print(f"Table '{table_name}' created successfully.")
    return table

# Helper functions for TF-IDF
tfidf_dict = None
tfidf_vectorizer = None
tfidf_matrix = None

def perform_tfidf_indexing(path, key_col_name, desc_col_name, vectorizer):
    df = pd.read_csv(path)
    descriptions = df[desc_col_name].fillna("").tolist()
    keys = df[key_col_name].tolist()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    tfidf_dict = {key: i for i, key in enumerate(keys)}
    return tfidf_dict, tfidf_matrix, vectorizer

def search_docs(query, tfidf_dict, tfidf_matrix, vectorizer, top_n=5):

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    key_similarity = {}
    for key, i in tfidf_dict.items():
        similarity_score = similarities[i] 
        key_similarity[key] = similarity_score 
    sorted_keys = sorted(key_similarity, key=key_similarity.get, reverse=True)
    return sorted_keys[:top_n]

def initialize_tfidf_index():
    global tfidf_dict, tfidf_vectorizer, tfidf_matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fetch data from DynamoDB
    try:
        table = dynamodb.Table(table_name)
        response = table.scan()
        items = response.get('Items', [])

        # Handle pagination if data exceeds 1 MB
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        # Extract BookIDs and Descriptions
        book_ids = []
        descriptions = []
        for item in items:
            if 'BookID' in item and 'Description' in item:
                book_ids.append(item['BookID'])
                descriptions.append(item['Description'])

        if not descriptions:
            raise ValueError("No descriptions found in the database for TF-IDF computation.")

        # Compute TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

        # Create a dictionary to map BookIDs to their TF-IDF indices
        tfidf_dict = {book_ids[i]: i for i in range(len(book_ids))}
        print("TF-IDF index initialized successfully from the database.")
    except Exception as e:
        print(f"Error initializing TF-IDF index from database: {e}")
        tfidf_dict = None
        tfidf_vectorizer = None
        tfidf_matrix = None


@app.route('/')
def home():
    return 'Home Page'

@app.route('/health')
def health_check():
    return 'Healthy', 200

@app.route('/initialize_db', methods=['POST'])
def initialize():
    create_db_table()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded for initialization'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(upload_path)

        
        try:
            with open(upload_path, mode='r') as f:
                reader = csv.DictReader(f)
                table = dynamodb.Table(table_name)
                
                
                for i, row in enumerate(reader):
                    # if i >= 1000: 
                    #     break
                    
                    item = {
                        'BookID': row['Unnamed: 0'],
                        'Book': row['Book'],
                        'Author': row['Author'],
                        'Description': row['Description'],
                        'Genres': row['Genres'],
                        'AvgRating': row['Avg_Rating'],
                        'NumRatings': row['Num_Ratings'],
                        'URL': row['URL'],
                    }
                    
                    table.put_item(Item=item)

            
            initialize_tfidf_index()
            return jsonify({'message': 'Database and TF-IDF index initialized successfully'}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to process CSV: {e}'}), 500

    return jsonify({'error': 'Allowed file types are CSV only'}), 400

@app.route('/insert', methods=['POST'])
def insert():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(upload_path)

        try:
            
            with open(upload_path, mode='r') as f:
                reader = csv.DictReader(f)
                table = dynamodb.Table(table_name)
                
                
                for i, row in enumerate(reader):
                    # if i >= 1000: 
                    #     break
                    
                    item = {
                        'BookID': row['Unnamed: 0'],
                        'Book': row['Book'],
                        'Author': row['Author'],
                        'Description': row['Description'],
                        'Genres': row['Genres'],
                        'AvgRating': row['Avg_Rating'],
                        'NumRatings': row['Num_Ratings'],
                        'URL': row['URL'],
                    }
                    table.put_item(Item=item)

            
            initialize_tfidf_index()
            return jsonify({'message': 'CSV inserted and TF-IDF updated successfully'}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to process CSV: {e}'}), 500

    return jsonify({'error': 'Allowed file types are CSV only'}), 400

@app.route('/search', methods=['GET'])
def search():
    global tfidf_dict, tfidf_vectorizer, tfidf_matrix

    query = request.args.get('query', '')
    top_n = int(request.args.get('top_n', 5))

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Check if TF-IDF is initialized
        if tfidf_dict is None or tfidf_vectorizer is None or tfidf_matrix is None:
            print("TF-IDF index is not initialized. Reinitializing from database...")
            initialize_tfidf_index()
            if tfidf_dict is None or tfidf_vectorizer is None or tfidf_matrix is None:
                return jsonify({'error': 'Failed to initialize TF-IDF index. Please check the database.'}), 500

        # Perform TF-IDF based search
        result_ids = search_docs(query, tfidf_dict, tfidf_matrix, tfidf_vectorizer, top_n)
        
        table = dynamodb.Table(table_name)
        results = []

        for book_id in result_ids:
            # Ensure the book_id is passed as a string
            response = table.get_item(Key={'BookID': str(book_id)})
            if 'Item' in response:
                book_data = response['Item']
                # Get TF-IDF score for the book
                tfidf_score = cosine_similarity(tfidf_vectorizer.transform([query]), tfidf_matrix[tfidf_dict[book_id]]).flatten()[0]
                book_data['TF-IDF_Score'] = tfidf_score
                results.append(book_data)
            else:
                results.append({'BookID': book_id, 'error': 'Book data not found in the database'})

        return jsonify({'results': results}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to search: {e}'}), 500

# return helpful information for user
@app.route('/help')
def help():
    return jsonify({
        "endpoints": {
            "/": "Home endpoint. No use for user.",
            "/search": {
                "method": "GET",
                "description": "Search for books using an inputed query and return top n results",
                "parameters": {
                    "query": "(required) The search query.",
                    "top_n": "(optional) Number of top results to return. Default is set to 5."
                },
                "response": {
                    "query": "The search query.",
                    "results": "A list of relevant books relating to the query with their details."
                }
            },
            "/initialize_db": {
                "method": "POST",
                "description": "Initialize the database with a new dataset from an inputed CSV file. All previous data in the database will be lost.",
                "parameters": {
                    "file": "(required) A CSV file in the proper format containing the dataset.",
                },
                "response": {
                    "message": "Indication if the database initialized sucessfully or if there was an error",
                },
                "csv_format": {
                    "headers": ["Unamed: 0","Book","Author","Description","Genres","Avg_Rating","Num_Ratings","URL"],
                    "example": ["6","1984","George Orwell","The new novel by...","['Classics','Dystopia','Politics']","4.19","4,201,429","http://example-link.com"]
                }
            },
            "/insert": {
                "method": "POST",
                "description": "Insert additional entries into the database from an inputed CSV file. All previous data in the database will be maintained and no duplicates will be added.",
                "parameters": {
                    "file": "(required) A CSV file in the proper format containing the new data.",
                },
                "response": {
                    "message": "Indication if the new data was added sucessfully or if there was an error",
                },
                "csv_format": {
                    "headers": ["Unamed: 0","Book","Author","Description","Genres","Avg_Rating","Num_Ratings","URL"],
                    "example": ["6","1984","George Orwell","The new novel by...","['Classics','Dystopia','Politics']","4.19","4,201,429","http://example-link.com"]
                }
            },
        },
        "notes": [
            "The CSV file must include the required headers exactly as specified.",
            "If there is an issue with the CSV file there will be an error that states: Failed to process CSV file and more information on the specific error.",
            "Initialize_db will overwrite all existing database entries while insert_db will add entries not including duplicates.",
        ]
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
