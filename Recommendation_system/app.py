from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the dataset
try:
    data = pd.read_csv('dataset/mobile.csv')
    logger.info("Mobile dataset loaded successfully.")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise e

# Preprocessing and creating the cosine similarity matrix
try:
    # Convert relevant columns to strings and handle NaN values
    data['Processor'] = data['Processor'].fillna("").astype(str)
    data['Rear camera'] = data['Rear camera'].fillna("").astype(str)
    data['Front camera'] = data['Front camera'].fillna("").astype(str)

    # Create the combined features column
    data['combined_features'] = (
        data['Processor'] + " " + data['Rear camera'] + " " + data['Front camera']
    )

    # Generate the TF-IDF matrix and cosine similarity matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    logger.info("Cosine similarity matrix created successfully.")
except Exception as e:
    logger.error(f"Error creating cosine similarity matrix: {str(e)}")
    raise e

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        logger.info("Received request for recommendations.")
        # Parse mobile name from the JSON request
        mobile_name = request.json.get('mobile_name', '').strip().lower()
        logger.info(f"Mobile name provided: {mobile_name}")

        # Validate input
        if not mobile_name:
            logger.error("No mobile name provided in request.")
            return jsonify({'error': 'Please enter a mobile name'}), 400

        # Search for matching mobile names (case-insensitive)
        name_matches = data[data['Name'].str.lower().str.contains(mobile_name, regex=False)]
        if name_matches.empty:
            logger.warning(f"No matches found for mobile name: {mobile_name}")
            return jsonify({'error': f'No phones found matching "{mobile_name}"'}), 404

        # Use the first matching mobile for similarity calculation
        idx = name_matches.index[0]
        logger.info(f"Matching index found: {idx}")
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Get top 5 recommendations excluding the phone itself
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        logger.info("Similarity scores calculated.")

        # Build recommendation results
        results = []
        for i, score in sorted_scores:
            phone = data.iloc[i]
            results.append({
                'name': str(phone['Name']),
                'brand': str(phone['Brand']),
                'price': int(phone['Price']) if pd.notnull(phone['Price']) else None,
                'score': round(float(score), 2),
                'specs': {
                    'processor': str(phone['Processor']),
                    'ram': f"{int(phone['RAM (MB)'])}MB" if pd.notnull(phone['RAM (MB)']) else "N/A",
                    'storage': f"{int(phone['Internal storage (GB)'])}GB" if pd.notnull(phone['Internal storage (GB)']) else "N/A",
                    'camera': f"{phone['Rear camera']} + {phone['Front camera']}"
                }
            })

        logger.info("Recommendations generated successfully.")
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error during recommendation: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
