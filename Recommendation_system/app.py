from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor

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

# News scraping helper functions
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_brand_from_title(title):
    brands = {
        'samsung': 'Samsung',
        'apple': 'Apple',
        'iphone': 'Apple',
        'xiaomi': 'Xiaomi',
        'redmi': 'Xiaomi',
        'oneplus': 'OnePlus',
        'google': 'Google',
        'pixel': 'Google',
        'huawei': 'Huawei',
        'oppo': 'Oppo',
        'vivo': 'Vivo'
    }
    
    title_lower = title.lower()
    for key, brand in brands.items():
        if key in title_lower:
            return brand
    return 'Other'

def scrape_gsmarena():
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get('https://www.gsmarena.com/news.php3', headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        for block in soup.select('div.news-item')[:10]:
            try:
                title = clean_text(block.select_one('h3').text)
                link = 'https://www.gsmarena.com/' + block.select_one('a')['href']
                img = block.select_one('img')
                # Fix image URL construction
                img_url = img['src'] if img and img['src'].startswith('http') else \
                         'https://www.gsmarena.com/' + img['src'] if img else \
                         '/static/placeholder.jpg'
                summary = clean_text(block.select_one('p.news-text').text if block.select_one('p.news-text') else "")
                
                news_items.append({
                    'title': title,
                    'summary': summary,
                    'url': link,
                    'image': img_url,
                    'timestamp': datetime.now().isoformat(),
                    'brand': extract_brand_from_title(title)
                })
            except Exception as e:
                logger.error(f"Error parsing GSMArena news item: {str(e)}")
                continue
                
        return news_items
    except Exception as e:
        logger.error(f"Error scraping GSMArena: {str(e)}")
        return []

def scrape_phonearena():
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get('https://www.phonearena.com/news', headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        for block in soup.select('div.article-item')[:10]:
            try:
                title = clean_text(block.select_one('h4').text)
                link = block.select_one('a')['href']
                # Fix link URL construction
                link = link if link.startswith('http') else 'https://www.phonearena.com' + link
                img = block.select_one('img')
                # Fix image URL handling
                img_url = img['src'] if img and 'src' in img.attrs and img['src'].startswith('http') else '/static/placeholder.jpg'
                summary = clean_text(block.select_one('p.article-excerpt').text if block.select_one('p.article-excerpt') else "")
                
                news_items.append({
                    'title': title,
                    'summary': summary,
                    'url': link,
                    'image': img_url,
                    'timestamp': datetime.now().isoformat(),
                    'brand': extract_brand_from_title(title)
                })
            except Exception as e:
                logger.error(f"Error parsing PhoneArena news item: {str(e)}")
                continue
                
        return news_items
    except Exception as e:
        logger.error(f"Error scraping PhoneArena: {str(e)}")
        return []

def get_all_news():
    try:
        news_items = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_source = {
                executor.submit(scrape_gsmarena): 'GSMArena',
                executor.submit(scrape_phonearena): 'PhoneArena'
            }
            
            for future in future_to_source:
                try:
                    news_items.extend(future.result())
                except Exception as e:
                    logger.error(f"Error getting news from {future_to_source[future]}: {str(e)}")
        
        news_items.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Select featured news
        featured = None
        for item in news_items:
            if item['image'] != '/static/placeholder.jpg' and len(item['summary']) > 100:
                featured = item
                news_items.remove(item)
                break
        
        if not featured and news_items:
            featured = news_items.pop(0)
        
        return {
            'featured': featured,
            'news': news_items[:8]
        }
    except Exception as e:
        logger.error(f"Error in get_all_news: {str(e)}")
        return {'featured': None, 'news': []}

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        return render_template('recommend.html')
        
    try:
        logger.info("Received request for recommendations.")
        mobile_name = request.json.get('mobile_name', '').strip().lower()
        logger.info(f"Mobile name provided: {mobile_name}")

        if not mobile_name:
            logger.error("No mobile name provided in request.")
            return jsonify({'error': 'Please enter a mobile name'}), 400

        name_matches = data[data['Name'].str.lower().str.contains(mobile_name, regex=False)]
        if name_matches.empty:
            logger.warning(f"No matches found for mobile name: {mobile_name}")
            return jsonify({'error': f'No phones found matching "{mobile_name}"'}), 404

        idx = name_matches.index[0]
        logger.info(f"Matching index found: {idx}")
        sim_scores = list(enumerate(cosine_sim[idx]))

        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        logger.info("Similarity scores calculated.")

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

@app.route('/api/mobile-news')
def get_news():
    try:
        brand_filter = request.args.get('brand', '').lower()
        news_data = get_all_news()
        
        if brand_filter and news_data['news']:
            filtered_news = [
                news for news in news_data['news'] 
                if news['brand'].lower() == brand_filter
            ]
            
            if filtered_news:
                if news_data['featured']['brand'].lower() != brand_filter:
                    news_data['featured'] = filtered_news.pop(0)
                news_data['news'] = filtered_news
            
        return jsonify(news_data)
    except Exception as e:
        logger.error(f"Error in get_news: {str(e)}")
        return jsonify({'error': 'Error fetching news'}), 500

if __name__ == '__main__':
    app.run(debug=True)