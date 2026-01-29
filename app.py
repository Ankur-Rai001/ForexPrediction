from flask import Flask, request, jsonify
from flask_cors import CORS
from config import *
import jwt
import datetime
from config import authdict
import sqlite3
import hashlib
import os
from werkzeug.utils import secure_filename
from flask_cors import cross_origin
import pandas as pd
from pathlib import Path


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

authdict = {
    'USERNAME': os.environ.get('AUTH_USERNAME', 'admin'),
    'PASSWORD_HASH': os.environ.get('AUTH_PASSWORD_HASH', hashlib.sha256('secret123'.encode()).hexdigest()),
    'SECRET_KEY': os.environ.get('SECRET_KEY', '6no!r0#)gf(y%$9fb#u*9_!t=!8v5_sorv8k^nzs!5gwcxj#v!')
}

### DB Connect ####
connect = sqlite3.connect('database.db')

########## Rplace null with empty string#######
def replace_null_with_empty_str(row):
    """
    Converts a sqlite3.Row object to a dictionary
    and replaces None values with empty strings.
    """
    return {key: ('' if value is None else value) for key, value in dict(row).items()}


# Route 1: Generate JWT Token (valid for 1 hour)
@app.route('/get-token', methods=['POST'])
def get_token():
    try:
        # You could also check for username/password here
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            'iat': datetime.datetime.utcnow(),
            'user': 'frontend'
        }
        token = jwt.encode(payload, authdict['SECRET_KEY'], algorithm='HS256')
        return jsonify({'token': token}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Authorization token missing'}), 401
        token = token.replace('Bearer ', '')
        data = request.get_json()
        try:
            payload = jwt.decode(token, authdict['SECRET_KEY'], algorithms=['HS256'])
            if not data or 'currency' not in data or 'interval' not in data:
                return jsonify({'error': 'Missing currency or interval'}), 400
            currency = data['currency']
            interval = data['interval']
            imgurl=''
            if interval=='1day':
                # imgurl=url_for('static', filename='gbpusd_forecast_1day.png', _external=True)
                imgurl="https://api.forexedge.in/api/static/gbpusd-dollar-volume-bars-3-day-performance-5-day-forecast.png"
            if interval=='4hour':
                imgurl="https://api.forexedge.in/api/static/gbpusd-dollar-volume-bars-4h-forecast-prediction-chart.png" 
            if interval=='1hour':
                pass
            if interval=='15min':
                pass
            # script_path = get_script_path(currency, interval)
            # result_data = run_prediction_script(script_path)
            if imgurl is None:
                app.logger.error(f"Failed to generate prediction data for {currency} {interval}")
                return jsonify({'error': 'Failed to generate prediction data'}), 500
            return jsonify({"imageurl":imgurl}), 200
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
    except ValueError as ve:
        app.logger.error(f"ValueError: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500



UPLOAD_FOLDER = 'static/articleimages'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/insert-article', methods=['POST'])
def insert_article():
    try:
        
        # --- Authorization ---
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Authorization token missing'}), 401
        token = token.replace('Bearer ', '')

        try:
            payload = jwt.decode(token, authdict['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        # --- Handle Form Data ---
        data = request.form
        required_fields = ['title', 'content', 'authorName', 'articleSeoUrl',
                           'metaTitle', 'metaKeyword', 'createdBy', 'updatedBy']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # --- Handle File Upload ---
        image_file = request.files.get('image')
        image_url = None

        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(file_path)
            image_url = f"/{file_path}"  # or full URL if using a domain
            image_url = '/' + file_path.replace('\\', '/')
            image_url = '' + file_path.replace(' ', '-')


        # --- Insert into SQLite ---
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO articles (
                title,
                content,
                authorName,
                articleSeoUrl,
                metaTitle,
                metaKeyword,
                imageURL,
                createdBy,
                createdAt,
                updatedBy,
                updatedAt,
                isDeleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, CURRENT_TIMESTAMP, 0)
        """, (
            data['title'],
            sqlite3.Binary(data['content'].encode('utf-8')),
            data['authorName'],
            data['articleSeoUrl'],
            data['metaTitle'],
            data['metaKeyword'],
            image_url,
            data['createdBy'],
            data['updatedBy']
        ))

        conn.commit()
        conn.close()
        return jsonify({'message': 'Article inserted successfully'}), 201

    except ValueError as ve:
        app.logger.error(f"ValueError: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'insertion failed: {str(e)}'}), 500



@app.route('/article/<string:article_seo_url>', methods=['GET', 'OPTIONS'])
@cross_origin(origins="*", allow_headers=["Authorization", "Content-Type"], methods=["GET", "OPTIONS"])
def get_article_by_seo_url(article_seo_url):
    if request.method == 'OPTIONS':
        return '', 204  # Preflight request successful

    try:
        # --- Authorization ---
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Authorization token missing'}), 401
        token = token.replace('Bearer ', '')

        try:
            payload = jwt.decode(token, authdict['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        # --- Database query ---
        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id,
                title,
                content,
                authorName,
                articleSeoUrl,
                metaTitle,
                metaKeyword,
                imageURL,
                createdBy,
                createdAt,
                updatedBy,
                updatedAt,
                isDeleted
            FROM articles
            WHERE articleSeoUrl = ? AND isDeleted = 0
            LIMIT 1
        """, (article_seo_url,))
        
        articles = cursor.fetchone()
        article = replace_null_with_empty_str(articles)
        conn.close()

        if not article:
            return jsonify({'error': 'Article not found'}), 404

        article_dict = dict(article)

        if article_dict['content']:
            try:
                article_dict['content'] = article_dict['content'].decode('utf-8')
            except Exception:
                pass

        return jsonify({'article': article_dict}), 200

    except Exception as e:
        app.logger.error(f"Error fetching article detail: {str(e)}")
        return jsonify({'error': f'Failed to fetch article: {str(e)}'}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password required'}), 400

    username = data['username']
    password = data['password']

    # --- Hash the input password ---
    hashed_input_password = hashlib.sha256(password.encode()).hexdigest()

    # --- Check credentials ---
    if username == authdict['USERNAME'] and hashed_input_password == authdict['PASSWORD_HASH']:
        return jsonify({"message": "Login Success"}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401



@app.route('/articles', methods=['GET'])
@cross_origin(origins="*", allow_headers=["Authorization"], methods=["GET"])
def get_articles():
    try:
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Authorization token missing'}), 401
        token = token.replace('Bearer ', '')

        try:
            payload = jwt.decode(token, authdict['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id,
                title,
                authorName,
                articleSeoUrl,
                metaTitle,
                metaKeyword,
                imageURL,
                createdBy,
                createdAt,
                updatedBy,
                updatedAt
            FROM articles
            WHERE isDeleted = 0
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        articles = [replace_null_with_empty_str(row) for row in rows]
        conn.close()

        return jsonify({'articles': articles}), 200

    except Exception as e:
        app.logger.error(f"Error fetching articles: {str(e)}")
        return jsonify({'error': f'Failed to fetch articles: {str(e)}'}), 500


@app.route('/fetch-forecast', methods=['POST'])
@cross_origin(origins="*", allow_headers=["Authorization", "Content-Type"], methods=["POST", "OPTIONS"])
def fetch_forecast():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        if not request.is_json:
            app.logger.error("Request Content-Type is not 'application/json'")
            return jsonify({'error': "Content-Type must be 'application/json'"}), 415
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Authorization token missing'}), 401
        token = token.replace('Bearer ', '')
        try:
            payload = jwt.decode(token, authdict['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        data = request.get_json()
        if not data or 'interval' not in data:
            return jsonify({'error': 'Missing interval'}), 400
        interval = data['interval'].lower()
        valid_intervals = ['1day', '4hour', '1hour']
        if interval not in valid_intervals:
            return jsonify({'error': f'Invalid interval. Must be one of {valid_intervals}'}), 400
        STATIC_DIR = os.path.join(os.getcwd(), 'static')
        CSV_PATH = os.path.join(STATIC_DIR, 'gbpusd_forecasted_prices.csv')
        if not Path(CSV_PATH).is_file():
            app.logger.error(f"CSV file not found at {CSV_PATH}")
            return jsonify({'error': 'Forecast data file not found'}), 404
        with open(CSV_PATH, 'r') as f:
            lines = f.readlines()
        current_section = None
        section_data = []
        all_sections = {'1day': [], '4hour': [], '1hour': []}
        for line in lines:
            line = line.strip()
            if not line:
                if current_section and section_data:
                    try:
                        df = pd.read_csv(
                            pd.io.common.StringIO('\n'.join(section_data)),
                            names=['Date', 'Price'],
                            skiprows=1 if section_data[0].startswith('Date') else 0
                        )
                        if current_section == '1 Day Forecast':
                            all_sections['1day'] = df.to_dict('records')
                        elif current_section == '4 Hour Forecast':
                            all_sections['4hour'] = df.to_dict('records')
                        elif current_section == '1 Hour Forecast':
                            all_sections['1hour'] = df.to_dict('records')
                    except Exception as e:
                        app.logger.error(f"Error parsing section {current_section}: {str(e)}")
                    section_data = []
                continue
            if line.startswith('#'):
                if current_section and section_data:
                    try:
                        df = pd.read_csv(
                            pd.io.common.StringIO('\n'.join(section_data)),
                            names=['Date', 'Price'],
                            skiprows=1 if section_data[0].startswith('Date') else 0
                        )
                        if current_section == '1 Day Forecast':
                            all_sections['1day'] = df.to_dict('records')
                        elif current_section == '4 Hour Forecast':
                            all_sections['4hour'] = df.to_dict('records')
                        elif current_section == '1 Hour Forecast':
                            all_sections['1hour'] = df.to_dict('records')
                    except Exception as e:
                        app.logger.error(f"Error parsing section {current_section}: {str(e)}")
                    section_data = []
                current_section = line[2:].strip()
                continue
            section_data.append(line)
        if current_section and section_data:
            try:
                df = pd.read_csv(
                    pd.io.common.StringIO('\n'.join(section_data)),
                    names=['Date', 'Price'],
                    skiprows=1 if section_data[0].startswith('Date') else 0
                )
                if current_section == '1 Day Forecast':
                    all_sections['1day'] = df.to_dict('records')
                elif current_section == '4 Hour Forecast':
                    all_sections['4hour'] = df.to_dict('records')
                elif current_section == '1 Hour Forecast':
                    all_sections['1hour'] = df.to_dict('records')
            except Exception as e:
                app.logger.error(f"Error parsing section {current_section}: {str(e)}")
        forecast_data = all_sections.get(interval, [])
        if not forecast_data:
            app.logger.warning(f"No forecast data found for interval: {interval}")
            return jsonify({'error': f'No forecast data available for interval: {interval}'}), 404
        return jsonify({'forecast': forecast_data}), 200
    except ValueError as ve:
        app.logger.error(f"ValueError: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Failed to fetch forecast: {str(e)}'}), 500





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)