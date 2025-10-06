#!/usr/bin/env python3
"""
Flask API Server for CFB Quarter Score Predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import logging
from datetime import datetime
import sys
import traceback
import os
from dotenv import load_dotenv
import requests
# Import model
try:
    from Q1 import CFBQuarterScorePredictor
except ImportError:
    print("ERROR: Could not import Q1.py")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Database configuration from environment
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'cfb')
}

# Global predictor instance
predictor = None
predictor_initialized = False
initialization_error = None

def initialize_predictor():
    """Initialize the predictor model at server startup"""
    global predictor, predictor_initialized, initialization_error
    
    try:
        print("Initializing model...")
        predictor = CFBQuarterScorePredictor(DB_CONFIG)
        
        if not predictor.load_historical_data():
            raise Exception("Failed to load historical data")
        
        predictor.calculate_empirical_distribution()
        predictor.fit_model()
        
        predictor_initialized = True
        print("Model initialized successfully")
        return True
        
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"Initialization failed: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_initialized': predictor_initialized
    })

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Model status information"""
    if not predictor_initialized:
        return jsonify({
            'initialized': False,
            'error': initialization_error
        }), 500
    
    try:
        total_games = len(predictor.historical_data) if predictor.historical_data is not None else 0
        unique_scores = predictor.historical_data['score_combination'].nunique() if predictor.historical_data is not None else 0
        
        return jsonify({
            'initialized': True,
            'training_data': {
                'total_games': total_games,
                'unique_scores': unique_scores
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-quarter-scores', methods=['POST'])
def predict_quarter_scores():
    """Main prediction endpoint"""
    if not predictor_initialized:
        return jsonify({
            'error': 'Prediction model not initialized',
            'details': initialization_error
        }), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        spread = data.get('spread')
        total = data.get('total') 
        home_team = data.get('homeTeam', 'Home')
        away_team = data.get('awayTeam', 'Away')
        
        if spread is None or total is None:
            return jsonify({'error': 'Missing required parameters: spread and total'}), 400
        
        try:
            spread = float(spread)
            total = float(total)
        except (ValueError, TypeError):
            return jsonify({'error': 'Spread and total must be numeric values'}), 400
        
        if not (-50 <= spread <= 50):
            return jsonify({'error': 'Spread must be between -50 and +50'}), 400
            
        if not (30 <= total <= 90):
            return jsonify({'error': 'Total must be between 30 and 90'}), 400
        
        logger.info(f"Prediction: {away_team} @ {home_team}, Spread: {spread}, Total: {total}")
        
        # Get ALL score probabilities from the model
        all_predictions = predictor.predict_score_probabilities(
            spread=spread,
            total=total
        )
        
        # Calculate betting markets using the FULL distribution
        markets = predictor.calculate_betting_markets(all_predictions, spread, total)
        
        # Sort by probability and take top 20 for display
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        top_20_predictions = sorted_predictions[:20]
        
        # Format the top 20 predictions for display
        formatted_predictions = []
        for score, probability in top_20_predictions:
            formatted_predictions.append({
                'score': str(score),
                'probability': float(probability)
            })
        
        if not formatted_predictions:
            return jsonify({'error': 'No valid predictions generated'}), 500
        
        # Return both predictions and pre-calculated markets
        return jsonify({
            'predictions': formatted_predictions,
            'markets': markets
        })
        
    except mysql.connector.Error as e:
        logger.error(f"Database error: {e}")
        return jsonify({'error': 'Database connection error'}), 500
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal prediction error', 'details': str(e)}), 500

@app.route('/')
def index():
    """Status page"""
    status = 'Model initialized and ready' if predictor_initialized else f'Model initialization failed: {initialization_error}'
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CFB Quarter Score Prediction API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 8px; }}
            .status {{ padding: 10px; border-radius: 4px; margin: 10px 0; }}
            .success {{ background: #d4edda; color: #155724; }}
            .error {{ background: #f8d7da; color: #721c24; }}
            code {{ background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CFB Quarter Score Prediction API</h1>
            <div class="status {'success' if predictor_initialized else 'error'}">
                <strong>Status:</strong> {status}
            </div>
            <h3>Endpoints:</h3>
            <ul>
                <li><code>GET /api/health</code> - Health check</li>
                <li><code>GET /api/model-status</code> - Model status</li>
                <li><code>POST /api/predict-quarter-scores</code> - Generate predictions</li>
                <li><code>GET /interface</code> - Web interface</li>
            </ul>
        </div>
    </body>
    </html>
    '''

@app.route('/interface')
def serve_interface():
    """Serve the CFB interface"""
    try:
        with open('CFBInterface.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "CFBInterface.html not found. Make sure it's in the same directory as this script.", 404

@app.route('/api/calculate-parlay', methods=['POST'])
def calculate_parlay():
    """Calculate SGP probability and fair odds"""
    if not predictor_initialized:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        data = request.get_json()
        spread = float(data.get('spread'))
        total = float(data.get('total'))
        selections = data.get('selections', [])
        
        if not selections:
            return jsonify({'error': 'No selections provided'}), 400
        
        # Get full probability distribution
        all_probs = predictor.predict_score_probabilities(spread, total)
        
        # Calculate parlay probability
        parlay_prob = predictor.calculate_parlay_probability(all_probs, selections)
        
        # Calculate fair odds
        fair_odds = predictor.calculate_parlay_payout(parlay_prob)
        
        return jsonify({
            'probability': float(parlay_prob),
            'fair_odds': int(fair_odds),
            'decimal_odds': float(1 / parlay_prob) if parlay_prob > 0 else 0,
            'selections_count': len(selections)
        })
        
    except Exception as e:
        logger.error(f"Parlay calculation error: {e}")
        return jsonify({'error': str(e)}), 500
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main server startup function"""
    print("="*60)
    print("CFB Quarter Score Prediction API Server")
    print("="*60)
    
    # Test database connection
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        connection.close()
        print("Database connection: OK")
    except mysql.connector.Error as e:
        print(f"Database connection failed: {e}")
        return
    
    # Initialize the prediction model
    if not initialize_predictor():
        print("Warning: Model initialization failed, but server will start")
    
    print("\n" + "="*60)
    print("Server starting at: http://localhost:5000")
    print("Interface available at: http://localhost:5000/interface")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Start Flask server
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        threaded=True
    )
@app.route('/api/cfbd/games', methods=['GET'])
def get_cfbd_games():
    """Proxy CFBD games API to hide API key"""
    try:
        year = request.args.get('year')
        season_type = request.args.get('seasonType', 'regular')
        
        api_key = os.getenv('CFBD_API_KEY')
        if not api_key:
            return jsonify({'error': 'API key not configured'}), 500
        
        response = requests.get(
            'https://api.collegefootballdata.com/games',
            params={'year': year, 'seasonType': season_type},
            headers={
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json'
            },
            timeout=10
        )
        
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({'error': f'CFBD API error: {response.status_code}'}), response.status_code
            
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        return jsonify({'error': 'Failed to fetch games'}), 500

@app.route('/api/cfbd/lines', methods=['GET'])
def get_cfbd_lines():
    """Proxy CFBD betting lines API to hide API key"""
    try:
        year = request.args.get('year')
        season_type = request.args.get('seasonType', 'regular')
        
        api_key = os.getenv('CFBD_API_KEY')
        if not api_key:
            return jsonify({'error': 'API key not configured'}), 500
        
        response = requests.get(
            'https://api.collegefootballdata.com/lines',
            params={'year': year, 'seasonType': season_type},
            headers={
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json'
            },
            timeout=10
        )
        
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({'error': f'CFBD API error: {response.status_code}'}), response.status_code
            
    except Exception as e:
        logger.error(f"Error fetching lines: {e}")
        return jsonify({'error': 'Failed to fetch betting lines'}), 500
if __name__ == '__main__':
    main()