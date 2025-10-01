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

# Import Bayesian model
try:
    from bayesQ1 import CFBQuarterScorePredictor
except ImportError:
    print("ERROR: Could not import bayesQ1.py")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'cfb'
}

# Global predictor instance
predictor = None
predictor_initialized = False
initialization_error = None

def initialize_predictor():
    """Initialize the Bayesian predictor model at server startup"""
    global predictor, predictor_initialized, initialization_error
    
    try:
        print("Initializing model...")
        predictor = CFBQuarterScorePredictor(DB_CONFIG)
        
        if not predictor.load_historical_data():
            raise Exception("Failed to load historical data")
        
        predictor.calculate_empirical_distribution()
        predictor.fit_bayesian_model()
        
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
        
        predictions_result = predictor.predict_score_probabilities(
            spread=spread,
            total=total,
            top_n=20
        )
        
        if 'score_probabilities' in predictions_result:
            predictions = predictions_result['score_probabilities']
        else:
            predictions = predictions_result
        
        formatted_predictions = []
        for item in predictions:
            if isinstance(item, tuple) and len(item) == 2:
                score, probability = item
            elif isinstance(item, dict):
                score = item.get('score')
                probability = item.get('probability')
            else:
                continue
            
            if score and probability is not None:
                try:
                    prob_float = float(probability)
                    if 0 <= prob_float <= 1:
                        formatted_predictions.append({
                            'score': str(score),
                            'probability': prob_float
                        })
                except (ValueError, TypeError):
                    continue
        
        if not formatted_predictions:
            return jsonify({'error': 'No valid predictions generated'}), 500
        
        formatted_predictions.sort(key=lambda x: x['probability'], reverse=True)
        top_predictions = formatted_predictions[:20]
        
        return jsonify(top_predictions)
        
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

if __name__ == '__main__':
    main()