#!/usr/bin/env python3
"""
Flask API Server for CFB All Quarters Score Predictions
Uses all.py for comprehensive quarter and half markets
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
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import random
import time

# Import model
try:
    from all import CFBAllQuartersPredictor
except ImportError:
    print("ERROR: Could not import all.py")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Suppress werkzeug logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

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
        print("Initializing all quarters model...")
        predictor = CFBAllQuartersPredictor(DB_CONFIG)
        
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
        
        return jsonify({
            'initialized': True,
            'training_data': {
                'total_games': total_games,
                'quarters': ['Q1', 'Q2', 'Q3', 'Q4', 'H1', 'H2', 'Full']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_highest_scoring_quarter(q1_dist, q2_dist, q3_dist, q4_dist, num_simulations=10000):
    """
    Calculate probability of each quarter being the highest scoring using Monte Carlo.
    
    Returns dict with probabilities for: q1, q2, q3, q4, tie
    """
    print(f"\n=== HIGHEST SCORING QUARTER CALCULATION ===")
    print(f"Running {num_simulations} simulations...")
    
    # Convert distributions to lists for sampling
    q1_scores = list(q1_dist.keys())
    q1_probs = list(q1_dist.values())
    
    q2_scores = list(q2_dist.keys())
    q2_probs = list(q2_dist.values())
    
    q3_scores = list(q3_dist.keys())
    q3_probs = list(q3_dist.values())
    
    q4_scores = list(q4_dist.keys())
    q4_probs = list(q4_dist.values())
    
    # Track results
    results = {
        'q1': 0,
        'q2': 0,
        'q3': 0,
        'q4': 0,
        'tie': 0
    }
    
    import random
    
    for _ in range(num_simulations):
        # Sample one score from each quarter
        q1_sample = random.choices(q1_scores, weights=q1_probs)[0]
        q2_sample = random.choices(q2_scores, weights=q2_probs)[0]
        q3_sample = random.choices(q3_scores, weights=q3_probs)[0]
        q4_sample = random.choices(q4_scores, weights=q4_probs)[0]
        
        # Calculate totals for each quarter
        q1_total = sum(map(int, q1_sample.split('-')))
        q2_total = sum(map(int, q2_sample.split('-')))
        q3_total = sum(map(int, q3_sample.split('-')))
        q4_total = sum(map(int, q4_sample.split('-')))
        
        # Find max and check for ties
        max_total = max(q1_total, q2_total, q3_total, q4_total)
        max_count = sum([
            q1_total == max_total,
            q2_total == max_total,
            q3_total == max_total,
            q4_total == max_total
        ])
        
        if max_count > 1:
            # Tie for highest scoring
            results['tie'] += 1
        elif q1_total == max_total:
            results['q1'] += 1
        elif q2_total == max_total:
            results['q2'] += 1
        elif q3_total == max_total:
            results['q3'] += 1
        else:
            results['q4'] += 1
    
    # Convert to probabilities
    for key in results:
        results[key] = results[key] / num_simulations
    
    print(f"Results:")
    print(f"  Q1: {results['q1']*100:.1f}%")
    print(f"  Q2: {results['q2']*100:.1f}%")
    print(f"  Q3: {results['q3']*100:.1f}%")
    print(f"  Q4: {results['q4']*100:.1f}%")
    print(f"  Tie: {results['tie']*100:.1f}%")
    print(f"=== END CALCULATION ===\n")
    
    return results

def sample_quarter_vectorized(dist, num_sims):
    """
    Vectorized sampling from a quarter score distribution.
    Returns two NumPy arrays: (home_scores, away_scores)
    """
    scores_str = list(dist.keys())
    probabilities = list(dist.values())
    
    # Parse scores into integer pairs
    scores_int = np.array([s.split('-') for s in scores_str], dtype=int)
    
    # Sample indices based on probabilities
    indices = np.arange(len(scores_str))
    sampled_indices = random.choices(indices, weights=probabilities, k=num_sims)
    
    # Get sampled scores
    sampled_scores = scores_int[sampled_indices]
    
    return sampled_scores[:, 0], sampled_scores[:, 1]

def calculate_sgp_monte_carlo(distributions, selections, num_simulations=100000):
    """
    Calculate SGP probability using vectorized Monte Carlo simulation.
    Naturally captures correlations between quarters, halves, and full game.
    """
    # Sample all quarters vectorized
    q1_home, q1_away = sample_quarter_vectorized(distributions['q1_home_away'], num_simulations)
    q2_home, q2_away = sample_quarter_vectorized(distributions['q2_home_away'], num_simulations)
    q3_home, q3_away = sample_quarter_vectorized(distributions['q3_home_away'], num_simulations)
    q4_home, q4_away = sample_quarter_vectorized(distributions['q4_home_away'], num_simulations)
    
    # Calculate all derived values vectorized
    q1_total = q1_home + q1_away
    q1_margin = q1_home - q1_away
    q2_total = q2_home + q2_away
    q2_margin = q2_home - q2_away
    q3_total = q3_home + q3_away
    q3_margin = q3_home - q3_away
    q4_total = q4_home + q4_away
    q4_margin = q4_home - q4_away
    
    h1_home = q1_home + q2_home
    h1_away = q1_away + q2_away
    h1_total = h1_home + h1_away
    h1_margin = h1_home - h1_away
    
    h2_home = q3_home + q4_home
    h2_away = q3_away + q4_away
    h2_total = h2_home + h2_away
    h2_margin = h2_home - h2_away
    
    full_home = h1_home + h2_home
    full_away = h1_away + h2_away
    full_total = full_home + full_away
    full_margin = full_home - full_away
    
    # Start with all simulations as hits
    all_hits = np.ones(num_simulations, dtype=bool)
    
    # Evaluate each selection
    for selection in selections:
        period = selection.get('period')
        bet_type = selection.get('type')
        line = float(selection.get('line', 0))
        side = selection.get('side')
        
        leg_hits = np.zeros(num_simulations, dtype=bool)
        
        # Handle total bets
        if bet_type == 'total':
            period_total = None
            if period == 'q1':
                period_total = q1_total
            elif period == 'q2':
                period_total = q2_total
            elif period == 'q3':
                period_total = q3_total
            elif period == 'q4':
                period_total = q4_total
            elif period == 'h1':
                period_total = h1_total
            elif period == 'h2':
                period_total = h2_total
            elif period == 'full':
                period_total = full_total
            
            if period_total is not None:
                if side == 'over':
                    leg_hits = (period_total > line)
                elif side == 'under':
                    leg_hits = (period_total < line)
        
        # Handle spread bets
        elif bet_type == 'spread':
            period_margin = None
            if period == 'q1':
                period_margin = q1_margin
            elif period == 'q2':
                period_margin = q2_margin
            elif period == 'q3':
                period_margin = q3_margin
            elif period == 'q4':
                period_margin = q4_margin
            elif period == 'h1':
                period_margin = h1_margin
            elif period == 'h2':
                period_margin = h2_margin
            elif period == 'full':
                period_margin = full_margin
            
            if period_margin is not None:
                if side == 'home':
                    leg_hits = (period_margin > line)
                elif side == 'away':
                    leg_hits = (period_margin < -line)
        
        # Handle moneyline bets
        elif bet_type == 'moneyline':
            period_margin = None
            if period == 'q1':
                period_margin = q1_margin
            elif period == 'q2':
                period_margin = q2_margin
            elif period == 'q3':
                period_margin = q3_margin
            elif period == 'q4':
                period_margin = q4_margin
            elif period == 'h1':
                period_margin = h1_margin
            elif period == 'h2':
                period_margin = h2_margin
            elif period == 'full':
                period_margin = full_margin
            
            if period_margin is not None:
                if side == 'home':
                    leg_hits = (period_margin > 0)
                elif side == 'away':
                    leg_hits = (period_margin < 0)
        
        # Handle moneyline_3way
        elif bet_type == 'moneyline_3way':
            period_margin = None
            if period == 'q1':
                period_margin = q1_margin
            elif period == 'q2':
                period_margin = q2_margin
            elif period == 'q3':
                period_margin = q3_margin
            elif period == 'q4':
                period_margin = q4_margin
            elif period == 'h1':
                period_margin = h1_margin
            elif period == 'h2':
                period_margin = h2_margin
            elif period == 'full':
                period_margin = full_margin
            
            if period_margin is not None:
                if side == 'home':
                    leg_hits = (period_margin > 0)
                elif side == 'away':
                    leg_hits = (period_margin < 0)
                elif side == 'draw':
                    leg_hits = (period_margin == 0)
        
        # Handle highest_scoring_quarter
        elif bet_type == 'highest_scoring_quarter':
            all_q_totals = np.stack([q1_total, q2_total, q3_total, q4_total], axis=1)
            max_scores = np.max(all_q_totals, axis=1, keepdims=True)
            num_at_max = np.sum(all_q_totals == max_scores, axis=1)
            
            if side == 'tie':
                leg_hits = (num_at_max > 1)
            elif side == 'q1':
                leg_hits = (q1_total == max_scores.squeeze()) & (num_at_max == 1)
            elif side == 'q2':
                leg_hits = (q2_total == max_scores.squeeze()) & (num_at_max == 1)
            elif side == 'q3':
                leg_hits = (q3_total == max_scores.squeeze()) & (num_at_max == 1)
            elif side == 'q4':
                leg_hits = (q4_total == max_scores.squeeze()) & (num_at_max == 1)
        
        # Combine with master hits
        all_hits = all_hits & leg_hits
    
    # Calculate final probability
    return np.mean(all_hits)

def calculate_markets_for_period(predictions, spread, total, period_name):
    """
    Calculate all betting markets for a specific period (quarter, half, or full game).
    
    Args:
        predictions: dict of 'home-away' score combinations to probabilities
        spread: game spread (SIGNED: negative = home favored, positive = away favored)
        total: game total (for calculating period total)
        period_name: 'q1', 'q2', 'q3', 'q4', 'h1', 'h2', or 'full'
    """
    
    print(f"\n=== MARKET CALCULATION DEBUG: {period_name.upper()} ===")
    print(f"Input spread: {spread}, Input total: {total}")
    
    # Determine who is favored
    home_favored = spread < 0
    
    print(f"Home favored: {home_favored}")
    
    # Period-specific spread and total adjustments
    period_factors = {
        'q1': {'spread': 0.285, 'total': 0.263},
        'q2': {'spread': 0.285, 'total': 0.257},
        'q3': {'spread': 0.285, 'total': 0.250},
        'q4': {'spread': 0.285, 'total': 0.230},
        'h1': {'spread': 0.5, 'total': 0.52},
        'h2': {'spread': 0.5, 'total': 0.48},
        'full': {'spread': 1.0, 'total': 1.0}
    }
    
    factor = period_factors.get(period_name, {'spread': 1.0, 'total': 1.0})
    period_spread = spread * factor['spread']
    period_total = total * factor['total']
    
    print(f"Period factors: spread={factor['spread']}, total={factor['total']}")
    print(f"Calculated period_spread: {period_spread}")
    print(f"Calculated period_total: {period_total}")
    
    # Initialize market probabilities
    home_win = 0
    away_win = 0
    draw = 0
    
    # Generate candidate spread lines from -15 to +15 in 0.5 increments
    all_spread_lines = {}
    for i in range(-30, 31):  # -15 to +15 in 0.5 increments
        line_val = i * 0.5
        all_spread_lines[line_val] = {'home_cover': 0, 'away_cover': 0}
    
    # Generate candidate total lines from 0 to 100 in 0.5 increments  
    all_total_lines = {}
    for i in range(0, 201):  # 0 to 100 in 0.5 increments
        line_val = i * 0.5
        all_total_lines[line_val] = {'over': 0, 'under': 0}
    
    # Calculate probabilities for all lines first regardless of period
    for score_str, prob in predictions.items():
        try:
            home_score, away_score = map(int, score_str.split('-'))
        except:
            continue
        
        score_total = home_score + away_score
        margin = home_score - away_score
        
        # 2-way moneyline
        if margin > 0:
            home_win += prob
        elif margin < 0:
            away_win += prob
        else:
            home_win += prob / 2
            away_win += prob / 2
            draw += prob
        
        # All spread lines - Lines are absolute values
        for line, values in all_spread_lines.items():
            # Calculate favorite's margin
            if home_favored:
                fav_margin = margin  # home is favorite
            else:
                fav_margin = -margin  # away is favorite
            
            # Favorite covers if their margin > line
            if fav_margin > line:
                values['home_cover'] += prob
            elif fav_margin < line:
                values['away_cover'] += prob
            else:
                values['home_cover'] += prob / 2
                values['away_cover'] += prob / 2
        
        # All total lines
        for line, values in all_total_lines.items():
            if score_total > line:
                values['over'] += prob
            elif score_total < line:
                values['under'] += prob
            else:
                values['over'] += prob / 2
                values['under'] += prob / 2
    
    # For FULL GAME ONLY: use the actual market spread/total as midpoint
    # and calibrate to ensure they are exactly 50/50
    if period_name == 'full':
        # Use exact market spread and total - THESE MUST BE 50/50
        midpoint_spread = abs(spread)  # Use absolute value since we're in fav-dog format
        midpoint_total = total
        
        print(f"FULL GAME: Using exact market lines as anchors")
        print(f"Market spread (abs): {midpoint_spread}")
        print(f"Market total: {midpoint_total}")
        
        # CRITICAL: Force the midpoint spread to be exactly 50/50
        if midpoint_spread in all_spread_lines:
            current_home_cover = all_spread_lines[midpoint_spread]['home_cover']
            current_away_cover = all_spread_lines[midpoint_spread]['away_cover']
            
            print(f"Original spread probs at {midpoint_spread}: Home {current_home_cover*100:.2f}%, Away {current_away_cover*100:.2f}%")
            
            # Force to 50/50 for calibration
            all_spread_lines[midpoint_spread]['home_cover'] = 0.5
            all_spread_lines[midpoint_spread]['away_cover'] = 0.5
            
            print(f"Calibrated spread to: Home 50.00%, Away 50.00%")
        
        # CRITICAL: Force the midpoint total to be exactly 50/50
        if midpoint_total in all_total_lines:
            current_over = all_total_lines[midpoint_total]['over']
            current_under = all_total_lines[midpoint_total]['under']
            
            print(f"Original total probs at {midpoint_total}: Over {current_over*100:.2f}%, Under {current_under*100:.2f}%")
            
            # Force to 50/50 for calibration
            all_total_lines[midpoint_total]['over'] = 0.5
            all_total_lines[midpoint_total]['under'] = 0.5
            
            print(f"Calibrated total to: Over 50.00%, Under 50.00%")
        
    else:
        # For quarters/halves: find the line closest to 50% probability
        # Probabilities are already calculated above
        # Find spread line closest to 50% for home team
        best_spread_line = None
        best_spread_diff = 1.0
        for line, probs in all_spread_lines.items():
            diff = abs(probs['home_cover'] - 0.5)
            if diff < best_spread_diff:
                best_spread_diff = diff
                best_spread_line = line
        
        # Find total line closest to 50% for over
        best_total_line = None
        best_total_diff = 1.0
        for line, probs in all_total_lines.items():
            diff = abs(probs['over'] - 0.5)
            if diff < best_total_diff:
                best_total_diff = diff
                best_total_line = line
        
        print(f"50% spread line: {best_spread_line} (home cover: {all_spread_lines[best_spread_line]['home_cover']*100:.1f}%)")
        print(f"50% total line: {best_total_line} (over: {all_total_lines[best_total_line]['over']*100:.1f}%)")
        
        midpoint_spread = best_spread_line
        midpoint_total = best_total_line
    
    # Select 7 spread lines centered at midpoint (3 on each side)
    spread_lines = {}
    for i in range(-6, 7):  # -3 to +3 in 0.5 increments
        line_val = midpoint_spread + (i * 0.5)
        if line_val in all_spread_lines:
            spread_lines[line_val] = all_spread_lines[line_val]
    
    total_lines = {}
    for i in range(-6, 7):  # -3 to +3 in 0.5 increments
        line_val = midpoint_total + (i * 0.5)
        if line_val >= 0 and line_val in all_total_lines:
            total_lines[line_val] = all_total_lines[line_val]
    
    print(f"Spread lines generated: {sorted(spread_lines.keys())}")
    print(f"Total lines generated: {sorted(total_lines.keys())}")
    
    # Special markets
    draw_and_over_half = 0
    for score_str, prob in predictions.items():
        try:
            home_score, away_score = map(int, score_str.split('-'))
        except:
            continue
        
        margin = home_score - away_score
        score_total = home_score + away_score
        
        if margin == 0 and score_total > (period_total / 2):
            draw_and_over_half += prob
    
    # Calculate fair odds using standard American odds formula
    def prob_to_american_odds(p):
        """Convert probability to American odds"""
        if p <= 0 or p >= 1:
            return 0
        if p >= 0.5:
            return int(-100 * p / (1 - p))
        else:
            return int(100 * (1 - p) / p)
    
    # Format spread markets
    spread_markets = {}
    for line, probs in sorted(spread_lines.items()):
        spread_markets[line] = {
            'home_cover': probs['home_cover'],
            'away_cover': probs['away_cover'],
            'home_odds': prob_to_american_odds(probs['home_cover']),
            'away_odds': prob_to_american_odds(probs['away_cover'])
        }
    
    # Format total markets
    total_markets = {}
    for line, probs in sorted(total_lines.items()):
        total_markets[line] = {
            'over': probs['over'],
            'under': probs['under'],
            'over_odds': prob_to_american_odds(probs['over']),
            'under_odds': prob_to_american_odds(probs['under'])
        }
    
    print(f"Final spread markets count: {len(spread_markets)}")
    print(f"Final total markets count: {len(total_markets)}")
    print(f"=== END DEBUG ===\n")
    
    result = {
        'moneyline_2way': {
            'home': home_win,
            'away': away_win,
            'home_odds': prob_to_american_odds(home_win),
            'away_odds': prob_to_american_odds(away_win)
        },
        'moneyline_3way': {
            'home': home_win - draw/2,
            'draw': draw,
            'away': away_win - draw/2,
            'home_odds': prob_to_american_odds(home_win - draw/2),
            'draw_odds': prob_to_american_odds(draw),
            'away_odds': prob_to_american_odds(away_win - draw/2)
        },
        'spread': spread_markets,
        'total': total_markets,
        'draw_and_over_half': draw_and_over_half,
        'estimated_spread': period_spread,
        'estimated_total': period_total
    }
    
    return result

@app.route('/api/predict-all-quarters', methods=['POST'])
def predict_all_quarters():
    """Main prediction endpoint for all quarters and halves"""
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
        
        # Call the predict method - suppress output
        with redirect_stdout(StringIO()):
            result = predictor.predict(spread, total, debug=False)
        
        # Calculate markets for each period
        # Handle both old and new all.py return formats
        markets = {}
        
        # Map period names to actual keys in result
        period_key_map = {
            'q1': 'q1_home_away',
            'q2': 'q2_home_away',
            'q3': 'q3_home_away',
            'q4': 'q4_home_away',
            'h1': 'h1_home_away',
            'h2': 'h2_home_away',
            'full': 'full_game_home_away'  # Note: uses 'full_game' not 'full'
        }
        
        for period, key in period_key_map.items():
            if key in result:
                predictions = result[key]
                markets[period] = calculate_markets_for_period(predictions, spread, total, period)
        
        # Calculate highest scoring quarter for full game
        if all(key in result for key in ['q1_home_away', 'q2_home_away', 'q3_home_away', 'q4_home_away']):
            highest_quarter_probs = calculate_highest_scoring_quarter(
                result['q1_home_away'],
                result['q2_home_away'],
                result['q3_home_away'],
                result['q4_home_away']
            )
            
            # Add to full game markets
            if 'full' in markets:
                def prob_to_american_odds(p):
                    if p <= 0 or p >= 1:
                        return 0
                    if p >= 0.5:
                        return int(-100 * p / (1 - p))
                    else:
                        return int(100 * (1 - p) / p)
                
                markets['full']['highest_scoring_quarter'] = {
                    'q1': {
                        'prob': highest_quarter_probs['q1'],
                        'odds': prob_to_american_odds(highest_quarter_probs['q1'])
                    },
                    'q2': {
                        'prob': highest_quarter_probs['q2'],
                        'odds': prob_to_american_odds(highest_quarter_probs['q2'])
                    },
                    'q3': {
                        'prob': highest_quarter_probs['q3'],
                        'odds': prob_to_american_odds(highest_quarter_probs['q3'])
                    },
                    'q4': {
                        'prob': highest_quarter_probs['q4'],
                        'odds': prob_to_american_odds(highest_quarter_probs['q4'])
                    },
                    'tie': {
                        'prob': highest_quarter_probs['tie'],
                        'odds': prob_to_american_odds(highest_quarter_probs['tie'])
                    }
                }
        
        # Get top predictions for each period (10 for each)
        top_predictions = {}
        for period, key in period_key_map.items():
            if key in result:
                predictions = result[key]
                sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
                top_predictions[period] = [
                    {'score': score, 'probability': float(prob)}
                    for score, prob in sorted_preds
                ]
        
        return jsonify({
            'predictions': top_predictions,
            'markets': markets,
            'game_info': {
                'spread': spread,
                'total': total,
                'home_team': home_team,
                'away_team': away_team
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal prediction error', 'details': str(e)}), 500

@app.route('/api/calculate-parlay', methods=['POST'])
def calculate_parlay():
    """Calculate SGP probability using Monte Carlo simulation"""
    start_time = time.time()
    
    if not predictor_initialized:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        data = request.get_json()
        spread = float(data.get('spread'))
        total = float(data.get('total'))
        selections = data.get('selections', [])
        
        if not selections:
            return jsonify({'error': 'No selections provided'}), 400
        
        # Get predictions for all periods (suppressed output)
        with redirect_stdout(StringIO()):
            result = predictor.predict(spread, total, debug=False)
        
        # Run Monte Carlo simulation
        parlay_prob = calculate_sgp_monte_carlo(result, selections, num_simulations=100000)
        
        # Calculate fair odds
        if parlay_prob <= 0:
            parlay_prob = 1e-9
        
        decimal_odds = 1 / parlay_prob
        if parlay_prob >= 0.5:
            fair_odds = int(-100 * parlay_prob / (1 - parlay_prob))
        else:
            fair_odds = int(100 * (1 - parlay_prob) / parlay_prob)
        
        end_time = time.time()
        
        return jsonify({
            'probability': float(parlay_prob),
            'fair_odds': int(fair_odds),
            'decimal_odds': round(decimal_odds, 2),
            'selections_count': len(selections),
            'simulation_time_ms': round((end_time - start_time) * 1000, 2)
        })
        
    except Exception as e:
        logger.error(f"Parlay calculation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Status page"""
    status = 'Model initialized and ready' if predictor_initialized else f'Model initialization failed: {initialization_error}'
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CFB All Quarters Prediction API</title>
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
            <h1>CFB All Quarters Prediction API</h1>
            <div class="status {'success' if predictor_initialized else 'error'}">
                <strong>Status:</strong> {status}
            </div>
            <h3>Endpoints:</h3>
            <ul>
                <li><code>GET /api/health</code> - Health check</li>
                <li><code>GET /api/model-status</code> - Model status</li>
                <li><code>POST /api/predict-all-quarters</code> - Generate predictions for all periods</li>
                <li><code>POST /api/calculate-parlay</code> - Calculate SGP across periods</li>
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

def main():
    """Main server startup function"""
    print("="*60)
    print("CFB All Quarters Prediction API Server")
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