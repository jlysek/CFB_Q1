#!/usr/bin/env python3
"""
Q1 Weekly Predictor - Specific Score Probabilities
Target Scores: 7-7, 7-6/6-7, 14-14
Filters: Abs(Spread) <= 10 AND Total >= 41
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import cfbd
from cfbd.models.division_classification import DivisionClassification
from cfbd.models.season_type import SeasonType
import warnings
warnings.filterwarnings('ignore')

try:
    from all import CFBAllQuartersPredictor
except ImportError:
    print("ERROR: Could not import all.py")
    sys.exit(1)

load_dotenv()

def get_upcoming_games_from_cfbd(season, week, api_key):
    """
    Fetch upcoming games directly from CFBD API
    Matches the pattern used in scraper.py
    """
    print(f"Fetching games for Season {season}, Week {week} from CFBD API...")
    
    # Initialize CFBD API client
    configuration = cfbd.Configuration(access_token=api_key)
    api_client = cfbd.ApiClient(configuration)
    games_api = cfbd.GamesApi(api_client)
    betting_api = cfbd.BettingApi(api_client)
    
    try:
        # Fetch games for the week
        games = games_api.get_games(
            year=season,
            week=week,
            season_type=SeasonType.REGULAR,
            classification=DivisionClassification.FBS
        )
        print(f"  Found {len(games)} games from API")
    except Exception as e:
        print(f"Error fetching games: {e}")
        return pd.DataFrame()
    
    try:
        # Fetch betting lines for the week
        betting_lines = betting_api.get_lines(
            year=season,
            week=week,
            season_type=SeasonType.REGULAR
        )
        print(f"  Found betting lines for {len(betting_lines) if betting_lines else 0} games")
    except Exception as e:
        print(f"Error fetching betting lines: {e}")
        betting_lines = []
    
    # Map betting lines to game_id
    betting_data = {}
    if betting_lines:
        for bet_game in betting_lines:
            if hasattr(bet_game, 'id') and bet_game.lines:
                # Prefer consensus, then major books
                best_line = None
                preferred_providers = ['consensus', 'Bovada', 'DraftKings', 'FanDuel', 'ESPN BET']
                
                for provider in preferred_providers:
                    for line in bet_game.lines:
                        if hasattr(line, 'provider') and line.provider == provider:
                            best_line = line
                            break
                    if best_line:
                        break
                
                if not best_line and bet_game.lines:
                    best_line = bet_game.lines[0]
                
                if best_line:
                    spread = getattr(best_line, 'spread', None)
                    total = getattr(best_line, 'over_under', None)
                    
                    if spread is not None and total is not None:
                        betting_data[bet_game.id] = {
                            'spread': float(spread),
                            'total': float(total),
                            'provider': getattr(best_line, 'provider', 'Unknown')
                        }
    
    # Process games and apply filters
    processed_games = []
    skipped_count = 0
    
    for game in games:
        game_id = game.id
        
        # Check if we have betting data for this game
        if game_id in betting_data:
            bet_info = betting_data[game_id]
            spread = bet_info['spread']
            total = bet_info['total']
            
            # Apply filters: Skip wide spreads or low totals
            if abs(spread) > 6.5 or total < 50:
                skipped_count += 1
                continue
            
            # Build game record
            processed_games.append({
                'game_id': game_id,
                'start_date': getattr(game, 'start_date', None),
                'home_team': game.home_team,
                'away_team': game.away_team,
                'Spread': spread,
                'Total': total
            })
    
    df = pd.DataFrame(processed_games)
    if not df.empty and 'start_date' in df.columns:
        df['start_date'] = df['start_date'].fillna('')
        df = df.sort_values('start_date')
    
    print(f"Found {len(df)} qualifying games (skipped {skipped_count} via filters).")
    return df

def extract_target_scores(q1_dist):
    """
    Extract probabilities for specific target scores:
    - 7-7 (exact tie at 7)
    - 7-6 or 6-7 (combined probability)
    - 14-14 (exact tie at 14)
    """
    prob_7_7 = 0.0
    prob_7_6_or_6_7 = 0.0
    prob_14_14 = 0.0
    
    for score, prob in q1_dist.items():
        try:
            h, a = map(int, score.split('-'))
            
            # Check for 7-7
            if h == 7 and a == 7:
                prob_7_7 += prob
            
            # Check for 7-6 or 6-7
            if (h == 7 and a == 6) or (h == 6 and a == 7):
                prob_7_6_or_6_7 += prob
            
            # Check for 14-14
            if h == 14 and a == 14:
                prob_14_14 += prob
                
        except:
            continue
    
    return prob_7_7, prob_7_6_or_6_7, prob_14_14

def main():
    # Get API key from environment
    api_key = os.getenv('CFBD_API_KEY')
    if not api_key:
        print("ERROR: CFBD_API_KEY not found in environment variables")
        print("Please set CFBD_API_KEY in your .env file")
        sys.exit(1)
    
    # Database config for historical data loading
    db_config = {
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'cfb')
    }
    
    # Parse command line arguments
    season = 2025
    week = 13
    if len(sys.argv) > 1: season = int(sys.argv[1])
    if len(sys.argv) > 2: week = int(sys.argv[2])
    
    print("="*60)
    print(f"CFB Q1 SCORE PREDICTIONS: {season} Week {week}")
    print("Target Scores: 7-7, 7-6/6-7, 14-14")
    print("Filters: Spread<=6.5, Total>=49")
    print("="*60)
    
    # Fetch games from CFBD API
    games_df = get_upcoming_games_from_cfbd(season, week, api_key)
    if games_df.empty:
        print("\nNo games found matching criteria.")
        return

    print(f"\nInitializing model and processing {len(games_df)} games...")
    
    # Suppress noisy initialization output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        predictor = CFBAllQuartersPredictor(db_config)
        if not predictor.load_historical_data(): 
            sys.stdout = original_stdout
            print("ERROR: Failed to load historical data")
            return
        predictor.calculate_empirical_distribution()
        predictor.fit_model()
    finally:
        sys.stdout = original_stdout

    print("Model ready. Running predictions...\n")
    
    results = []
    for idx, game in games_df.iterrows():
        game_num = idx + 1
        
        # Format Date
        try:
            if game['start_date']:
                dt = datetime.fromisoformat(str(game['start_date']).replace('Z', '+00:00'))
                date_str = dt.strftime('%a %m/%d %I:%M %p')
            else:
                date_str = "TBD"
        except:
            date_str = str(game['start_date'])
        
        # Format game matchup
        game_str = f"{game['away_team']} @ {game['home_team']}"
        
        print(f"[{game_num}/{len(games_df)}] {game_str}")
        print(f"   Line: {game['Spread']:.1f} / {game['Total']:.1f}", end=" ... ", flush=True)

        try:
            # Run full prediction with calibration
            res = predictor.predict(game['Spread'], game['Total'], debug=False, n_sims=5000)
            
            # Extract target score probabilities
            prob_7_7, prob_7_6_or_6_7, prob_14_14 = extract_target_scores(res['q1'])
            
            results.append({
                'Date': date_str,
                'Game': game_str,
                'Prob_7-7': round(prob_7_7, 4),
                'Prob_7-6_or_6-7': round(prob_7_6_or_6_7, 4),
                'Prob_14-14': round(prob_14_14, 4)
            })
            
            print(f"7-7: {prob_7_7*100:.1f}% | 7-6/6-7: {prob_7_6_or_6_7*100:.1f}% | 14-14: {prob_14_14*100:.1f}%")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        
        outfile = f'q1_scores_week{week}_{season}.csv'
        results_df.to_csv(outfile, index=False)
        
        print("\n" + "="*60)
        print(f"COMPLETE: {len(results)} games analyzed")
        print("="*60)
        print(f"\nResults saved to: {outfile}")
        print("="*60)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()