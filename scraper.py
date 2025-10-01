import mysql.connector
import cfbd
from cfbd.models.division_classification import DivisionClassification
from cfbd.models.season_type import SeasonType
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CFBDataUpdater:
    """
    Updates CFB games and quarter scoring data from College Football Data API
    
    This class handles:
    1. Fetching completed FBS games with scores and betting lines
    2. Storing game data in cfb.games table
    3. Storing quarter-by-quarter scoring in cfb.quarter_scoring table
    4. Managing incremental updates to avoid duplicating data
    5. Handling API rate limits and error recovery
    """
    
    def __init__(self, api_key: str, db_config: Dict):
        """
        Initialize with CFBD API key and MySQL database configuration
        
        Args:
            api_key: College Football Data API key from https://collegefootballdata.com
            db_config: MySQL connection parameters
                      Example: {'host': '127.0.0.1', 'user': 'root', 'password': '', 'database': 'cfb'}
        """
        self.api_key = api_key
        self.db_config = db_config
        
        # Initialize CFBD API client with proper authentication
        # Using the same method as your working scripts
        self.configuration = cfbd.Configuration(access_token=api_key)
        self.api_client = cfbd.ApiClient(self.configuration)
        self.games_api = cfbd.GamesApi(self.api_client)
        self.betting_api = cfbd.BettingApi(self.api_client)
        
        # Track API calls for rate limiting
        self.api_calls_made = 0
        self.last_api_call_time = time.time()
        
        # Connect to database
        self.connect_to_database()
        
        # Test API connection immediately after initialization
        print("Testing API connection...")
        if not self.test_api_connection():
            raise Exception("API authentication failed - please check your API key")
    
    def test_api_connection(self):
        """
        Test API connection with a simple request to verify authentication
        """
        try:
            # Try a simple API call to test authentication
            test_games = self.games_api.get_games(year=2024, week=1)
            print(f"API connection successful! Found {len(test_games) if test_games else 0} games in test call")
            return True
        except Exception as e:
            print(f"API connection failed: {e}")
            return False
    
    def connect_to_database(self):
        """
        Establish MySQL database connection and verify table structures
        """
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("Successfully connected to MySQL database")
            
            # Verify required tables exist
            self.verify_table_structure()
            
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            raise
    
    def verify_table_structure(self):
        """
        Check if required tables exist and have proper structure
        Creates tables if they don't exist
        """
        print("Verifying database table structure...")
        
        # Check cfb.games table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id INT PRIMARY KEY,
                season INT NOT NULL,
                week INT NOT NULL,
                home_team VARCHAR(100) NOT NULL,
                away_team VARCHAR(100) NOT NULL,
                home_points INT,
                away_points INT,
                neutral_site TINYINT(1) DEFAULT 0,
                conference_game TINYINT(1) DEFAULT 0,
                start_date DATETIME,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                home_classification VARCHAR(10),
                away_classification VARCHAR(10),
                pregame_spread FLOAT,
                pregame_total FLOAT,
                home_ml_prob FLOAT,
                away_ml_prob FLOAT,
                betting_provider VARCHAR(50),
                INDEX idx_season_week (season, week),
                INDEX idx_game_id (game_id)
            )
        """)
        
        # Check cfb.quarter_scoring table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS quarter_scoring (
                id INT AUTO_INCREMENT PRIMARY KEY,
                game_id INT NOT NULL,
                quarter INT NOT NULL,
                home_score INT NOT NULL,
                away_score INT NOT NULL,
                total_score INT GENERATED ALWAYS AS (home_score + away_score) STORED,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_game_quarter (game_id, quarter),
                INDEX idx_game_id (game_id)
            )
        """)
        
        self.connection.commit()
        print("Database tables verified/created successfully")
    
    def rate_limit_check(self):
        """
        Implement API rate limiting to respect CFBD limits
        CFBD API allows 200 calls per minute
        """
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        # Reset counter every minute
        if time_since_last_call > 60:
            self.api_calls_made = 0
            self.last_api_call_time = current_time
        
        # If approaching rate limit, wait
        if self.api_calls_made >= 180:  # Conservative limit
            wait_time = 60 - time_since_last_call
            if wait_time > 0:
                print(f"Rate limit approaching, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.api_calls_made = 0
                self.last_api_call_time = time.time()
        
        self.api_calls_made += 1
    
    def is_fbs_team(self, team_name: str, conference: str = None) -> bool:
        """
        Determine if a team is FBS based on team name and conference
        """
        fbs_conferences = {
            'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12', 'Pac-10',
            'American Athletic', 'American', 'AAC',
            'Conference USA', 'C-USA', 'CUSA',
            'Mid-American', 'MAC',
            'Mountain West', 'MWC',
            'Sun Belt',
            'FBS Independents', 'Independent'
        }
        
        if conference:
            for fbs_conf in fbs_conferences:
                if fbs_conf.lower() in conference.lower():
                    return True
        
        fbs_teams = {
            'Notre Dame', 'Army', 'Navy', 'BYU', 'Liberty', 'New Mexico State',
            'UConn', 'UMass'
        }
        
        return team_name in fbs_teams
    
    def get_existing_games(self, season: int, week: int = None) -> set:
        """Get set of game_ids that already exist in database"""
        if week is not None:
            self.cursor.execute("""
                SELECT game_id FROM games 
                WHERE season = %s AND week = %s
            """, (season, week))
        else:
            self.cursor.execute("""
                SELECT game_id FROM games 
                WHERE season = %s
            """, (season,))
        
        existing_games = {row[0] for row in self.cursor.fetchall()}
        return existing_games
    
    def moneyline_to_probability(self, home_ml: Optional[float], away_ml: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """Convert American moneyline odds to implied probabilities"""
        if not home_ml or not away_ml:
            return None, None
        
        try:
            if home_ml > 0:
                home_implied = 100 / (home_ml + 100)
            else:
                home_implied = abs(home_ml) / (abs(home_ml) + 100)
            
            if away_ml > 0:
                away_implied = 100 / (away_ml + 100)
            else:
                away_implied = abs(away_ml) / (abs(away_ml) + 100)
            
            total_implied = home_implied + away_implied
            if total_implied > 0:
                home_prob = home_implied / total_implied
                away_prob = away_implied / total_implied
                return home_prob, away_prob
            else:
                return None, None
                
        except (ValueError, ZeroDivisionError):
            return None, None
    
    def fetch_betting_lines(self, season: int, week: int) -> Dict[int, Dict]:
        """Fetch betting lines for a specific season and week"""
        self.rate_limit_check()
        
        try:
            print(f"  Fetching betting lines for {season} week {week}...")
            
            betting_lines = self.betting_api.get_lines(
                year=season,
                week=week,
                season_type=SeasonType.REGULAR
            )
            
            betting_data = {}
            
            if betting_lines:
                for bet_game in betting_lines:
                    if hasattr(bet_game, 'id') and bet_game.lines:
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
                            home_ml = getattr(best_line, 'home_moneyline', None)
                            away_ml = getattr(best_line, 'away_moneyline', None)
                            provider = getattr(best_line, 'provider', 'Unknown')
                            
                            home_ml_prob, away_ml_prob = self.moneyline_to_probability(home_ml, away_ml)
                            
                            betting_data[bet_game.id] = {
                                'spread': spread,
                                'total': total,
                                'home_ml_prob': home_ml_prob,
                                'away_ml_prob': away_ml_prob,
                                'provider': provider
                            }
            
            print(f"  Found betting lines for {len(betting_data)} games")
            return betting_data
            
        except Exception as e:
            print(f"  Error fetching betting lines: {e}")
            return {}
    
    def fetch_games_for_week(self, season: int, week: int) -> List[Dict]:
        """Fetch completed games for a specific season and week"""
        self.rate_limit_check()
        
        try:
            print(f"Fetching games for {season} week {week}...")
            
            games = self.games_api.get_games(
                year=season,
                week=week,
                season_type=SeasonType.REGULAR,
                classification=DivisionClassification.FBS
            )
            
            completed_games = []
            
            for game in games:
                if (hasattr(game, 'home_points') and hasattr(game, 'away_points') and 
                    game.home_points is not None and game.away_points is not None):
                    
                    game_data = {
                        'game_id': game.id,
                        'season': game.season,
                        'week': game.week,
                        'home_team': game.home_team,
                        'away_team': game.away_team,
                        'home_points': game.home_points,
                        'away_points': game.away_points,
                        'neutral_site': getattr(game, 'neutral_site', False),
                        'conference_game': getattr(game, 'conference_game', False),
                        'start_date': getattr(game, 'start_date', None),
                        'home_conference': getattr(game, 'home_conference', None),
                        'away_conference': getattr(game, 'away_conference', None)
                    }
                    
                    home_is_fbs = self.is_fbs_team(game.home_team, game_data['home_conference'])
                    away_is_fbs = self.is_fbs_team(game.away_team, game_data['away_conference'])
                    
                    game_data['home_classification'] = 'fbs' if home_is_fbs else 'fcs'
                    game_data['away_classification'] = 'fbs' if away_is_fbs else 'fcs'
                    
                    if home_is_fbs or away_is_fbs:
                        quarter_scores = []
                        if (hasattr(game, 'home_line_scores') and hasattr(game, 'away_line_scores') and
                            game.home_line_scores and game.away_line_scores):
                            
                            for quarter, (home_q, away_q) in enumerate(zip(game.home_line_scores, game.away_line_scores), 1):
                                if quarter <= 4:
                                    quarter_scores.append({
                                        'quarter': quarter,
                                        'home_score': int(home_q),
                                        'away_score': int(away_q)
                                    })
                        
                        game_data['quarter_scores'] = quarter_scores
                        completed_games.append(game_data)
            
            print(f"  Found {len(completed_games)} completed FBS games")
            return completed_games
            
        except Exception as e:
            print(f"  Error fetching games: {e}")
            return []
    
    def store_game_data(self, game_data: Dict, betting_data: Dict = None):
        """Store game data and quarter scores in database"""
        try:
            spread = None
            total = None
            home_ml_prob = None
            away_ml_prob = None
            provider = None
            
            if betting_data and game_data['game_id'] in betting_data:
                bet_info = betting_data[game_data['game_id']]
                spread = bet_info.get('spread')
                total = bet_info.get('total')
                home_ml_prob = bet_info.get('home_ml_prob')
                away_ml_prob = bet_info.get('away_ml_prob')
                provider = bet_info.get('provider')
            
            self.cursor.execute("""
                INSERT INTO games (
                    game_id, season, week, home_team, away_team,
                    home_points, away_points, neutral_site, conference_game, start_date,
                    home_classification, away_classification,
                    pregame_spread, pregame_total, home_ml_prob, away_ml_prob, betting_provider
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    home_points = VALUES(home_points),
                    away_points = VALUES(away_points),
                    home_classification = VALUES(home_classification),
                    away_classification = VALUES(away_classification),
                    pregame_spread = COALESCE(VALUES(pregame_spread), pregame_spread),
                    pregame_total = COALESCE(VALUES(pregame_total), pregame_total),
                    home_ml_prob = COALESCE(VALUES(home_ml_prob), home_ml_prob),
                    away_ml_prob = COALESCE(VALUES(away_ml_prob), away_ml_prob),
                    betting_provider = COALESCE(VALUES(betting_provider), betting_provider)
            """, (
                game_data['game_id'], game_data['season'], game_data['week'],
                game_data['home_team'], game_data['away_team'],
                game_data['home_points'], game_data['away_points'],
                game_data['neutral_site'], game_data['conference_game'], game_data['start_date'],
                game_data['home_classification'], game_data['away_classification'],
                spread, total, home_ml_prob, away_ml_prob, provider
            ))
            
            if game_data.get('quarter_scores'):
                for quarter_data in game_data['quarter_scores']:
                    self.cursor.execute("""
                        INSERT INTO quarter_scoring (
                            game_id, quarter, home_score, away_score
                        ) VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            home_score = VALUES(home_score),
                            away_score = VALUES(away_score)
                    """, (
                        game_data['game_id'],
                        quarter_data['quarter'],
                        quarter_data['home_score'],
                        quarter_data['away_score']
                    ))
            
        except mysql.connector.Error as err:
            print(f"Database error storing game {game_data['game_id']}: {err}")
            raise
    
    def update_season_data(self, season: int, specific_week: int = None):
        """Update data for a complete season or specific week"""
        print(f"Updating data for {season} season" + (f" week {specific_week}" if specific_week else ""))
        
        if specific_week is not None:
            week_range = [specific_week]
        else:
            if season == 2025:
                week_range = list(range(0, 6))
            else:
                week_range = list(range(1, 16))
        
        games_processed = 0
        games_stored = 0
        
        for week in week_range:
            try:
                print(f"\nProcessing {season} Week {week}...")
                
                existing_games = self.get_existing_games(season, week)
                games = self.fetch_games_for_week(season, week)
                
                if not games:
                    print(f"  No completed games found for week {week}")
                    continue
                
                betting_data = self.fetch_betting_lines(season, week)
                
                new_games = 0
                for game_data in games:
                    games_processed += 1
                    
                    if game_data['game_id'] not in existing_games:
                        new_games += 1
                    
                    self.store_game_data(game_data, betting_data)
                    games_stored += 1
                
                self.connection.commit()
                print(f"  Week {week}: {len(games)} games found, {new_games} new games stored")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error processing week {week}: {e}")
                continue
        
        print(f"\nSeason update complete:")
        print(f"  Total games processed: {games_processed}")
        print(f"  Total games stored/updated: {games_stored}")
        print(f"  API calls made: {self.api_calls_made}")
    
    def update_current_season(self):
        """Update data for current season (2025)"""
        current_year = 2025
        
        print("Updating current season data...")
        
        self.cursor.execute("""
            SELECT DISTINCT week FROM games 
            WHERE season = %s 
            ORDER BY week DESC
        """, (current_year,))
        
        existing_weeks = {row[0] for row in self.cursor.fetchall()}
        max_week_to_check = 6
        
        weeks_to_update = list(range(0, max_week_to_check))
        print(f"Checking weeks: {weeks_to_update}")
        
        for week in weeks_to_update:
            self.update_season_data(current_year, week)
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")

def main():
    """Main execution function"""
    # Use your actual API key
    API_KEY = "meWCW794Y9rcfCIbk8aRXkLpI35TPBpvf8uzKg9A1yKXJBryMKqx80pETU79J4xT"
    
    DB_CONFIG = {
        'host': '127.0.0.1',
        'port': 3306,
        'user': 'root',
        'password': '',
        'database': 'cfb'
    }
    
    try:
        print("Initializing CFB Data Updater...")
        updater = CFBDataUpdater(API_KEY, DB_CONFIG)
        
        print("\n" + "="*60)
        updater.update_current_season()
        
        print("\n" + "="*60)
        print("UPDATE COMPLETE")
        print("="*60)
        
        # Show summary statistics
        updater.cursor.execute("SELECT COUNT(*) FROM games")
        total_games = updater.cursor.fetchone()[0]
        
        updater.cursor.execute("SELECT COUNT(*) FROM quarter_scoring")
        total_quarters = updater.cursor.fetchone()[0]
        
        updater.cursor.execute("""
            SELECT COUNT(*) FROM games 
            WHERE pregame_spread IS NOT NULL
        """)
        games_with_betting = updater.cursor.fetchone()[0]
        
        print(f"Database Summary:")
        print(f"  Total games: {total_games:,}")
        print(f"  Quarter records: {total_quarters:,}")
        print(f"  Games with betting data: {games_with_betting:,}")
        print(f"  Betting data coverage: {games_with_betting/max(total_games,1):.1%}")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        try:
            updater.close_connection()
        except:
            pass

if __name__ == "__main__":
    main()