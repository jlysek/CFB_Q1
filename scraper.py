import os
import sys
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import cfbd
import pandas as pd

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# CFBD API configuration
CFBD_API_KEY = os.getenv('CFBD_API_KEY')

def connect_to_database():
    """
    Establish connection to MySQL database
    Returns connection object or None if failed
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            print("Successfully connected to database")
            return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

def get_cfbd_configuration():
    """
    Configure CFBD API client
    Returns configured API client instances
    """
    configuration = cfbd.Configuration(access_token=CFBD_API_KEY)
    api_client = cfbd.ApiClient(configuration)
    games_api = cfbd.GamesApi(api_client)
    betting_api = cfbd.BettingApi(api_client)
    
    return games_api, betting_api

def cleanup_duplicate_quarter_records(cursor, conn):
    """
    Remove duplicate quarter_scoring records, keeping only the earliest created_at entry
    for each unique (game_id, quarter) combination.
    """
    print("\n" + "="*80)
    print("CLEANING UP DUPLICATE QUARTER RECORDS")
    print("="*80)
    
    # Find duplicates
    cursor.execute("""
        SELECT game_id, quarter, COUNT(*) as cnt
        FROM quarter_scoring
        GROUP BY game_id, quarter
        HAVING COUNT(*) > 1
    """)
    duplicates = cursor.fetchall()
    
    if not duplicates:
        print("No duplicate records found")
        return
    
    print(f"Found {len(duplicates)} duplicate (game_id, quarter) combinations")
    
    total_deleted = 0
    
    # For each duplicate, keep only the earliest record
    for game_id, quarter, count in duplicates:
        # Get all IDs for this game_id, quarter combination, ordered by created_at
        cursor.execute("""
            SELECT id
            FROM quarter_scoring
            WHERE game_id = %s AND quarter = %s
            ORDER BY created_at ASC
        """, (game_id, quarter))
        
        ids = [row[0] for row in cursor.fetchall()]
        
        # Keep the first (earliest) ID, delete the rest
        ids_to_delete = ids[1:]
        
        if ids_to_delete:
            placeholders = ','.join(['%s'] * len(ids_to_delete))
            cursor.execute(f"""
                DELETE FROM quarter_scoring
                WHERE id IN ({placeholders})
            """, ids_to_delete)
            
            deleted = cursor.rowcount
            total_deleted += deleted
    
    conn.commit()
    print(f"Deleted {total_deleted} duplicate records")
    print("="*80 + "\n")

def backfill_overtime_periods(cursor, conn):
    """
    Backfill period 5 (overtime) records for games that went to OT.
    Fixes the calculation to properly compute OT scores.
    """
    print("\n" + "="*80)
    print("BACKFILLING OVERTIME PERIODS")
    print("="*80)
    
    # Find games where final score != sum of Q1-Q4 scores
    cursor.execute("""
        SELECT 
            g.game_id,
            g.home_team,
            g.away_team,
            g.home_points as final_home,
            g.away_points as final_away,
            COALESCE(SUM(CASE WHEN qs.quarter <= 4 THEN qs.home_score ELSE 0 END), 0) as reg_home,
            COALESCE(SUM(CASE WHEN qs.quarter <= 4 THEN qs.away_score ELSE 0 END), 0) as reg_away
        FROM cfb.games g
        LEFT JOIN quarter_scoring qs ON g.game_id = qs.game_id
        GROUP BY g.game_id, g.home_team, g.away_team, g.home_points, g.away_points
        HAVING 
            g.home_points != COALESCE(SUM(CASE WHEN qs.quarter <= 4 THEN qs.home_score ELSE 0 END), 0)
            OR g.away_points != COALESCE(SUM(CASE WHEN qs.quarter <= 4 THEN qs.away_score ELSE 0 END), 0)
    """)
    
    ot_games = cursor.fetchall()
    print(f"Found {len(ot_games)} games with score discrepancies (potential OT)")
    
    records_added = 0
    
    for game_id, home_team, away_team, final_home, final_away, reg_home, reg_away in ot_games:
        # Calculate OT scores: OT = Final - Regulation
        ot_home = final_home - reg_home
        ot_away = final_away - reg_away
        ot_total = ot_home + ot_away
        
        # Validate that OT scores are positive (sanity check)
        if ot_home < 0 or ot_away < 0:
            print(f"  Skipping game {game_id}: Invalid OT calculation ({ot_home}, {ot_away})")
            continue
        
        # Check if period 5 already exists
        cursor.execute("""
            SELECT COUNT(*) FROM quarter_scoring
            WHERE game_id = %s AND quarter = 5
        """, (game_id,))
        
        if cursor.fetchone()[0] > 0:
            continue
        
        # Insert period 5 record
        cursor.execute("""
            INSERT INTO quarter_scoring (game_id, quarter, home_score, away_score, total_score, created_at)
            VALUES (%s, 5, %s, %s, %s, NOW())
        """, (game_id, ot_home, ot_away, ot_total))
        
        records_added += 1
        print(f"  Game {game_id} ({home_team} vs {away_team}): Reg {reg_home}-{reg_away}, Final {final_home}-{final_away}, OT {ot_home}-{ot_away}")
    
    conn.commit()
    
    print("\n" + "="*80)
    print(f"OVERTIME BACKFILL COMPLETE: {records_added} period 5 records added")
    print("="*80 + "\n")

def get_most_recent_game_date(cursor):
    """
    Get the most recent game date in the database
    Returns datetime object or None
    """
    cursor.execute("""
        SELECT start_date, season, week 
        FROM cfb.games 
        ORDER BY start_date DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result and result[0]:
        return result[0], result[1], result[2]
    return None, None, None

def fetch_games_for_week(games_api, season, week):
    """
    Fetch all FBS games for a given season and week
    Returns list of games
    """
    try:
        games = games_api.get_games(
            year=season,
            week=week,
            season_type='both',
            classification='fbs'
        )
        return games
    except Exception as e:
        print(f"  Error fetching games: {e}")
        return []

def fetch_betting_lines(betting_api, season, week):
    """
    Fetch betting lines for a given season and week
    Returns list of betting lines
    """
    try:
        lines = betting_api.get_lines(
            year=season,
            week=week,
            season_type='both'
        )
        return lines
    except Exception as e:
        print(f"  Error fetching betting lines: {e}")
        return []

def aggregate_betting_lines(lines):
    """
    Aggregate betting lines across providers, taking median values
    Returns dict mapping game_id to betting data
    """
    game_lines = {}
    
    for line in lines:
        game_id = line.id
        
        if game_id not in game_lines:
            game_lines[game_id] = {
                'spreads': [],
                'totals': [],
                'home_ml': [],
                'away_ml': [],
                'providers': []
            }
        
        if line.lines:
            for provider_line in line.lines:
                game_lines[game_id]['providers'].append(provider_line.provider)
                
                if hasattr(provider_line, 'spread') and provider_line.spread:
                    game_lines[game_id]['spreads'].append(float(provider_line.spread))
                
                if hasattr(provider_line, 'over_under') and provider_line.over_under:
                    game_lines[game_id]['totals'].append(float(provider_line.over_under))
                
                if hasattr(provider_line, 'home_moneyline') and provider_line.home_moneyline:
                    game_lines[game_id]['home_ml'].append(int(provider_line.home_moneyline))
                
                if hasattr(provider_line, 'away_moneyline') and provider_line.away_moneyline:
                    game_lines[game_id]['away_ml'].append(int(provider_line.away_moneyline))
    
    # Calculate medians
    aggregated = {}
    for game_id, data in game_lines.items():
        aggregated[game_id] = {
            'spread': float(pd.Series(data['spreads']).median()) if data['spreads'] else None,
            'total': float(pd.Series(data['totals']).median()) if data['totals'] else None,
            'home_ml': float(pd.Series(data['home_ml']).median()) if data['home_ml'] else None,
            'away_ml': float(pd.Series(data['away_ml']).median()) if data['away_ml'] else None,
            'provider': 'Aggregated'
        }
        
        # Convert moneylines to probabilities
        if aggregated[game_id]['home_ml'] is not None:
            home_ml = aggregated[game_id]['home_ml']
            if home_ml < 0:
                aggregated[game_id]['home_ml_prob'] = abs(home_ml) / (abs(home_ml) + 100)
            else:
                aggregated[game_id]['home_ml_prob'] = 100 / (home_ml + 100)
        else:
            aggregated[game_id]['home_ml_prob'] = None
        
        if aggregated[game_id]['away_ml'] is not None:
            away_ml = aggregated[game_id]['away_ml']
            if away_ml < 0:
                aggregated[game_id]['away_ml_prob'] = abs(away_ml) / (abs(away_ml) + 100)
            else:
                aggregated[game_id]['away_ml_prob'] = 100 / (away_ml + 100)
        else:
            aggregated[game_id]['away_ml_prob'] = None
    
    return aggregated

def game_exists(cursor, game_id):
    """
    Check if a game already exists in the database
    Returns True if exists, False otherwise
    """
    cursor.execute("SELECT COUNT(*) FROM cfb.games WHERE game_id = %s", (game_id,))
    return cursor.fetchone()[0] > 0

def insert_game(cursor, game, betting_data):
    """
    Insert a game into the cfb.games table
    """
    spread = betting_data.get('spread') if betting_data else None
    total = betting_data.get('total') if betting_data else None
    home_ml_prob = betting_data.get('home_ml_prob') if betting_data else None
    away_ml_prob = betting_data.get('away_ml_prob') if betting_data else None
    provider = betting_data.get('provider') if betting_data else None
    
    cursor.execute("""
        INSERT INTO cfb.games 
        (game_id, season, week, home_team, away_team, home_points, away_points, 
         neutral_site, conference_game, start_date, created_at, home_classification, 
         away_classification, pregame_spread, pregame_total, home_ml_prob, 
         away_ml_prob, betting_provider)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s)
    """, (
        game.id,
        game.season,
        game.week,
        game.home_team,
        game.away_team,
        game.home_points or 0,
        game.away_points or 0,
        1 if game.neutral_site else 0,
        1 if game.conference_game else 0,
        game.start_date,
        game.home_classification or 'fbs',
        game.away_classification or 'fbs',
        spread,
        total,
        home_ml_prob,
        away_ml_prob,
        provider
    ))

def fetch_quarter_scores(games_api, game_id):
    """
    Fetch quarter scores for a specific game
    Returns list of quarter scores
    """
    try:
        team_stats = games_api.get_team_game_stats(game_id=game_id)
        
        quarter_scores = {}
        
        for team_stat in team_stats:
            team = team_stat.school
            
            for stat in team_stat.stats:
                if stat.category == 'quarters':
                    quarters = stat.stat.split(',')
                    
                    for i, quarter_score in enumerate(quarters, start=1):
                        if i not in quarter_scores:
                            quarter_scores[i] = {'home': 0, 'away': 0}
                        
                        score = int(quarter_score.strip()) if quarter_score.strip() else 0
                        
                        if team == team_stats[0].school:
                            quarter_scores[i]['home'] = score
                        else:
                            quarter_scores[i]['away'] = score
        
        return quarter_scores
    
    except Exception as e:
        print(f"  Error fetching quarter scores for game {game_id}: {e}")
        return {}

def insert_quarter_scores(cursor, game_id, quarter_scores):
    """
    Insert quarter scores into quarter_scoring table
    """
    for quarter, scores in quarter_scores.items():
        if quarter > 4:
            continue
        
        # Check if already exists
        cursor.execute("""
            SELECT COUNT(*) FROM quarter_scoring 
            WHERE game_id = %s AND quarter = %s
        """, (game_id, quarter))
        
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO quarter_scoring 
                (game_id, quarter, home_score, away_score, total_score, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                game_id,
                quarter,
                scores['home'],
                scores['away'],
                scores['home'] + scores['away']
            ))

def run_incremental_update(cursor, conn, games_api, betting_api):
    """
    Run incremental update to fetch missing games
    """
    print("\n" + "="*80)
    print("INCREMENTAL UPDATE - FINDING MISSING GAMES")
    print("="*80)
    
    # Get most recent game in database
    most_recent_date, most_recent_season, most_recent_week = get_most_recent_game_date(cursor)
    
    if most_recent_date:
        print(f"Most recent game in database: {most_recent_date} (Season {most_recent_season}, Week {most_recent_week})")
    else:
        print("No games found in database, starting from 2014")
        most_recent_season = 2014
        most_recent_week = 1
    
    # Determine current date and season
    current_date = datetime.now()
    current_year = current_date.year
    
    # Determine if we're in football season (August-January)
    if current_date.month >= 8:
        current_season = current_year
    else:
        current_season = current_year - 1
    
    print(f"Current date: {current_date.strftime('%Y-%m-%d')}")
    
    # Calculate weeks to check (from most recent to current + buffer)
    weeks_to_check = []
    
    if most_recent_season == current_season:
        # Same season - check from most recent week to current week + 5
        for week in range(most_recent_week, min(most_recent_week + 20, 21)):
            weeks_to_check.append((most_recent_season, week))
    else:
        # Different seasons - need to check multiple seasons
        for season in range(most_recent_season, current_season + 1):
            if season == most_recent_season:
                start_week = most_recent_week
            else:
                start_week = 1
            
            for week in range(start_week, 21):
                weeks_to_check.append((season, week))
    
    print(f"Will check {len(weeks_to_check)} season-week combinations from {weeks_to_check[0][0]} to {weeks_to_check[-1][0]}")
    
    total_new_games = 0
    total_quarter_records = 0
    
    for season, week in weeks_to_check:
        print(f"\nChecking {season} Week {week}...")
        
        # Fetch games
        games = fetch_games_for_week(games_api, season, week)
        
        if not games:
            continue
        
        print(f"  Found {len(games)} FBS games for {season} Week {week}")
        
        # Fetch betting lines
        betting_lines = fetch_betting_lines(betting_api, season, week)
        aggregated_lines = aggregate_betting_lines(betting_lines)
        
        print(f"  Found betting lines for {len(aggregated_lines)} games (median across providers)")
        
        # Check which games are new
        new_games = []
        for game in games:
            if not game_exists(cursor, game.id):
                new_games.append(game)
        
        if not new_games:
            print(f"  All {len(games)} games already in database")
            continue
        
        print(f"  Found {len(new_games)} new games to add")
        
        # Insert new games
        for game in new_games:
            betting_data = aggregated_lines.get(game.id)
            insert_game(cursor, game, betting_data)
            
            # Fetch and insert quarter scores if game is complete
            if game.home_points is not None and game.away_points is not None:
                quarter_scores = fetch_quarter_scores(games_api, game.id)
                if quarter_scores:
                    insert_quarter_scores(cursor, game.id, quarter_scores)
                    total_quarter_records += len(quarter_scores)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
        
        conn.commit()
        total_new_games += len(new_games)
        
        # Delay between weeks
        time.sleep(0.5)
    
    print("\n" + "="*80)
    print("INCREMENTAL UPDATE COMPLETE")
    print("="*80)
    print(f"New games added: {total_new_games}")
    print(f"Quarter records added: {total_quarter_records}")
    print("="*80 + "\n")

def print_database_summary(cursor):
    """
    Print summary statistics about the database
    """
    # Date range
    cursor.execute("SELECT MIN(start_date), MAX(start_date) FROM cfb.games")
    min_date, max_date = cursor.fetchone()
    
    # Total games
    cursor.execute("SELECT COUNT(*) FROM cfb.games")
    total_games = cursor.fetchone()[0]
    
    # Quarter records
    cursor.execute("SELECT COUNT(*) FROM quarter_scoring")
    quarter_records = cursor.fetchone()[0]
    
    # Overtime periods
    cursor.execute("SELECT COUNT(*) FROM quarter_scoring WHERE quarter = 5")
    ot_periods = cursor.fetchone()[0]
    
    # Games with betting data
    cursor.execute("SELECT COUNT(*) FROM cfb.games WHERE pregame_spread IS NOT NULL")
    games_with_betting = cursor.fetchone()[0]
    
    print("Database Summary:")
    print(f"  Date range: {min_date} to {max_date}")
    print(f"  Total games: {total_games:,}")
    print(f"  Quarter records: {quarter_records:,}")
    print(f"  Overtime periods (period 5): {ot_periods}")
    print(f"  Games with betting data: {games_with_betting:,}")

def main():
    """
    Main execution function
    """
    # Connect to database
    conn = connect_to_database()
    if not conn:
        sys.exit(1)
    
    cursor = conn.cursor()
    
    # Configure CFBD API
    games_api, betting_api = get_cfbd_configuration()
    
    # STEP 1: Clean up any duplicate quarter records
    cleanup_duplicate_quarter_records(cursor, conn)
    
    # STEP 2: Backfill overtime periods
    backfill_overtime_periods(cursor, conn)
    
    # STEP 3: Run incremental update to fetch new games
    run_incremental_update(cursor, conn, games_api, betting_api)
    
    # Print summary
    print_database_summary(cursor)
    
    # Close connection
    cursor.close()
    conn.close()
    print("Database connection closed")

if __name__ == "__main__":
    main()