import cfbd
import os
import pandas as pd
from collections import defaultdict

# Helper function to parse clock string to seconds
def parse_clock(clock_str):
    """
    Parse clock string like '15:00' or '14:37' to seconds
    Returns seconds remaining in the period
    """
    if not clock_str or ':' not in clock_str:
        return 0
    try:
        parts = clock_str.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except:
        return 0

# Configure API client with your API key
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = os.environ.get('CFBD_API_KEY', 'meWCW794Y9rcfCIbk8aRXkLpI35TPBpvf8uzKg9A1yKXJBryMKqx80pETU79J4xT')
configuration.api_key_prefix['Authorization'] = 'Bearer'

# Initialize API instances
plays_api = cfbd.PlaysApi(cfbd.ApiClient(configuration))
coaches_api = cfbd.CoachesApi(cfbd.ApiClient(configuration))

# Test parameters - using recent season and week to get sample data
test_year = 2024
test_week = 1

print(f"Fetching play data for {test_year} Week {test_week}...")
print("=" * 80)

try:
    # Get plays for the test week
    # Note: This will return all plays for all games in that week
    plays = plays_api.get_plays(
        year=test_year,
        week=test_week,
        season_type='regular'  # Can also try 'postseason'
    )
    
    print(f"\nTotal plays retrieved: {len(plays)}")
    
    if len(plays) > 0:
        # Examine the first few plays to understand structure
        print("\n" + "=" * 80)
        print("EXAMINING FIRST 5 PLAYS:")
        print("=" * 80)
        
        for i, play in enumerate(plays[:5]):
            print(f"\nPlay {i+1}:")
            print(f"  Game ID: {play.game_id}")
            print(f"  Period: {play.period}")
            print(f"  Clock: {play.clock}")
            print(f"  Offense: {play.offense}")
            print(f"  Defense: {play.defense}")
            print(f"  Home Team: {play.home}")
            print(f"  Away Team: {play.away}")
            print(f"  Play Type: {play.play_type}")
            print(f"  Play Text: {play.play_text}")
            print(f"  Down: {play.down}")
            print(f"  Distance: {play.distance}")
            print(f"  Yard Line: {play.yard_line}")
            
        # Group plays by game to find first plays of each game
        print("\n" + "=" * 80)
        print("FIRST PLAY OF EACH GAME:")
        print("=" * 80)
        
        games_dict = defaultdict(list)
        for play in plays:
            games_dict[play.game_id].append(play)
        
        # For each game, find the first play (should be kickoff)
        first_plays_data = []
        
        for game_id, game_plays in games_dict.items():
            # Sort by period then by clock descending (15:00 is start)
            sorted_plays = sorted(game_plays, key=lambda x: (x.period, -parse_clock(x.clock)))
            
            first_play = sorted_plays[0]
            
            print(f"\nGame {game_id}: {first_play.away} @ {first_play.home}")
            print(f"  First Play Period: {first_play.period}")
            print(f"  First Play Clock: {first_play.clock}")
            print(f"  First Play Type: {first_play.play_type}")
            print(f"  Play Text: {first_play.play_text}")
            print(f"  Offense: {first_play.offense}")
            print(f"  Defense: {first_play.defense}")
            
            # Try to find the first non-kickoff play (actual first possession)
            first_possession_play = None
            for play in sorted_plays[1:6]:  # Check next 5 plays
                if play.play_type not in ['Kickoff', 'Kickoff Return (Offense)', 'Kickoff Return Touchdown']:
                    first_possession_play = play
                    break
            
            if first_possession_play:
                print(f"\n  First Possession Play:")
                print(f"    Offense Team: {first_possession_play.offense}")
                print(f"    Play Type: {first_possession_play.play_type}")
                print(f"    Play Text: {first_possession_play.play_text}")
            
            first_plays_data.append({
                'game_id': game_id,
                'home_team': first_play.home,
                'away_team': first_play.away,
                'first_play_type': first_play.play_type,
                'first_play_offense': first_play.offense,
                'first_possession_offense': first_possession_play.offense if first_possession_play else None
            })
        
        # Create DataFrame to analyze
        df_first_plays = pd.DataFrame(first_plays_data)
        print("\n" + "=" * 80)
        print("SUMMARY OF FIRST POSSESSIONS:")
        print("=" * 80)
        print(df_first_plays.to_string())
        
        # Check what play types exist for kickoffs
        print("\n" + "=" * 80)
        print("ALL UNIQUE PLAY TYPES IN DATASET:")
        print("=" * 80)
        play_types = set(play.play_type for play in plays if play.play_type)
        for pt in sorted(play_types):
            print(f"  - {pt}")
        
        # Check if there are any plays specifically marked as kickoff or coin toss
        print("\n" + "=" * 80)
        print("KICKOFF-RELATED PLAYS:")
        print("=" * 80)
        kickoff_plays = [p for p in plays if p.play_type and 'kick' in p.play_type.lower()]
        print(f"Found {len(kickoff_plays)} kickoff-related plays")
        if kickoff_plays:
            for kp in kickoff_plays[:10]:
                print(f"  Game {kp.game_id}, Period {kp.period}: {kp.play_type} - {kp.play_text}")
        
    else:
        print("No plays returned. This might indicate:")
        print("  1. Invalid API key")
        print("  2. No data available for this week/year")
        print("  3. API endpoint issue")

except Exception as e:
    print(f"Error occurred: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nPlease check:")
    print("  1. Your API key is valid and set correctly")
    print("  2. The year/week combination has available data")
    print("  3. Your internet connection is working")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("""
Based on the output above, we can determine the best approach:

1. If kickoff plays are clearly labeled with offense/defense teams:
   - The team on OFFENSE during the kickoff is kicking off
   - The team on DEFENSE is receiving
   - The receiving team got the ball first

2. If we need to infer from first scrimmage play:
   - Find the first non-kickoff play after the opening kickoff
   - The offense team on that play received the opening kickoff

3. We'll then need to:
   - Map this back to coaches via the games table
   - Track which coach got the ball first (0) or second (1)
   - Build a running tally for each coach across all games
""")