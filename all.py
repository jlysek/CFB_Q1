import os
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

class CFBAllQuartersPredictor:
    """
    Predicts all four quarter score probabilities with proper calibration and correlation modeling.
    
    Key improvements:
    - True calibration: adjusts quarter distributions proportionally
    - Percentage-based anchor threshold with floor
    - Bayesian quarter correlation: P(Q2|Q1), P(Q3|Q1,Q2), P(Q4|Q1,Q2,Q3)
    - Conditional probability models learned from historical data
    """
    
    def __init__(self, db_config):
        """Initialize with database config and empty containers for all quarter models."""
        self.db_config = db_config
        
        # Separate storage for each quarter
        self.empirical_distribution_q1 = {}
        self.empirical_distribution_q2 = {}
        self.empirical_distribution_q3 = {}
        self.empirical_distribution_q4 = {}
        
        self.standardized_empirical_dist_q1 = {}
        self.standardized_empirical_dist_q2 = {}
        self.standardized_empirical_dist_q3 = {}
        self.standardized_empirical_dist_q4 = {}
        
        self.model_params_q1 = {}
        self.model_params_q2 = {}
        self.model_params_q3 = {}
        self.model_params_q4 = {}
        
        self.anchor_scores_q1 = []
        self.anchor_scores_q2 = []
        self.anchor_scores_q3 = []
        self.anchor_scores_q4 = []
        
        # NEW: Conditional probability storage
        self.conditional_q2_given_q1 = {}  # P(Q2|Q1)
        self.conditional_q3_given_h1 = {}  # P(Q3|Q1,Q2) = P(Q3|H1)
        self.conditional_q4_given_h1_q3 = {}  # P(Q4|Q1,Q2,Q3)
        
        # NEW: Correlation factors
        self.total_correlation_q1_q2 = 0.0
        self.total_correlation_q3_q4 = 0.0
        self.margin_correlation_q1_q2 = 0.0
        self.margin_correlation_q3_q4 = 0.0
        
        self.historical_data = None
        self.typical_game_score = None
        
        # Quarter-specific empirical percentages
        self.Q1_PERCENTAGE_POINTS = None
        self.Q2_PERCENTAGE_POINTS = None
        self.Q3_PERCENTAGE_POINTS = None
        self.Q4_PERCENTAGE_POINTS = None
        
        # Q1 spread interpolator
        self.q1_spread_interpolator = None
        self.Q1_PERCENTAGE_SPREAD = None
        
        # NEW: Anchor threshold parameters
        self.MIN_ANCHOR_OCCURRENCES = 50  # Statistical stability floor
        self.ANCHOR_PERCENTAGE = 0.0075   # 0.75% of games
        
    def connect_to_database(self):
        """Connect to MySQL database."""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            return None
    
    def load_historical_data(self):
        """
        Load all four quarters from database.
        Standardize each quarter to favored-underdog orientation.
        Calculate Q1 spread interpolator and conditional probability distributions.
        """
        connection = self.connect_to_database()
        if not connection:
            return False
            
        try:
            # JOIN all 4 quarters onto one row per game
            query = """
            SELECT DISTINCT
                g.game_id,
                g.pregame_spread,
                g.pregame_total,
                g.home_points,
                g.away_points,
                q1.home_score AS q1_home_score,
                q1.away_score AS q1_away_score,
                q2.home_score AS q2_home_score,
                q2.away_score AS q2_away_score,
                q3.home_score AS q3_home_score,
                q3.away_score AS q3_away_score,
                q4.home_score AS q4_home_score,
                q4.away_score AS q4_away_score
            FROM cfb.games g
            LEFT JOIN cfb.quarter_scoring q1 ON g.game_id = q1.game_id AND q1.quarter = 1
            LEFT JOIN cfb.quarter_scoring q2 ON g.game_id = q2.game_id AND q2.quarter = 2
            LEFT JOIN cfb.quarter_scoring q3 ON g.game_id = q3.game_id AND q3.quarter = 3
            LEFT JOIN cfb.quarter_scoring q4 ON g.game_id = q4.game_id AND q4.quarter = 4
            WHERE 
                g.pregame_spread IS NOT NULL
                AND g.pregame_total IS NOT NULL
                AND g.pregame_spread BETWEEN -70 AND 70
                AND g.pregame_total BETWEEN 20 AND 90
                AND (g.home_classification = 'fbs' OR g.away_classification = 'fbs')
                AND g.home_points IS NOT NULL
                AND g.away_points IS NOT NULL
                AND q1.home_score IS NOT NULL
                AND q2.home_score IS NOT NULL
                AND q3.home_score IS NOT NULL
                AND q4.home_score IS NOT NULL
            ORDER BY g.game_id
            """
            
            df = pd.read_sql(query, connection)
            connection.close()
            
            # Remove duplicates
            if df['game_id'].duplicated().sum() > 0:
                df = df.drop_duplicates(subset=['game_id'], keep='first')
            
            print(f"Loaded {len(df)} games with all 4 quarters")
            
            # Calculate game-level aggregates
            df['game_total'] = df['home_points'] + df['away_points']
            df['game_margin'] = df['home_points'] - df['away_points']
            df['abs_game_margin'] = df['game_margin'].abs()
            df['abs_spread'] = df['pregame_spread'].abs()
            
            # Calculate quarter statistics
            for q in [1, 2, 3, 4]:
                df[f'q{q}_total'] = df[f'q{q}_home_score'] + df[f'q{q}_away_score']
                df[f'q{q}_margin'] = df[f'q{q}_home_score'] - df[f'q{q}_away_score']
                df[f'abs_q{q}_margin'] = df[f'q{q}_margin'].abs()
            
            # Q1 percentage of game points
            df['q1_points_pct'] = df['q1_total'] / df['game_total'].replace(0, np.nan)
            q1_pct_points = df['q1_points_pct'].mean()
            
            # Q1 percentage of game spread (non-tie games only)
            non_tie_games = df[df['abs_game_margin'] > 0].copy()
            non_tie_games['q1_spread_pct'] = non_tie_games['abs_q1_margin'] / non_tie_games['abs_game_margin']
            q1_pct_spread = non_tie_games['q1_spread_pct'].mean()
            
            # Create smooth interpolator for Q1 spread percentage
            spread_sorted = non_tie_games.sort_values('abs_spread')
            window_size = max(100, len(spread_sorted) // 50)
            spread_sorted['q1_spread_pct_smooth'] = spread_sorted['q1_spread_pct'].rolling(
                window=window_size, center=True, min_periods=50).mean()
            
            smooth_data = spread_sorted[['abs_spread', 'q1_spread_pct_smooth']].dropna()
            interpolation_points = [0, 3, 7, 10, 14, 17, 21, 28, 35, 50]
            interpolation_values = []
            
            for point in interpolation_points:
                nearby = smooth_data[
                    (smooth_data['abs_spread'] >= point - 2) & 
                    (smooth_data['abs_spread'] <= point + 2)
                ]
                if len(nearby) > 0:
                    interpolation_values.append(nearby['q1_spread_pct_smooth'].mean())
                else:
                    interpolation_values.append(q1_pct_spread)
            
            self.q1_spread_interpolator = UnivariateSpline(
                interpolation_points, interpolation_values, s=0.0005, k=3)
            
            self.Q1_PERCENTAGE_POINTS = q1_pct_points
            self.Q1_PERCENTAGE_SPREAD = q1_pct_spread
            
            # Calculate empirical percentages for all quarters
            print(f"\nEmpirical Quarter Statistics:")
            for q in [1, 2, 3, 4]:
                df[f'q{q}_points_pct'] = df[f'q{q}_total'] / df['game_total'].replace(0, np.nan)
                q_pct_points = df[f'q{q}_points_pct'].mean()
                setattr(self, f'Q{q}_PERCENTAGE_POINTS', q_pct_points)
                print(f"  Q{q} points as % of game total: {q_pct_points*100:.2f}%")
                print(f"  Mean Q{q} total: {df[f'q{q}_total'].mean():.2f} points")
            
            # NEW: Calculate correlation factors between quarters
            print(f"\nQuarter Correlation Analysis:")
            
            # Q1-Q2 correlations
            valid_games = df[(df['q1_total'] > 0) & (df['q2_total'] > 0)].copy()
            if len(valid_games) > 0:
                self.total_correlation_q1_q2 = valid_games['q1_total'].corr(valid_games['q2_total'])
                self.margin_correlation_q1_q2 = valid_games['q1_margin'].corr(valid_games['q2_margin'])
                print(f"  Q1-Q2 total correlation: {self.total_correlation_q1_q2:.3f}")
                print(f"  Q1-Q2 margin correlation: {self.margin_correlation_q1_q2:.3f}")
            
            # Q3-Q4 correlations
            valid_games_h2 = df[(df['q3_total'] > 0) & (df['q4_total'] > 0)].copy()
            if len(valid_games_h2) > 0:
                self.total_correlation_q3_q4 = valid_games_h2['q3_total'].corr(valid_games_h2['q4_total'])
                self.margin_correlation_q3_q4 = valid_games_h2['q3_margin'].corr(valid_games_h2['q4_margin'])
                print(f"  Q3-Q4 total correlation: {self.total_correlation_q3_q4:.3f}")
                print(f"  Q3-Q4 margin correlation: {self.margin_correlation_q3_q4:.3f}")
            print()
            
            # Store typical game score
            self.typical_game_score = df['game_total'].mean() / 2
            print(f"Typical game score per team: {self.typical_game_score:.2f} points\n")
            
            # Standardize all quarters to favored-underdog format
            for q in [1, 2, 3, 4]:
                # Raw home-away format
                df[f'q{q}_raw_score_combo'] = (
                    df[f'q{q}_home_score'].astype(str) + '-' + 
                    df[f'q{q}_away_score'].astype(str)
                )
                
                # Favored-underdog format
                df[f'q{q}_favored_score'] = np.where(
                    df['pregame_spread'] < 0,
                    df[f'q{q}_home_score'],
                    df[f'q{q}_away_score']
                )
                
                df[f'q{q}_underdog_score'] = np.where(
                    df['pregame_spread'] < 0,
                    df[f'q{q}_away_score'],
                    df[f'q{q}_home_score']
                )
                
                df[f'q{q}_score_combo'] = (
                    df[f'q{q}_favored_score'].astype(str) + '-' + 
                    df[f'q{q}_underdog_score'].astype(str)
                )
            
            # NEW: Build conditional probability distributions
            print("Building conditional probability distributions...")
            self._build_conditional_distributions(df)
            
            self.historical_data = df
            print("Data standardized to favored-underdog format for all quarters")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _build_conditional_distributions(self, df):
        """
        Build empirical conditional distributions:
        - P(Q2 | Q1)
        - P(Q3 | H1) where H1 = Q1 + Q2
        - P(Q4 | H1, Q3)
        
        We bucket Q1, H1, and (H1,Q3) contexts to ensure sufficient sample sizes.
        """
        print("  Building P(Q2 | Q1)...")
        
        # Bucket Q1 outcomes by pattern and total
        for _, game in df.iterrows():
            q1_score = game['q1_score_combo']
            q2_score = game['q2_score_combo']
            
            # Categorize Q1 context
            q1_context = self._categorize_quarter_outcome(q1_score)
            
            if q1_context not in self.conditional_q2_given_q1:
                self.conditional_q2_given_q1[q1_context] = {}
            
            if q2_score not in self.conditional_q2_given_q1[q1_context]:
                self.conditional_q2_given_q1[q1_context][q2_score] = 0
            
            self.conditional_q2_given_q1[q1_context][q2_score] += 1
        
        # Normalize P(Q2 | Q1_context)
        for q1_context in self.conditional_q2_given_q1:
            total = sum(self.conditional_q2_given_q1[q1_context].values())
            if total > 0:
                for q2_score in self.conditional_q2_given_q1[q1_context]:
                    self.conditional_q2_given_q1[q1_context][q2_score] /= total
        
        print(f"    P(Q2|Q1) contexts: {len(self.conditional_q2_given_q1)}")
        
        # Build P(Q3 | H1)
        print("  Building P(Q3 | H1)...")
        for _, game in df.iterrows():
            q1_fav = game['q1_favored_score']
            q1_dog = game['q1_underdog_score']
            q2_fav = game['q2_favored_score']
            q2_dog = game['q2_underdog_score']
            q3_score = game['q3_score_combo']
            
            # H1 = Q1 + Q2
            h1_fav = q1_fav + q2_fav
            h1_dog = q1_dog + q2_dog
            h1_score = f"{h1_fav}-{h1_dog}"
            
            # Categorize H1 context
            h1_context = self._categorize_quarter_outcome(h1_score)
            
            if h1_context not in self.conditional_q3_given_h1:
                self.conditional_q3_given_h1[h1_context] = {}
            
            if q3_score not in self.conditional_q3_given_h1[h1_context]:
                self.conditional_q3_given_h1[h1_context][q3_score] = 0
            
            self.conditional_q3_given_h1[h1_context][q3_score] += 1
        
        # Normalize P(Q3 | H1_context)
        for h1_context in self.conditional_q3_given_h1:
            total = sum(self.conditional_q3_given_h1[h1_context].values())
            if total > 0:
                for q3_score in self.conditional_q3_given_h1[h1_context]:
                    self.conditional_q3_given_h1[h1_context][q3_score] /= total
        
        print(f"    P(Q3|H1) contexts: {len(self.conditional_q3_given_h1)}")
        
        # Build P(Q4 | H1, Q3)
        print("  Building P(Q4 | H1, Q3)...")
        for _, game in df.iterrows():
            q1_fav = game['q1_favored_score']
            q1_dog = game['q1_underdog_score']
            q2_fav = game['q2_favored_score']
            q2_dog = game['q2_underdog_score']
            q3_fav = game['q3_favored_score']
            q3_dog = game['q3_underdog_score']
            q4_score = game['q4_score_combo']
            
            # H1 = Q1 + Q2
            h1_fav = q1_fav + q2_fav
            h1_dog = q1_dog + q2_dog
            h1_score = f"{h1_fav}-{h1_dog}"
            q3_score = game['q3_score_combo']
            
            # Categorize (H1, Q3) context
            h1_context = self._categorize_quarter_outcome(h1_score)
            q3_context = self._categorize_quarter_outcome(q3_score)
            combined_context = f"{h1_context}|{q3_context}"
            
            if combined_context not in self.conditional_q4_given_h1_q3:
                self.conditional_q4_given_h1_q3[combined_context] = {}
            
            if q4_score not in self.conditional_q4_given_h1_q3[combined_context]:
                self.conditional_q4_given_h1_q3[combined_context][q4_score] = 0
            
            self.conditional_q4_given_h1_q3[combined_context][q4_score] += 1
        
        # Normalize P(Q4 | H1_context, Q3_context)
        for combined_context in self.conditional_q4_given_h1_q3:
            total = sum(self.conditional_q4_given_h1_q3[combined_context].values())
            if total > 0:
                for q4_score in self.conditional_q4_given_h1_q3[combined_context]:
                    self.conditional_q4_given_h1_q3[combined_context][q4_score] /= total
        
        print(f"    P(Q4|H1,Q3) contexts: {len(self.conditional_q4_given_h1_q3)}")
    
    def _categorize_quarter_outcome(self, score_str):
        """
        Categorize a quarter outcome into a context bucket for conditional probability modeling.
        
        Buckets by:
        - Total points: low (0-7), medium (8-20), high (21+)
        - Margin: tie, close (1-7), blowout (8+)
        - Direction: favorite ahead, underdog ahead
        """
        try:
            fav_score, dog_score = map(int, score_str.split('-'))
        except:
            return 'unknown'
        
        margin = fav_score - dog_score
        total = fav_score + dog_score
        
        # Categorize total
        if total <= 7:
            total_cat = 'low'
        elif total <= 20:
            total_cat = 'med'
        else:
            total_cat = 'high'
        
        # Categorize margin
        if margin == 0:
            margin_cat = 'tie'
        elif abs(margin) <= 7:
            margin_cat = 'close_fav' if margin > 0 else 'close_dog'
        else:
            margin_cat = 'blow_fav' if margin > 0 else 'blow_dog'
        
        return f"{total_cat}_{margin_cat}"
    
    def calculate_empirical_distribution(self):
        """Calculate empirical distributions for all quarters."""
        if self.historical_data is None:
            return
        
        total_games = len(self.historical_data)
        
        # NEW: Calculate adaptive anchor threshold
        anchor_threshold = max(
            self.MIN_ANCHOR_OCCURRENCES,
            int(self.ANCHOR_PERCENTAGE * total_games)
        )
        self.anchor_threshold = anchor_threshold
        
        print(f"\nAnchor Threshold Calculation:")
        print(f"  Total games: {total_games}")
        print(f"  Percentage-based threshold (0.75%): {int(self.ANCHOR_PERCENTAGE * total_games)}")
        print(f"  Floor: {self.MIN_ANCHOR_OCCURRENCES}")
        print(f"  Selected threshold: {anchor_threshold}")
        print(f"  (Scores appearing >= {anchor_threshold} times are anchors)")
        
        for q in [1, 2, 3, 4]:
            # Home-away format
            raw_counts = self.historical_data[f'q{q}_raw_score_combo'].value_counts()
            raw_dist = {score: count / total_games for score, count in raw_counts.items()}
            setattr(self, f'empirical_distribution_q{q}', raw_dist)
            
            # Favored-underdog format
            std_counts = self.historical_data[f'q{q}_score_combo'].value_counts()
            std_dist = {score: count / total_games for score, count in std_counts.items()}
            setattr(self, f'standardized_empirical_dist_q{q}', std_dist)
            
            print(f"\nQ{q}: {len(std_dist)} unique favored-underdog scores")
            
            # Count anchors vs rare scores with new threshold
            anchors = sum(1 for count in std_counts.values if count >= anchor_threshold)
            rares = sum(1 for count in std_counts.values if count < anchor_threshold)
            print(f"  Anchors (>= {anchor_threshold}): {anchors}")
            print(f"  Rare (< {anchor_threshold}): {rares}")
    
    def get_score_pattern(self, score_str):
        """Classify score into pattern (same as before)."""
        fav_score, dog_score = map(int, score_str.split('-'))
        margin = fav_score - dog_score
        score_total = fav_score + dog_score
        
        # Ties
        if margin == 0:
            if score_total == 0:
                return 'scoreless_tie', score_total
            elif score_total <= 14:
                return 'low_tie', score_total
            elif score_total <= 20:
                return 'mid_tie', score_total
            else:
                return 'high_tie', score_total
        
        # Shutouts
        elif dog_score == 0:
            if fav_score <= 10:
                return 'fav_shutout_low', score_total
            elif fav_score <= 17:
                return 'fav_shutout_mid', score_total
            else:
                return 'fav_shutout_high', score_total
        
        elif fav_score == 0:
            if dog_score <= 10:
                return 'dog_shutout_low', score_total
            elif dog_score <= 17:
                return 'dog_shutout_mid', score_total
            else:
                return 'dog_shutout_high', score_total
        
        # Scoring games by margin
        elif margin >= 14:
            return 'fav_blowout', score_total
        elif margin >= 7:
            return 'fav_win_multi', score_total
        elif margin > 0:
            return 'fav_win_close', score_total
        elif margin <= -14:
            return 'dog_blowout', score_total
        elif margin <= -7:
            return 'dog_win_multi', score_total
        else:
            return 'dog_win_close', score_total
    
    def find_nearest_anchor(self, target_score, anchor_scores):
        """Find nearest anchor score by pattern and total points."""
        target_pattern, target_total = self.get_score_pattern(target_score)
        
        # Find anchors with same pattern
        same_pattern_anchors = []
        for anchor in anchor_scores:
            anchor_pattern, anchor_total = self.get_score_pattern(anchor)
            if anchor_pattern == target_pattern:
                same_pattern_anchors.append((anchor, anchor_total))
        
        # If we have same-pattern anchors, find closest by total
        if same_pattern_anchors:
            same_pattern_anchors.sort(key=lambda x: abs(x[1] - target_total))
            return same_pattern_anchors[0][0]
        
        # Otherwise find closest by total regardless of pattern
        all_anchors_with_totals = []
        for anchor in anchor_scores:
            anchor_pattern, anchor_total = self.get_score_pattern(anchor)
            all_anchors_with_totals.append((anchor, anchor_total))
        
        all_anchors_with_totals.sort(key=lambda x: abs(x[1] - target_total))
        return all_anchors_with_totals[0][0]
    
    def get_coefficient_bounds(self, score_str, n_occurrences):
        """Get bounds for total and margin coefficients."""
        fav_score, dog_score = map(int, score_str.split('-'))
        margin = fav_score - dog_score
        score_total = fav_score + dog_score
        
        # For common scores, use wide bounds; for rare, use tighter
        bound_width = 3 if n_occurrences >= 100 else 1.5
        
        if fav_score == dog_score:
            # Ties
            if score_total == 0:
                total_bounds = (-bound_width, 0)
            elif score_total <= 14:
                total_bounds = (-bound_width, bound_width)
            else:
                total_bounds = (0, bound_width)
            margin_bounds = (-1, 1)
            
        elif dog_score == 0:
            # Favorite shutouts
            if fav_score <= 10:
                total_bounds = (-bound_width, 1)
            elif fav_score <= 17:
                total_bounds = (-bound_width, bound_width)
            else:
                total_bounds = (0, bound_width)
            margin_bounds = (0, bound_width)
            
        elif fav_score == 0:
            # Underdog shutouts
            if dog_score <= 10:
                total_bounds = (-bound_width, 1)
            elif dog_score <= 17:
                total_bounds = (-bound_width, bound_width)
            else:
                total_bounds = (0, bound_width)
            margin_bounds = (-bound_width, 0)
            
        elif margin >= 14:
            # Large margins
            total_bounds = (0, bound_width)
            margin_bounds = (0 if margin > 0 else -bound_width, bound_width if margin > 0 else 0)
            
        elif score_total >= 21:
            # High-scoring games
            total_bounds = (0, bound_width)
            margin_bounds = (-bound_width, bound_width)
            
        else:
            # Default: flexible bounds
            total_bounds = (-bound_width, bound_width)
            margin_bounds = (-bound_width, bound_width)
        
        return total_bounds, margin_bounds
    
    def fit_model_for_quarter(self, quarter):
        """Fit model for specific quarter using anchor + inheritance approach."""
        if self.historical_data is None:
            return
        
        print(f"\n{'='*80}")
        print(f"FITTING MODELS FOR QUARTER {quarter}")
        print(f"{'='*80}")
        
        # Get score counts for this quarter
        score_col = f'q{quarter}_score_combo'
        score_counts = self.historical_data[score_col].value_counts()
        
        # Use adaptive anchor threshold
        anchor_threshold = self.anchor_threshold
        anchor_scores = score_counts[score_counts >= anchor_threshold].index.tolist()
        rare_scores = score_counts[score_counts < anchor_threshold].index.tolist()
        
        # Store anchor scores
        setattr(self, f'anchor_scores_q{quarter}', anchor_scores)
        
        print(f"Anchor scores (>= {anchor_threshold} occurrences): {len(anchor_scores)}")
        print(f"Rare scores (< {anchor_threshold} occurrences): {len(rare_scores)}")
        
        # Prepare feature matrix
        df_filtered = self.historical_data.copy()
        
        # Calculate orthogonal features
        df_filtered['abs_spread'] = df_filtered['pregame_spread'].abs()
        df_filtered['implied_fav_total'] = (df_filtered['pregame_total'] + df_filtered['abs_spread']) / 2
        df_filtered['implied_dog_total'] = (df_filtered['pregame_total'] - df_filtered['abs_spread']) / 2
        
        df_filtered['norm_fav'] = df_filtered['implied_fav_total'] / self.typical_game_score
        df_filtered['norm_dog'] = df_filtered['implied_dog_total'] / self.typical_game_score
        
        df_filtered['feature_total'] = df_filtered['norm_fav'] + df_filtered['norm_dog']
        df_filtered['feature_margin'] = df_filtered['norm_fav'] - df_filtered['norm_dog']
        
        # Feature matrix: [1, feature_total, feature_margin]
        X = df_filtered[['feature_total', 'feature_margin']].values
        X = np.column_stack([np.ones(len(X)), X])
        
        model_params = {}
        
        # Fit anchor scores (individual models)
        print("Fitting anchor scores...")
        anchor_fits = 0
        
        for score_combo in anchor_scores:
            try:
                # Get observations for this score
                mask = df_filtered[score_col] == score_combo
                y = mask.astype(int).values
                n_occurrences = y.sum()
                
                if n_occurrences < 10:
                    continue
                
                # Empirical probability
                emp_prob = n_occurrences / len(y)
                prior_logit = np.log(emp_prob / (1 - emp_prob + 1e-10))
                
                # Get coefficient bounds
                total_bounds, margin_bounds = self.get_coefficient_bounds(score_combo, n_occurrences)
                
                # Negative log-likelihood
                def negative_log_likelihood(params):
                    logits = X @ params
                    probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
                    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
                    ll = np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
                    
                    # L2 regularization
                    regularization = 0.01 * np.sum(params[1:]**2)
                    return -ll + regularization
                
                # Initial parameters
                initial_params = np.array([prior_logit, 0.0, 0.0])
                
                # Bounds
                param_bounds = [
                    (prior_logit - 5, prior_logit + 5),
                    total_bounds,
                    margin_bounds
                ]
                
                # Optimize
                result = minimize(negative_log_likelihood, initial_params,
                                method='L-BFGS-B', bounds=param_bounds)
                
                if result.success:
                    model_params[score_combo] = result.x
                    anchor_fits += 1
                    
            except Exception as e:
                continue
        
        print(f"Successfully fit {anchor_fits}/{len(anchor_scores)} anchor score models")
        
        # Fit rare scores (coefficient inheritance)
        print("Fitting rare scores with coefficient inheritance...")
        rare_fits = 0
        
        for score_combo in rare_scores:
            try:
                # Get observations
                mask = df_filtered[score_col] == score_combo
                y = mask.astype(int).values
                n_occurrences = y.sum()
                
                if n_occurrences < 5:
                    continue
                
                # Empirical probability
                emp_prob = n_occurrences / len(y)
                prior_logit = np.log(emp_prob / (1 - emp_prob + 1e-10))
                
                # Find nearest anchor
                nearest_anchor = self.find_nearest_anchor(score_combo, anchor_scores)
                anchor_params = model_params.get(nearest_anchor)
                
                if anchor_params is None:
                    continue
                
                # Inherit coefficients from anchor
                inherited_coef_total = anchor_params[1]
                inherited_coef_margin = anchor_params[2]
                
                # Get tighter bounds around inherited values
                total_bounds_tight = (inherited_coef_total - 0.5, inherited_coef_total + 0.5)
                margin_bounds_tight = (inherited_coef_margin - 0.5, inherited_coef_margin + 0.5)
                
                # Negative log-likelihood with stronger regularization
                def negative_log_likelihood_inherited(params):
                    logits = X @ params
                    probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
                    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
                    ll = np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
                    
                    # Stronger regularization toward inherited coefficients
                    reg_total = 0.1 * (params[1] - inherited_coef_total)**2
                    reg_margin = 0.1 * (params[2] - inherited_coef_margin)**2
                    return -ll + reg_total + reg_margin
                
                # Initial parameters start from inherited
                initial_params = np.array([prior_logit, inherited_coef_total, inherited_coef_margin])
                
                param_bounds = [
                    (prior_logit - 5, prior_logit + 5),
                    total_bounds_tight,
                    margin_bounds_tight
                ]
                
                # Optimize
                result = minimize(negative_log_likelihood_inherited, initial_params,
                                method='L-BFGS-B', bounds=param_bounds)
                
                if result.success:
                    model_params[score_combo] = result.x
                    rare_fits += 1
                    
            except Exception as e:
                continue
        
        print(f"Successfully fit {rare_fits}/{len(rare_scores)} rare score models")
        
        # Store model parameters
        setattr(self, f'model_params_q{quarter}', model_params)
        print(f"Total models for Q{quarter}: {len(model_params)}")
    
    def fit_model(self):
        """Fit models for all four quarters."""
        print(f"\n{'='*80}")
        print(f"FITTING MODELS FOR ALL QUARTERS")
        print(f"{'='*80}")
        
        for quarter in [1, 2, 3, 4]:
            self.fit_model_for_quarter(quarter)
    
    def predict_standardized_probability(self, spread, total, quarter):
        """
        Predict probabilities for a quarter in favored-underdog format.
        Uses same approach as before.
        """
        empirical_dist = getattr(self, f'standardized_empirical_dist_q{quarter}')
        model_params = getattr(self, f'model_params_q{quarter}')
        
        if not empirical_dist:
            return {}
        
        # Calculate features
        abs_spread = abs(spread)
        implied_fav_total = (total + abs_spread) / 2
        implied_dog_total = (total - abs_spread) / 2
        
        norm_fav = implied_fav_total / self.typical_game_score
        norm_dog = implied_dog_total / self.typical_game_score
        
        feature_total = norm_fav + norm_dog
        feature_margin = norm_fav - norm_dog
        
        # Feature vector
        X_pred = np.array([1.0, feature_total, feature_margin])
        
        # Generate predictions
        predictions = {}
        
        for score in empirical_dist.keys():
            base_prob = empirical_dist[score]
            
            # Apply model adjustment if available
            if score in model_params:
                params = model_params[score]
                logit = X_pred @ params
                model_prob = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
                
                # Blend: 90% model, 10% empirical
                predicted_prob = 0.9 * model_prob + 0.1 * base_prob
            else:
                # No model for this score, use empirical only
                predicted_prob = base_prob
            
            predictions[score] = predicted_prob
        
        # Normalize
        total_prob = sum(predictions.values())
        if total_prob > 0:
            predictions = {score: prob / total_prob for score, prob in predictions.items()}
        
        return predictions
    
    def adjust_quarter_for_correlation(self, current_q_dist, prior_context, correlation_map):
        """
        Adjust quarter distribution based on prior quarter context using learned conditional probabilities.
        
        Args:
            current_q_dist: Base distribution for current quarter
            prior_context: Context string from prior quarter(s)
            correlation_map: Conditional probability map P(Q_current | Q_prior)
        
        Returns:
            Adjusted distribution that accounts for correlation
        """
        if prior_context not in correlation_map:
            # No learned conditional distribution, return base distribution
            return current_q_dist
        
        conditional_dist = correlation_map[prior_context]
        
        # Blend base prediction with conditional distribution
        # Use 70% base (from regression model) + 30% conditional (from empirical correlation)
        adjusted_dist = {}
        
        all_scores = set(current_q_dist.keys()) | set(conditional_dist.keys())
        
        for score in all_scores:
            base_prob = current_q_dist.get(score, 0.0)
            cond_prob = conditional_dist.get(score, base_prob)  # Default to base if not in conditional
            
            # Weighted blend
            adjusted_prob = 0.7 * base_prob + 0.3 * cond_prob
            
            if adjusted_prob > 0:
                adjusted_dist[score] = adjusted_prob
        
        # Normalize
        total_prob = sum(adjusted_dist.values())
        if total_prob > 0:
            adjusted_dist = {score: prob / total_prob for score, prob in adjusted_dist.items()}
        
        return adjusted_dist
    
    def convert_to_home_away_format(self, fav_dog_predictions, home_favored):
        """Convert predictions from favored-underdog to home-away format."""
        home_away_predictions = {}
        
        for score, prob in fav_dog_predictions.items():
            fav_score, dog_score = map(int, score.split('-'))
            
            if home_favored:
                home_score = fav_score
                away_score = dog_score
            else:
                home_score = dog_score
                away_score = fav_score
            
            home_away_score = f"{home_score}-{away_score}"
            home_away_predictions[home_away_score] = prob
        
        return home_away_predictions
    
    def simulate_game_with_correlation(self, pregame_spread, total, n_sims=20000, 
                                       q1_dist=None, q2_dist=None, q3_dist=None, q4_dist=None):
        """
        NEW: Simulate full game using Bayesian updating with quarter correlations.
        
        This replaces simple convolution with proper P(Q2|Q1), P(Q3|H1), P(Q4|H1,Q3).
        
        Args:
            pregame_spread: Game spread
            total: Game total
            n_sims: Number of Monte Carlo simulations
            q1_dist, q2_dist, q3_dist, q4_dist: Optional quarter distributions to use
                If None, will generate from model
        
        Returns:
            Dictionary with full game score distribution
        """
        # Generate base quarter distributions (or use provided ones)
        dist_q1_base = q1_dist if q1_dist is not None else self.predict_standardized_probability(pregame_spread, total, 1)
        dist_q2_base = q2_dist if q2_dist is not None else self.predict_standardized_probability(pregame_spread, total, 2)
        dist_q3_base = q3_dist if q3_dist is not None else self.predict_standardized_probability(pregame_spread, total, 3)
        dist_q4_base = q4_dist if q4_dist is not None else self.predict_standardized_probability(pregame_spread, total, 4)
        
        # Convert to numpy arrays for vectorized sampling
        q1_scores = list(dist_q1_base.keys())
        q1_probs = np.array([dist_q1_base[s] for s in q1_scores])
        
        # Storage for full game outcomes
        full_game_scores = {}
        
        # Run Monte Carlo simulation
        for _ in range(n_sims):
            # Sample Q1
            q1_idx = np.random.choice(len(q1_scores), p=q1_probs)
            q1_score = q1_scores[q1_idx]
            
            # Adjust Q2 based on Q1 context
            q1_context = self._categorize_quarter_outcome(q1_score)
            dist_q2_adj = self.adjust_quarter_for_correlation(
                dist_q2_base, q1_context, self.conditional_q2_given_q1
            )
            
            # Sample Q2
            q2_scores = list(dist_q2_adj.keys())
            q2_probs = np.array([dist_q2_adj[s] for s in q2_scores])
            q2_probs /= q2_probs.sum()  # Ensure normalized
            q2_idx = np.random.choice(len(q2_scores), p=q2_probs)
            q2_score = q2_scores[q2_idx]
            
            # Calculate H1 (first half)
            q1_fav, q1_dog = map(int, q1_score.split('-'))
            q2_fav, q2_dog = map(int, q2_score.split('-'))
            h1_fav = q1_fav + q2_fav
            h1_dog = q1_dog + q2_dog
            h1_score = f"{h1_fav}-{h1_dog}"
            
            # Adjust Q3 based on H1 context
            h1_context = self._categorize_quarter_outcome(h1_score)
            dist_q3_adj = self.adjust_quarter_for_correlation(
                dist_q3_base, h1_context, self.conditional_q3_given_h1
            )
            
            # Sample Q3
            q3_scores = list(dist_q3_adj.keys())
            q3_probs = np.array([dist_q3_adj[s] for s in q3_scores])
            q3_probs /= q3_probs.sum()
            q3_idx = np.random.choice(len(q3_scores), p=q3_probs)
            q3_score = q3_scores[q3_idx]
            
            # Adjust Q4 based on (H1, Q3) context
            q3_context = self._categorize_quarter_outcome(q3_score)
            combined_context = f"{h1_context}|{q3_context}"
            dist_q4_adj = self.adjust_quarter_for_correlation(
                dist_q4_base, combined_context, self.conditional_q4_given_h1_q3
            )
            
            # Sample Q4
            q4_scores = list(dist_q4_adj.keys())
            q4_probs = np.array([dist_q4_adj[s] for s in q4_scores])
            q4_probs /= q4_probs.sum()
            q4_idx = np.random.choice(len(q4_scores), p=q4_probs)
            q4_score = q4_scores[q4_idx]
            
            # Calculate full game
            q3_fav, q3_dog = map(int, q3_score.split('-'))
            q4_fav, q4_dog = map(int, q4_score.split('-'))
            full_fav = h1_fav + q3_fav + q4_fav
            full_dog = h1_dog + q3_dog + q4_dog
            full_score = f"{full_fav}-{full_dog}"
            
            # Accumulate
            if full_score not in full_game_scores:
                full_game_scores[full_score] = 0
            full_game_scores[full_score] += 1
        
        # Convert counts to probabilities
        for score in full_game_scores:
            full_game_scores[score] /= n_sims
        
        return full_game_scores
    
    def convolve_quarters(self, dist1, dist2):
        """
        Convolve two quarter distributions.
        Uses independence assumption: P(A and B) = P(A) * P(B)
        
        NOTE: This is now only used for generating initial distributions.
        The actual game simulation uses simulate_game_with_correlation().
        """
        combined_dist = {}
        
        for score1, prob1 in dist1.items():
            fav1, dog1 = map(int, score1.split('-'))
            
            for score2, prob2 in dist2.items():
                fav2, dog2 = map(int, score2.split('-'))
                
                # Add scores together
                combined_fav = fav1 + fav2
                combined_dog = dog1 + dog2
                combined_score = f"{combined_fav}-{combined_dog}"
                
                # Multiply probabilities (independence)
                combined_prob = prob1 * prob2
                
                # Accumulate probability
                if combined_score not in combined_dist:
                    combined_dist[combined_score] = 0.0
                combined_dist[combined_score] += combined_prob
        
        return combined_dist
    
    def calibrate_full_game_distribution(self, pregame_spread, total, max_iterations=20):
        """
        NEW: True calibration approach that adjusts SOURCE quarter distributions.
        
        This replaces the old approach of just forcing one line to 50/50.
        
        Process:
        1. Generate initial quarter distributions
        2. Simulate full game
        3. Check if P(Fav covers) ≈ 50% and P(Over) ≈ 50%
        4. If not, calculate adjustment factors
        5. Proportionally adjust ALL quarter distributions
        6. Repeat until calibrated
        
        Returns:
            Calibrated full game distribution
        """
        abs_spread = abs(pregame_spread)
        
        print(f"\n{'='*80}")
        print(f"TRUE CALIBRATION: Adjusting quarter distributions")
        print(f"Target: 50% favorite covers {abs_spread:.1f}, 50% over {total:.1f}")
        print(f"{'='*80}")
        
        # Function to calculate full game metrics
        def calculate_metrics(dist):
            prob_fav_cover = 0.0
            prob_over = 0.0
            
            for score, prob in dist.items():
                fav_score, dog_score = map(int, score.split('-'))
                margin = fav_score - dog_score
                game_total = fav_score + dog_score
                
                # Spread: handle pushes for whole number lines
                if margin > abs_spread:
                    prob_fav_cover += prob
                elif margin == abs_spread:
                    prob_fav_cover += prob * 0.5
                
                # Total: handle pushes for whole number lines
                if game_total > total:
                    prob_over += prob
                elif game_total == total:
                    prob_over += prob * 0.5
            
            return prob_fav_cover, prob_over
        
        # Function to adjust a quarter distribution based on calibration needs
        def adjust_quarter_dist(q_dist, total_adjustment, spread_adjustment, quarter_pct_total):
            """
            Proportionally adjust quarter distribution.
            
            total_adjustment: < 1.0 means we need to shift toward lower totals
            spread_adjustment: < 1.0 means we need to shift toward underdog
            """
            adjusted_dist = {}
            
            expected_q_total = total * quarter_pct_total
            expected_q_margin = abs_spread * quarter_pct_total if quarter_pct_total else 0
            
            for score, prob in q_dist.items():
                fav_score, dog_score = map(int, score.split('-'))
                score_total = fav_score + dog_score
                margin = fav_score - dog_score
                
                # Calculate adjustment factor based on score characteristics
                factor = 1.0
                
                # Adjust for total calibration
                if expected_q_total > 0:
                    total_deviation = (score_total - expected_q_total) / expected_q_total
                    # If we need lower totals (total_adjustment < 1), penalize high-scoring outcomes
                    # If we need higher totals (total_adjustment > 1), boost high-scoring outcomes
                    total_factor = 1.0 + (total_adjustment - 1.0) * total_deviation
                    factor *= total_factor
                
                # Adjust for spread calibration
                if expected_q_margin > 0:
                    margin_deviation = (margin - expected_q_margin) / expected_q_margin if expected_q_margin else 0
                    # If we need underdog to do better (spread_adjustment < 1), penalize fav blowouts
                    # If we need favorite to do better (spread_adjustment > 1), boost fav blowouts
                    spread_factor = 1.0 + (spread_adjustment - 1.0) * margin_deviation
                    factor *= spread_factor
                
                # Clamp factor to reasonable range
                factor = np.clip(factor, 0.5, 1.5)
                
                adjusted_dist[score] = prob * factor
            
            # Normalize
            total_prob = sum(adjusted_dist.values())
            if total_prob > 0:
                adjusted_dist = {score: prob / total_prob for score, prob in adjusted_dist.items()}
            
            return adjusted_dist
        
        # Initialize quarter distributions
        dist_q1 = self.predict_standardized_probability(pregame_spread, total, 1)
        dist_q2 = self.predict_standardized_probability(pregame_spread, total, 2)
        dist_q3 = self.predict_standardized_probability(pregame_spread, total, 3)
        dist_q4 = self.predict_standardized_probability(pregame_spread, total, 4)
        
        # Iterative calibration
        for iteration in range(max_iterations):
            # Simulate full game with current adjusted quarter distributions
            full_game_dist = self.simulate_game_with_correlation(
                pregame_spread, total, n_sims=50000,
                q1_dist=dist_q1, q2_dist=dist_q2, q3_dist=dist_q3, q4_dist=dist_q4
            )
            
            # Check calibration
            prob_fav_cover, prob_over = calculate_metrics(full_game_dist)
            
            print(f"\nIteration {iteration + 1}:")
            print(f"  P(Fav covers {abs_spread:.1f}): {prob_fav_cover*100:.2f}%")
            print(f"  P(Over {total:.1f}): {prob_over*100:.2f}%")
            
            # Check if within tolerance (46.5% to 53.5%)
            if (0.465 <= prob_fav_cover <= 0.535) and (0.465 <= prob_over <= 0.535):
                print(f"\n✓ Calibration converged!")
                # Return both calibrated quarters and full game
                return {
                    'q1': dist_q1,
                    'q2': dist_q2,
                    'q3': dist_q3,
                    'q4': dist_q4,
                    'full_game': full_game_dist
                }
            
            # Calculate adjustment factors
            # If prob_over = 53%, we want to shift toward lower totals
            # total_adjustment = 0.50 / 0.53 = 0.943
            total_adjustment = 0.50 / prob_over if prob_over > 0 else 1.0
            
            # If prob_fav_cover = 48%, we want to shift toward favorite
            # spread_adjustment = 0.50 / 0.48 = 1.042
            spread_adjustment = 0.50 / prob_fav_cover if prob_fav_cover > 0 else 1.0
            
            # Clamp adjustments to prevent overcorrection
            total_adjustment = np.clip(total_adjustment, 0.9, 1.1)
            spread_adjustment = np.clip(spread_adjustment, 0.9, 1.1)
            
            print(f"  Total adjustment factor: {total_adjustment:.4f}")
            print(f"  Spread adjustment factor: {spread_adjustment:.4f}")
            
            # Adjust all quarter distributions proportionally
            dist_q1 = adjust_quarter_dist(dist_q1, total_adjustment, spread_adjustment, self.Q1_PERCENTAGE_POINTS)
            dist_q2 = adjust_quarter_dist(dist_q2, total_adjustment, spread_adjustment, self.Q2_PERCENTAGE_POINTS)
            dist_q3 = adjust_quarter_dist(dist_q3, total_adjustment, spread_adjustment, self.Q3_PERCENTAGE_POINTS)
            dist_q4 = adjust_quarter_dist(dist_q4, total_adjustment, spread_adjustment, self.Q4_PERCENTAGE_POINTS)
        
        # Max iterations reached
        print(f"\nCalibration reached max iterations ({max_iterations})")
        print(f"Final metrics:")
        print(f"  P(Fav covers): {prob_fav_cover*100:.2f}%")
        print(f"  P(Over): {prob_over*100:.2f}%")
        
        # Return both the calibrated quarter distributions and full game
        return {
            'q1': dist_q1,
            'q2': dist_q2,
            'q3': dist_q3,
            'q4': dist_q4,
            'full_game': full_game_dist
        }
    
    def predict(self, pregame_spread, total, debug=False):
        """
        Generate predictions for all quarters and combine with proper calibration.
        
        NEW: Uses Bayesian simulation with quarter correlations and true calibration.
        
        Returns dictionary with all quarter distributions plus calibrated full game.
        """
        home_favored = pregame_spread < 0
        
        if not debug:
            # Suppress output for clean API usage
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
        
        try:
            # NEW: Use calibration method which returns both calibrated full game 
            # and the adjusted quarter distributions used to produce it
            calibration_result = self.calibrate_full_game_distribution(pregame_spread, total)
            
            # Extract calibrated quarters and full game
            dist_q1 = calibration_result['q1']
            dist_q2 = calibration_result['q2']
            dist_q3 = calibration_result['q3']
            dist_q4 = calibration_result['q4']
            dist_full_game = calibration_result['full_game']
            
            # Generate halves using simple convolution for display
            # (Could also extract these from calibration if needed)
            dist_h1 = self.convolve_quarters(dist_q1, dist_q2)
            dist_h2 = self.convolve_quarters(dist_q3, dist_q4)
            
            # Convert to home-away format
            dist_q1_home_away = self.convert_to_home_away_format(dist_q1, home_favored)
            dist_q2_home_away = self.convert_to_home_away_format(dist_q2, home_favored)
            dist_q3_home_away = self.convert_to_home_away_format(dist_q3, home_favored)
            dist_q4_home_away = self.convert_to_home_away_format(dist_q4, home_favored)
            dist_h1_home_away = self.convert_to_home_away_format(dist_h1, home_favored)
            dist_h2_home_away = self.convert_to_home_away_format(dist_h2, home_favored)
            dist_full_game_home_away = self.convert_to_home_away_format(dist_full_game, home_favored)
            
            return {
                'q1_fav_dog': dist_q1,
                'q2_fav_dog': dist_q2,
                'q3_fav_dog': dist_q3,
                'q4_fav_dog': dist_q4,
                'h1_fav_dog': dist_h1,
                'h2_fav_dog': dist_h2,
                'full_game_fav_dog': dist_full_game,
                'q1_home_away': dist_q1_home_away,
                'q2_home_away': dist_q2_home_away,
                'q3_home_away': dist_q3_home_away,
                'q4_home_away': dist_q4_home_away,
                'h1_home_away': dist_h1_home_away,
                'h2_home_away': dist_h2_home_away,
                'full_game_home_away': dist_full_game_home_away,
                'spread': pregame_spread,
                'total': total,
                'home_favored': home_favored
            }
            
        finally:
            if not debug:
                sys.stdout = old_stdout


load_dotenv()

def main():
    """Main execution function."""
    db_config = {
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'cfb')
    }
    
    predictor = CFBAllQuartersPredictor(db_config)
    
    print("="*80)
    print("CFB ALL QUARTERS PREDICTOR - IMPROVED VERSION")
    print("="*80)
    print("NEW Features:")
    print("  ✓ True calibration: adjusts quarter distributions proportionally")
    print("  ✓ Percentage-based anchor threshold (0.75% with floor of 50)")
    print("  ✓ Bayesian quarter correlation: P(Q2|Q1), P(Q3|H1), P(Q4|H1,Q3)")
    print("  ✓ Learned conditional probabilities from historical data")
    print("="*80)
    
    print("\nLoading historical data...")
    if not predictor.load_historical_data():
        print("Failed to load data")
        return
    
    print("\nCalculating empirical distributions...")
    predictor.calculate_empirical_distribution()
    
    print("\nFitting models for all quarters...")
    predictor.fit_model()
    
    print("\n" + "="*80)
    print("MODEL READY")
    print("="*80)
    print("Enter signed spread and total for predictions")
    print("Format: spread total (e.g., -3.5 58.5)")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("\nEnter spread and total: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            parts = user_input.split()
            if len(parts) != 2:
                print("Enter two numbers: spread and total")
                continue
            
            pregame_spread = float(parts[0])
            total = float(parts[1])
            
            if not (-50 <= pregame_spread <= 50) or not (30 <= total <= 90):
                print("Use realistic values: spread -50 to 50, total 30 to 90")
                continue
            
            result = predictor.predict(pregame_spread, total, debug=True)
            
            # Display top predictions
            print(f"\n{'='*80}")
            print(f"TOP PREDICTIONS BY QUARTER")
            print(f"{'='*80}")
            
            for q in [1, 2, 3, 4]:
                dist = result[f'q{q}_home_away']
                sorted_scores = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10]
                print(f"\nQuarter {q} - Top 10:")
                for score, prob in sorted_scores:
                    print(f"  {score:<10} {prob*100:>6.2f}%")
            
            # Display first half
            dist_h1 = result['h1_home_away']
            sorted_h1 = sorted(dist_h1.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\nFirst Half - Top 10:")
            for score, prob in sorted_h1:
                print(f"  {score:<10} {prob*100:>6.2f}%")
            
            # Display full game (calibrated)
            dist_fg = result['full_game_home_away']
            sorted_fg = sorted(dist_fg.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\nFull Game (Calibrated with Correlation) - Top 10:")
            for score, prob in sorted_fg:
                print(f"  {score:<10} {prob*100:>6.2f}%")
            
        except ValueError:
            print("Enter valid numbers")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()