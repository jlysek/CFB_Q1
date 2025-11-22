import os
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

class CFBAllQuartersPredictor:
    """
    Predicts all four quarter score probabilities with:
    - Bayesian credibility weighting for correlations
    - Enhanced halftime differential modeling for Q3/Q4
    - Logit-space calibration for better distribution shape preservation
    - Smooth garbage time detection for Q4
    - Laplace smoothing for credibility weighting
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
        
        # Conditional probability storage with sample size tracking
        self.conditional_q2_given_q1 = {}
        self.conditional_q3_given_h1 = {}
        self.conditional_q4_given_h1_q3 = {}
        
        # Sample size tracking for credibility weighting
        self.conditional_q2_sample_sizes = {}
        self.conditional_q3_sample_sizes = {}
        self.conditional_q4_sample_sizes = {}
        
        # Garbage time: Separate Q4 conditional distributions
        self.conditional_q4_competitive = {}
        self.conditional_q4_garbage = {}
        self.conditional_q4_sample_sizes_competitive = {}
        self.conditional_q4_sample_sizes_garbage = {}
        
        # Correlation factors
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
        
        # Anchor threshold parameters
        self.MIN_ANCHOR_OCCURRENCES = 50
        self.ANCHOR_PERCENTAGE = 0.0075
        
        # Credibility smoothing parameter
        self.CREDIBILITY_K = 50
        
        # Laplace smoothing parameter
        self.LAPLACE_ALPHA = 0.01
        
        # Garbage time parameters (learned from data)
        self.GARBAGE_TIME_MIDPOINT = None
        self.GARBAGE_TIME_STEEPNESS = None
        
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
        Load all four quarters from database with temporal weighting.
        Apply higher weights to recent seasons (2023+) due to rule changes.
        """
        connection = self.connect_to_database()
        if not connection:
            return False
            
        try:
            query = """
            SELECT DISTINCT
                g.game_id,
                g.season,
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
                AND g.betting_provider NOT IN ('median_1_providers', 'teamrankings', 'consensus', 'null')
            ORDER BY g.game_id
            """
            
            df = pd.read_sql(query, connection)
            connection.close()
            
            # Remove duplicates
            if df['game_id'].duplicated().sum() > 0:
                df = df.drop_duplicates(subset=['game_id'], keep='first')
            
            print(f"Loaded {len(df)} games with all 4 quarters")
            
            # Apply temporal weights for rule changes
            df['temporal_weight'] = df['season'].apply(lambda s: 
                2.5 if s >= 2023 else
                1.5 if s >= 2020 else
                1.0
            )
            
            print(f"\nTemporal Weighting Applied:")
            for season_group, weight in [(2023, 2.5), (2020, 1.5), (2014, 1.0)]:
                count = len(df[df['temporal_weight'] == weight])
                pct = count / len(df) * 100
                print(f"  Season {season_group}+: Weight {weight}x, {count} games ({pct:.1f}%)")
            
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
            
            # Q1 percentage of game points (weighted)
            df['q1_points_pct'] = df['q1_total'] / df['game_total'].replace(0, np.nan)
            q1_pct_points = np.average(df['q1_points_pct'].dropna(), weights=df.loc[df['q1_points_pct'].notna(), 'temporal_weight'])
            
            # Q1 percentage of game spread (weighted, non-tie games only)
            non_tie_games = df[df['abs_game_margin'] > 0].copy()
            non_tie_games['q1_spread_pct'] = non_tie_games['abs_q1_margin'] / non_tie_games['abs_game_margin']
            q1_pct_spread = np.average(
                non_tie_games['q1_spread_pct'].dropna(),
                weights=non_tie_games.loc[non_tie_games['q1_spread_pct'].notna(), 'temporal_weight']
            )
            
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
            
            # Calculate empirical percentages for all quarters (weighted)
            print(f"\nWeighted Empirical Quarter Statistics:")
            for q in [1, 2, 3, 4]:
                df[f'q{q}_points_pct'] = df[f'q{q}_total'] / df['game_total'].replace(0, np.nan)
                q_pct_points = np.average(
                    df[f'q{q}_points_pct'].dropna(),
                    weights=df.loc[df[f'q{q}_points_pct'].notna(), 'temporal_weight']
                )
                setattr(self, f'Q{q}_PERCENTAGE_POINTS', q_pct_points)
                
                # Weighted mean for total points
                weighted_mean_total = np.average(df[f'q{q}_total'], weights=df['temporal_weight'])
                print(f"  Q{q} points as % of game total: {q_pct_points*100:.2f}%")
                print(f"  Mean Q{q} total (weighted): {weighted_mean_total:.2f} points")
            
            # Quarter correlation analysis
            print(f"\nQuarter Correlation Analysis:")
            valid_games = df[(df['q1_total'] > 0) & (df['q2_total'] > 0)].copy()
            if len(valid_games) > 0:
                self.total_correlation_q1_q2 = valid_games['q1_total'].corr(valid_games['q2_total'])
                self.margin_correlation_q1_q2 = valid_games['q1_margin'].corr(valid_games['q2_margin'])
                print(f"  Q1-Q2 total correlation: {self.total_correlation_q1_q2:.3f}")
                print(f"  Q1-Q2 margin correlation: {self.margin_correlation_q1_q2:.3f}")
            
            valid_games_h2 = df[(df['q3_total'] > 0) & (df['q4_total'] > 0)].copy()
            if len(valid_games_h2) > 0:
                self.total_correlation_q3_q4 = valid_games_h2['q3_total'].corr(valid_games_h2['q4_total'])
                self.margin_correlation_q3_q4 = valid_games_h2['q3_margin'].corr(valid_games_h2['q4_margin'])
                print(f"  Q3-Q4 total correlation: {self.total_correlation_q3_q4:.3f}")
                print(f"  Q3-Q4 margin correlation: {self.margin_correlation_q3_q4:.3f}")
            print()
            
            # Store typical game score (weighted)
            self.typical_game_score = np.average(df['game_total'], weights=df['temporal_weight']) / 2
            print(f"Typical game score per team (weighted): {self.typical_game_score:.2f} points\n")
            
            # Calculate halftime differential for Q3 context
            df['h1_fav_score'] = np.where(
                df['pregame_spread'] < 0,
                df['q1_home_score'] + df['q2_home_score'],
                df['q1_away_score'] + df['q2_away_score']
            )
            df['h1_dog_score'] = np.where(
                df['pregame_spread'] < 0,
                df['q1_away_score'] + df['q2_away_score'],
                df['q1_home_score'] + df['q2_home_score']
            )
            df['h1_margin'] = df['h1_fav_score'] - df['h1_dog_score']
            
            # Standardize all quarters to favored-underdog format
            for q in [1, 2, 3, 4]:
                df[f'q{q}_raw_score_combo'] = (
                    df[f'q{q}_home_score'].astype(str) + '-' + 
                    df[f'q{q}_away_score'].astype(str)
                )
                
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
            
            # Learn garbage time threshold from data
            print("Learning garbage time threshold from data...")
            self._learn_garbage_time_threshold(df)
            
            # Build conditional probability distributions
            print("Building conditional probability distributions with sample size tracking...")
            self._build_conditional_distributions(df)
            
            self.historical_data = df
            print("Data standardized to favored-underdog format for all quarters")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _learn_garbage_time_threshold(self, df):
        """
        Learn garbage time threshold from historical Q4 patterns using changepoint detection.
        Uses data to find where Q4 correlation inverts (garbage time effect).
        """
        # Calculate entering Q4 margins
        df_q4_analysis = df.copy()
        
        # Determine favorite from pregame spread
        df_q4_analysis['favorite_is_home'] = df_q4_analysis['pregame_spread'] < 0
        
        # Calculate margins from favorite's perspective
        df_q4_analysis['q3_fav'] = np.where(
            df_q4_analysis['favorite_is_home'],
            df_q4_analysis['q3_home_score'],
            df_q4_analysis['q3_away_score']
        )
        df_q4_analysis['q3_dog'] = np.where(
            df_q4_analysis['favorite_is_home'],
            df_q4_analysis['q3_away_score'],
            df_q4_analysis['q3_home_score']
        )
        df_q4_analysis['q4_fav'] = np.where(
            df_q4_analysis['favorite_is_home'],
            df_q4_analysis['q4_home_score'],
            df_q4_analysis['q4_away_score']
        )
        df_q4_analysis['q4_dog'] = np.where(
            df_q4_analysis['favorite_is_home'],
            df_q4_analysis['q4_away_score'],
            df_q4_analysis['q4_home_score']
        )
        
        # Calculate entering Q4 margin (H1 + Q3)
        df_q4_analysis['q3_margin'] = df_q4_analysis['q3_fav'] - df_q4_analysis['q3_dog']
        df_q4_analysis['entering_q4_margin'] = df_q4_analysis['h1_margin'] + df_q4_analysis['q3_margin']
        df_q4_analysis['q4_margin'] = df_q4_analysis['q4_fav'] - df_q4_analysis['q4_dog']
        
        # Bin by entering margin
        margin_bins = np.arange(-35, 36, 2)
        bin_centers = []
        q4_margin_means = []
        sample_sizes = []
        
        for i in range(len(margin_bins) - 1):
            lower, upper = margin_bins[i], margin_bins[i+1]
            mask = (df_q4_analysis['entering_q4_margin'] >= lower) & (df_q4_analysis['entering_q4_margin'] < upper)
            subset = df_q4_analysis[mask]
            
            if len(subset) >= 15:
                bin_centers.append((lower + upper) / 2)
                q4_margin_means.append(subset['q4_margin'].mean())
                sample_sizes.append(len(subset))
        
        if len(bin_centers) == 0:
            print("  Insufficient data for garbage time analysis, using default threshold")
            self.GARBAGE_TIME_MIDPOINT = 18.5
            self.GARBAGE_TIME_STEEPNESS = 0.15
            return
        
        bin_centers = np.array(bin_centers)
        q4_margin_means = np.array(q4_margin_means)
        sample_sizes = np.array(sample_sizes)
        
        # Find changepoint: where does correlation flip?
        positive_mask = bin_centers > 7
        positive_centers = bin_centers[positive_mask]
        positive_means = q4_margin_means[positive_mask]
        positive_sizes = sample_sizes[positive_mask]
        
        if len(positive_centers) < 10:
            print("  Insufficient data for changepoint detection, using default")
            self.GARBAGE_TIME_MIDPOINT = 18.5
            self.GARBAGE_TIME_STEEPNESS = 0.15
            return
        
        def find_changepoint(threshold):
            """Find threshold that maximizes difference in slopes before/after."""
            before_mask = positive_centers < threshold
            after_mask = positive_centers >= threshold
            
            if before_mask.sum() < 3 or after_mask.sum() < 3:
                return 1e6
            
            X_before = positive_centers[before_mask]
            y_before = positive_means[before_mask]
            w_before = positive_sizes[before_mask]
            
            X_after = positive_centers[after_mask]
            y_after = positive_means[after_mask]
            w_after = positive_sizes[after_mask]
            
            # Weighted linear regression slopes
            if len(X_before) > 1:
                W_before = np.diag(w_before)
                X_before_mat = np.column_stack([np.ones(len(X_before)), X_before])
                try:
                    beta_before = np.linalg.lstsq(
                        X_before_mat.T @ W_before @ X_before_mat,
                        X_before_mat.T @ W_before @ y_before,
                        rcond=None
                    )[0]
                    slope_before = beta_before[1]
                except:
                    slope_before = 0
            else:
                slope_before = 0
            
            if len(X_after) > 1:
                W_after = np.diag(w_after)
                X_after_mat = np.column_stack([np.ones(len(X_after)), X_after])
                try:
                    beta_after = np.linalg.lstsq(
                        X_after_mat.T @ W_after @ X_after_mat,
                        X_after_mat.T @ W_after @ y_after,
                        rcond=None
                    )[0]
                    slope_after = beta_after[1]
                except:
                    slope_after = 0
            else:
                slope_after = 0
            
            slope_difference = slope_before - slope_after
            return -slope_difference
        
        # Search for optimal threshold
        thresholds_to_test = np.arange(12, 28, 0.5)
        losses = [find_changepoint(t) for t in thresholds_to_test]
        optimal_idx = np.argmin(losses)
        optimal_threshold = thresholds_to_test[optimal_idx]
        
        # Fit sigmoid for smooth transitions
        positive_big_lead = (bin_centers > 10)
        garbage_indicator = np.where(
            positive_big_lead & (q4_margin_means < 0),
            1.0,
            0.0
        )
        
        def sigmoid_loss(params):
            midpoint, steepness = params
            predicted = expit(steepness * (bin_centers - midpoint))
            weights = np.sqrt(sample_sizes)
            return np.sum(weights * (predicted - garbage_indicator)**2)
        
        try:
            result = minimize(
                sigmoid_loss,
                x0=[optimal_threshold, 0.15],
                bounds=[(12.0, 28.0), (0.05, 0.5)],
                method='L-BFGS-B'
            )
            
            self.GARBAGE_TIME_MIDPOINT = result.x[0]
            self.GARBAGE_TIME_STEEPNESS = result.x[1]
        except:
            # Fallback if optimization fails
            self.GARBAGE_TIME_MIDPOINT = optimal_threshold
            self.GARBAGE_TIME_STEEPNESS = 0.15
        
        print(f"  Garbage time midpoint: {self.GARBAGE_TIME_MIDPOINT:.1f} points (~{self.GARBAGE_TIME_MIDPOINT/7:.1f} TDs)")
        print(f"  Garbage time steepness: {self.GARBAGE_TIME_STEEPNESS:.3f}")
    
    def predict_garbage_probability(self, entering_q4_margin):
        """
        Predict probability that game is in garbage time using smooth sigmoid.
        NO hard thresholds - smooth transition based on learned parameters.
        
        Args:
            entering_q4_margin: Favorite's cumulative margin (H1 + Q3)
        
        Returns:
            float: Probability between 0 and 1
        """
        if self.GARBAGE_TIME_MIDPOINT is None:
            return 0.0
        
        # Sigmoid: smooth transition, no discontinuities
        prob = expit(self.GARBAGE_TIME_STEEPNESS * (abs(entering_q4_margin) - self.GARBAGE_TIME_MIDPOINT))
        return float(prob)
    
    def _build_conditional_distributions(self, df):
        """
        Build empirical conditional distributions with sample size tracking
        for Bayesian credibility weighting. Q4 uses smooth garbage time weighting.
        """
        print("  Building P(Q2 | Q1) with sample sizes...")
        
        # Build P(Q2 | Q1)
        for _, game in df.iterrows():
            q1_score = game['q1_score_combo']
            q2_score = game['q2_score_combo']
            weight = game['temporal_weight']
            
            q1_context = self._categorize_quarter_outcome(q1_score)
            
            if q1_context not in self.conditional_q2_given_q1:
                self.conditional_q2_given_q1[q1_context] = {}
                self.conditional_q2_sample_sizes[q1_context] = 0
            
            if q2_score not in self.conditional_q2_given_q1[q1_context]:
                self.conditional_q2_given_q1[q1_context][q2_score] = 0
            
            self.conditional_q2_given_q1[q1_context][q2_score] += weight
            self.conditional_q2_sample_sizes[q1_context] += weight
        
        # Normalize P(Q2 | Q1)
        for q1_context in self.conditional_q2_given_q1:
            total = sum(self.conditional_q2_given_q1[q1_context].values())
            if total > 0:
                for q2_score in self.conditional_q2_given_q1[q1_context]:
                    self.conditional_q2_given_q1[q1_context][q2_score] /= total
        
        sample_sizes_q2 = list(self.conditional_q2_sample_sizes.values())
        print(f"    P(Q2|Q1) contexts: {len(self.conditional_q2_given_q1)}")
        print(f"    Sample sizes - Min: {min(sample_sizes_q2):.0f}, Median: {np.median(sample_sizes_q2):.0f}, Max: {max(sample_sizes_q2):.0f}")
        
        # Build P(Q3 | H1)
        print("  Building P(Q3 | H1) with halftime differential emphasis...")
        for _, game in df.iterrows():
            h1_fav = game['h1_fav_score']
            h1_dog = game['h1_dog_score']
            q3_score = game['q3_score_combo']
            weight = game['temporal_weight']
            
            h1_score = f"{h1_fav}-{h1_dog}"
            h1_context = self._categorize_halftime_outcome(h1_score)
            
            if h1_context not in self.conditional_q3_given_h1:
                self.conditional_q3_given_h1[h1_context] = {}
                self.conditional_q3_sample_sizes[h1_context] = 0
            
            if q3_score not in self.conditional_q3_given_h1[h1_context]:
                self.conditional_q3_given_h1[h1_context][q3_score] = 0
            
            self.conditional_q3_given_h1[h1_context][q3_score] += weight
            self.conditional_q3_sample_sizes[h1_context] += weight
        
        # Normalize P(Q3 | H1)
        for h1_context in self.conditional_q3_given_h1:
            total = sum(self.conditional_q3_given_h1[h1_context].values())
            if total > 0:
                for q3_score in self.conditional_q3_given_h1[h1_context]:
                    self.conditional_q3_given_h1[h1_context][q3_score] /= total
        
        sample_sizes_q3 = list(self.conditional_q3_sample_sizes.values())
        print(f"    P(Q3|H1) contexts: {len(self.conditional_q3_given_h1)}")
        print(f"    Sample sizes - Min: {min(sample_sizes_q3):.0f}, Median: {np.median(sample_sizes_q3):.0f}, Max: {max(sample_sizes_q3):.0f}")
        
        # Build P(Q4 | H1, Q3) with SMOOTH garbage time weighting
        print("  Building P(Q4 | H1, Q3) with smooth garbage time...")
        
        for _, game in df.iterrows():
            h1_fav = game['h1_fav_score']
            h1_dog = game['h1_dog_score']
            
            # Parse Q3 score
            try:
                q3_fav, q3_dog = map(int, game['q3_score_combo'].split('-'))
            except:
                continue
            
            q4_score = game['q4_score_combo']
            weight = game['temporal_weight']
            
            # Calculate entering Q4 margin
            h1_margin = h1_fav - h1_dog
            q3_margin = q3_fav - q3_dog
            entering_margin = h1_margin + q3_margin
            
            # Get SMOOTH garbage probability (no hard threshold)
            garbage_prob = self.predict_garbage_probability(entering_margin)
            
            # Create context category for bucketing
            entering_total = (h1_fav + h1_dog) + (q3_fav + q3_dog)
            
            if entering_total <= 28:
                total_cat = 'low'
            elif entering_total <= 56:
                total_cat = 'med'
            else:
                total_cat = 'high'
            
            abs_margin = abs(entering_margin)
            if abs_margin < 4:
                margin_cat = 'close'
            elif abs_margin < 8:
                margin_cat = 'onescore'
            elif abs_margin < 12:
                margin_cat = 'onescore_plus'
            elif abs_margin < 17:
                margin_cat = 'twoscore'
            elif abs_margin < 24:
                margin_cat = 'threescore'
            else:
                margin_cat = 'blowout'
            
            if entering_margin > 0:
                margin_cat += '_fav'
            elif entering_margin < 0:
                margin_cat += '_dog'
            
            context = f"{total_cat}_{margin_cat}"
            
            # Weight this observation across BOTH distributions
            competitive_weight = weight * (1 - garbage_prob)
            garbage_weight = weight * garbage_prob
            
            # Add to competitive distribution
            if context not in self.conditional_q4_competitive:
                self.conditional_q4_competitive[context] = {}
                self.conditional_q4_sample_sizes_competitive[context] = 0
            
            if q4_score not in self.conditional_q4_competitive[context]:
                self.conditional_q4_competitive[context][q4_score] = 0
            
            self.conditional_q4_competitive[context][q4_score] += competitive_weight
            self.conditional_q4_sample_sizes_competitive[context] += competitive_weight
            
            # Add to garbage distribution
            if context not in self.conditional_q4_garbage:
                self.conditional_q4_garbage[context] = {}
                self.conditional_q4_sample_sizes_garbage[context] = 0
            
            if q4_score not in self.conditional_q4_garbage[context]:
                self.conditional_q4_garbage[context][q4_score] = 0
            
            self.conditional_q4_garbage[context][q4_score] += garbage_weight
            self.conditional_q4_sample_sizes_garbage[context] += garbage_weight
        
        # Normalize both distributions
        for context in self.conditional_q4_competitive:
            total = sum(self.conditional_q4_competitive[context].values())
            if total > 0:
                for score in self.conditional_q4_competitive[context]:
                    self.conditional_q4_competitive[context][score] /= total
        
        for context in self.conditional_q4_garbage:
            total = sum(self.conditional_q4_garbage[context].values())
            if total > 0:
                for score in self.conditional_q4_garbage[context]:
                    self.conditional_q4_garbage[context][score] /= total
        
        print(f"    Competitive contexts: {len(self.conditional_q4_competitive)}")
        print(f"    Garbage contexts: {len(self.conditional_q4_garbage)}")
        
        # Also keep old Q4 distribution for backwards compatibility
        for _, game in df.iterrows():
            h1_fav = game['h1_fav_score']
            h1_dog = game['h1_dog_score']
            q3_score = game['q3_score_combo']
            q4_score = game['q4_score_combo']
            weight = game['temporal_weight']
            
            h1_score = f"{h1_fav}-{h1_dog}"
            h1_context = self._categorize_halftime_outcome(h1_score)
            q3_context = self._categorize_quarter_outcome(q3_score)
            combined_context = f"{h1_context}|{q3_context}"
            
            if combined_context not in self.conditional_q4_given_h1_q3:
                self.conditional_q4_given_h1_q3[combined_context] = {}
                self.conditional_q4_sample_sizes[combined_context] = 0
            
            if q4_score not in self.conditional_q4_given_h1_q3[combined_context]:
                self.conditional_q4_given_h1_q3[combined_context][q4_score] = 0
            
            self.conditional_q4_given_h1_q3[combined_context][q4_score] += weight
            self.conditional_q4_sample_sizes[combined_context] += weight
        
        # Normalize P(Q4 | H1, Q3)
        for combined_context in self.conditional_q4_given_h1_q3:
            total = sum(self.conditional_q4_given_h1_q3[combined_context].values())
            if total > 0:
                for q4_score in self.conditional_q4_given_h1_q3[combined_context]:
                    self.conditional_q4_given_h1_q3[combined_context][q4_score] /= total
        
        sample_sizes_q4 = list(self.conditional_q4_sample_sizes.values())
        print(f"    P(Q4|H1,Q3) contexts: {len(self.conditional_q4_given_h1_q3)}")
        print(f"    Sample sizes - Min: {min(sample_sizes_q4):.0f}, Median: {np.median(sample_sizes_q4):.0f}, Max: {max(sample_sizes_q4):.0f}")
    
    def _categorize_quarter_outcome(self, score_str):
        """
        Categorize a quarter outcome into a context bucket.
        Used for Q1, Q2, Q4.
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
    
    def _categorize_halftime_outcome(self, h1_score_str):
        """
        Enhanced categorization for halftime score that emphasizes margin magnitude.
        Used for conditioning Q3 and Q4 where possession and catch-up dynamics matter.
        """
        try:
            fav_score, dog_score = map(int, h1_score_str.split('-'))
        except:
            return 'unknown'
        
        margin = fav_score - dog_score
        total = fav_score + dog_score
        
        # Categorize total
        if total <= 14:
            total_cat = 'low'
        elif total <= 35:
            total_cat = 'med'
        else:
            total_cat = 'high'
        
        # Enhanced margin categorization with more granularity for Q3/Q4
        if margin == 0:
            margin_cat = 'tie'
        elif margin >= 21:
            margin_cat = 'blowout_fav'
        elif margin >= 14:
            margin_cat = 'twocore_fav'
        elif margin >= 7:
            margin_cat = 'onescore_fav'
        elif margin > 0:
            margin_cat = 'close_fav'
        elif margin >= -7:
            margin_cat = 'close_dog'
        elif margin >= -14:
            margin_cat = 'onescore_dog'
        elif margin >= -21:
            margin_cat = 'twoscore_dog'
        else:
            margin_cat = 'blowout_dog'
        
        return f"{total_cat}_{margin_cat}"
    
    def calculate_empirical_distribution(self):
        """Calculate empirical distributions for all quarters with temporal weighting."""
        if self.historical_data is None:
            return
        
        df = self.historical_data
        total_weight = df['temporal_weight'].sum()
        
        # Calculate adaptive anchor threshold
        anchor_threshold = max(
            self.MIN_ANCHOR_OCCURRENCES,
            int(self.ANCHOR_PERCENTAGE * len(df))
        )
        self.anchor_threshold = anchor_threshold
        
        print(f"\nAnchor Threshold Calculation:")
        print(f"  Total games: {len(df)}")
        print(f"  Percentage-based threshold (0.75%): {int(self.ANCHOR_PERCENTAGE * len(df))}")
        print(f"  Floor: {self.MIN_ANCHOR_OCCURRENCES}")
        print(f"  Selected threshold: {anchor_threshold}")
        
        for q in [1, 2, 3, 4]:
            # Calculate weighted empirical distributions
            score_col = f'q{q}_score_combo'
            
            # Weighted counts
            score_weights = df.groupby(score_col)['temporal_weight'].sum()
            score_counts_dict = score_weights.to_dict()
            
            # Store raw empirical distribution
            empirical_dist_attr = f'empirical_distribution_q{q}'
            setattr(self, empirical_dist_attr, score_counts_dict)
            
            # Standardize empirical distributions
            standardized_empirical_attr = f'standardized_empirical_dist_q{q}'
            standardized_empirical = {}
            
            for score, weighted_count in score_counts_dict.items():
                prob = weighted_count / total_weight
                standardized_empirical[score] = prob
            
            setattr(self, standardized_empirical_attr, standardized_empirical)
            
            print(f"\nQ{q} Empirical Distribution:")
            print(f"  Unique scores: {len(score_counts_dict)}")
            print(f"  Total weighted observations: {total_weight:.1f}")
            
            # Identify anchor scores
            anchor_scores_attr = f'anchor_scores_q{q}'
            anchor_scores = [
                score for score, weighted_count in score_counts_dict.items()
                if weighted_count >= anchor_threshold
            ]
            setattr(self, anchor_scores_attr, anchor_scores)
            
            print(f"  Anchor scores (>= {anchor_threshold} weighted occurrences): {len(anchor_scores)}")
            if len(anchor_scores) > 0:
                print(f"  Top 5 anchor scores: {sorted(anchor_scores, key=lambda s: score_counts_dict[s], reverse=True)[:5]}")
    
    def get_score_pattern(self, score_str):
        """
        Extract score pattern for bucketing similar rare scores.
        Returns tuple of (total_bucket, margin_bucket, pattern_signature).
        """
        try:
            fav_score, dog_score = map(int, score_str.split('-'))
        except:
            return None
        
        total = fav_score + dog_score
        margin = fav_score - dog_score
        
        # Bucket total
        if total <= 7:
            total_bucket = 'low'
        elif total <= 14:
            total_bucket = 'low_mid'
        elif total <= 21:
            total_bucket = 'mid'
        elif total <= 28:
            total_bucket = 'mid_high'
        else:
            total_bucket = 'high'
        
        # Bucket margin
        abs_margin = abs(margin)
        if abs_margin == 0:
            margin_bucket = 'tie'
        elif abs_margin <= 3:
            margin_bucket = 'close'
        elif abs_margin <= 7:
            margin_bucket = 'one_score'
        elif abs_margin <= 14:
            margin_bucket = 'two_score'
        else:
            margin_bucket = 'blowout'
        
        # Add direction
        if margin > 0:
            margin_bucket += '_fav'
        elif margin < 0:
            margin_bucket += '_dog'
        
        # Pattern signature
        pattern = f"{total_bucket}_{margin_bucket}"
        
        return (total_bucket, margin_bucket, pattern)
    
    def find_nearest_anchor(self, target_score, anchor_scores):
        """
        Find the nearest anchor score to target_score using Euclidean distance.
        Returns (nearest_anchor, distance).
        """
        try:
            target_fav, target_dog = map(int, target_score.split('-'))
        except:
            return None, float('inf')
        
        target_vec = np.array([target_fav, target_dog])
        
        min_dist = float('inf')
        nearest_anchor = None
        
        for anchor in anchor_scores:
            try:
                anchor_fav, anchor_dog = map(int, anchor.split('-'))
                anchor_vec = np.array([anchor_fav, anchor_dog])
                
                dist = np.linalg.norm(target_vec - anchor_vec)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_anchor = anchor
            except:
                continue
        
        return nearest_anchor, min_dist
    
    def get_coefficient_bounds(self, score_str, n_occurrences):
        """
        Generate dynamic coefficient bounds based on score characteristics and sample size.
        Larger samples allow tighter bounds around football-realistic values.
        Returns: (spread_lower, spread_upper, total_lower, total_upper)
        """
        try:
            fav_score, dog_score = map(int, score_str.split('-'))
        except:
            return (-10, 10, -10, 10)
        
        total = fav_score + dog_score
        margin = fav_score - dog_score
        
        # Base bounds: wider for rare scores, tighter for common scores
        if n_occurrences >= 200:
            base_width = 3.0
        elif n_occurrences >= 100:
            base_width = 5.0
        elif n_occurrences >= 50:
            base_width = 7.0
        else:
            base_width = 10.0
        
        # Adjust based on score characteristics
        if total <= 7:
            # Low-scoring: tighter bounds
            spread_bound = base_width * 0.8
            total_bound = base_width * 0.6
        elif total >= 28:
            # High-scoring: slightly wider for spread
            spread_bound = base_width * 1.2
            total_bound = base_width * 1.0
        else:
            # Mid-range: use base
            spread_bound = base_width
            total_bound = base_width
        
        # Special handling for ties and blowouts
        if abs(margin) == 0:
            spread_bound *= 0.7
        elif abs(margin) > 21:
            spread_bound *= 1.3
        
        return (-spread_bound, spread_bound, -total_bound, total_bound)
    
    def fit_model_for_quarter(self, quarter):
        """
        Fit anchor + inheritance logistic regression models for a specific quarter
        with Elastic Net regularization and L-BFGS-B optimization.
        """
        empirical_dist = getattr(self, f'empirical_distribution_q{quarter}')
        anchor_scores = getattr(self, f'anchor_scores_q{quarter}')
        
        if len(empirical_dist) == 0:
            print(f"No empirical data for Q{quarter}")
            return
        
        print(f"\n{'='*60}")
        print(f"Q{quarter} Model Fitting")
        print(f"{'='*60}")
        
        # Fit individual models for anchor scores
        anchor_models = {}
        
        print(f"\nFitting {len(anchor_scores)} anchor score models...")
        
        for i, score in enumerate(anchor_scores, 1):
            if i % 20 == 0 or i == len(anchor_scores):
                print(f"  Progress: {i}/{len(anchor_scores)} anchor scores fitted")
            
            try:
                fav_score, dog_score = map(int, score.split('-'))
            except:
                continue
            
            # Create binary target
            df = self.historical_data.copy()
            df[f'target_{score}'] = (df[f'q{quarter}_score_combo'] == score).astype(float)
            
            X = df[['pregame_spread', 'pregame_total']].values
            y = df[f'target_{score}'].values
            weights = df['temporal_weight'].values
            
            n_occurrences = empirical_dist.get(score, 0)
            
            # Initial guess
            spread_coef_init = (fav_score - dog_score) * 0.02
            total_coef_init = ((fav_score + dog_score) / self.typical_game_score - 1.0) * 0.5
            
            base_prob = n_occurrences / self.historical_data['temporal_weight'].sum()
            if base_prob <= 0 or base_prob >= 1:
                base_prob = np.clip(base_prob, 0.001, 0.999)
            intercept_init = np.log(base_prob / (1 - base_prob))
            
            initial_params = np.array([intercept_init, spread_coef_init, total_coef_init])
            
            # Regularization strengths
            alpha_l1 = 0.005
            alpha_l2 = 0.01
            
            # Coefficient bounds
            spread_lower, spread_upper, total_lower, total_upper = self.get_coefficient_bounds(score, n_occurrences)
            
            bounds = [
                (-10, 10),
                (spread_lower, spread_upper),
                (total_lower, total_upper)
            ]
            
            def negative_log_likelihood(params):
                """Elastic Net penalized negative log-likelihood."""
                intercept, spread_coef, total_coef = params
                
                logits = intercept + spread_coef * X[:, 0] + total_coef * X[:, 1]
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                
                # Weighted negative log-likelihood
                nll = -np.sum(weights * (y * np.log(probs) + (1 - y) * np.log(1 - probs)))
                
                # Elastic Net penalty
                l1_penalty = alpha_l1 * (abs(spread_coef) + abs(total_coef))
                l2_penalty = alpha_l2 * (spread_coef**2 + total_coef**2)
                
                return nll + l1_penalty + l2_penalty
            
            result = minimize(
                negative_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200}
            )
            
            if result.success:
                anchor_models[score] = {
                    'params': result.x,
                    'n_occurrences': n_occurrences
                }
        
        print(f"Successfully fitted {len(anchor_models)} anchor models")
        
        # Fit models for rare scores using inheritance
        rare_scores = [score for score in empirical_dist.keys() if score not in anchor_scores]
        
        print(f"\nFitting {len(rare_scores)} rare score models using inheritance...")
        
        rare_models = {}
        inheritance_decay = 0.8
        
        for i, score in enumerate(rare_scores, 1):
            if i % 50 == 0 or i == len(rare_scores):
                print(f"  Progress: {i}/{len(rare_scores)} rare scores fitted")
            
            # Find nearest anchor
            nearest_anchor, distance = self.find_nearest_anchor(score, anchor_scores)
            
            if nearest_anchor is None or nearest_anchor not in anchor_models:
                continue
            
            # Inherit coefficients from nearest anchor
            inherited_params = anchor_models[nearest_anchor]['params'].copy()
            
            # Apply inheritance decay
            inherited_params[1:] *= inheritance_decay
            
            # Now fit with inherited coefficients as prior
            try:
                fav_score, dog_score = map(int, score.split('-'))
            except:
                continue
            
            df = self.historical_data.copy()
            df[f'target_{score}'] = (df[f'q{quarter}_score_combo'] == score).astype(float)
            
            X = df[['pregame_spread', 'pregame_total']].values
            y = df[f'target_{score}'].values
            weights = df['temporal_weight'].values
            
            n_occurrences = empirical_dist.get(score, 0)
            
            # Strong regularization toward inherited params
            alpha_l1 = 0.01
            alpha_l2 = 0.02
            prior_strength = 0.1
            
            spread_lower, spread_upper, total_lower, total_upper = self.get_coefficient_bounds(score, n_occurrences)
            
            bounds = [
                (-10, 10),
                (spread_lower, spread_upper),
                (total_lower, total_upper)
            ]
            
            def negative_log_likelihood_inherited(params):
                """NLL with strong prior toward inherited params."""
                intercept, spread_coef, total_coef = params
                
                logits = intercept + spread_coef * X[:, 0] + total_coef * X[:, 1]
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                
                # Weighted negative log-likelihood
                nll = -np.sum(weights * (y * np.log(probs) + (1 - y) * np.log(1 - probs)))
                
                # Elastic Net penalty
                l1_penalty = alpha_l1 * (abs(spread_coef) + abs(total_coef))
                l2_penalty = alpha_l2 * (spread_coef**2 + total_coef**2)
                
                # Prior toward inherited params
                prior_penalty = prior_strength * np.sum((params[1:] - inherited_params[1:])**2)
                
                return nll + l1_penalty + l2_penalty + prior_penalty
            
            result = minimize(
                negative_log_likelihood_inherited,
                inherited_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success:
                rare_models[score] = {
                    'params': result.x,
                    'n_occurrences': n_occurrences,
                    'inherited_from': nearest_anchor
                }
        
        print(f"Successfully fitted {len(rare_models)} rare score models")
        
        # Combine all models
        all_models = {}
        all_models.update(anchor_models)
        all_models.update(rare_models)
        
        model_params_attr = f'model_params_q{quarter}'
        setattr(self, model_params_attr, all_models)
        
        print(f"\nQ{quarter} Model Summary:")
        print(f"  Anchor models: {len(anchor_models)}")
        print(f"  Inherited models: {len(rare_models)}")
        print(f"  Total models: {len(all_models)}")
    
    def fit_model(self):
        """Fit logistic regression models for all four quarters."""
        print("\nFitting Anchor + Inheritance Models for All Quarters")
        print("="*60)
        
        for quarter in [1, 2, 3, 4]:
            self.fit_model_for_quarter(quarter)
        
        print(f"\n{'='*60}")
        print("All Quarter Models Fitted Successfully")
        print(f"{'='*60}\n")
    
    def predict_standardized_probability(self, spread, total, quarter):
        """
        Predict probabilities for all scores in specified quarter.
        Returns distribution in favored-underdog format.
        """
        model_params = getattr(self, f'model_params_q{quarter}')
        
        if len(model_params) == 0:
            print(f"No model for Q{quarter}")
            return {}
        
        predictions = {}
        
        for score, model_info in model_params.items():
            params = model_info['params']
            intercept, spread_coef, total_coef = params
            
            # Calculate logit
            logit = intercept + spread_coef * spread + total_coef * total
            
            # Convert to probability
            prob = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
            
            if prob > 1e-6:
                predictions[score] = prob
        
        # Normalize
        total_prob = sum(predictions.values())
        if total_prob > 0:
            predictions = {score: prob / total_prob for score, prob in predictions.items()}
        
        return predictions
    
    def adjust_quarter_for_correlation(self, current_q_dist, prior_context, correlation_map, sample_size_map):
        """
        Adjust quarter distribution using Bayesian credibility weighting with Laplace smoothing.
        Laplace smoothing prevents zero probabilities from eliminating valid scores.
        
        Args:
            current_q_dist: Base distribution for current quarter
            prior_context: Context string from prior quarter(s)
            correlation_map: Conditional probability map
            sample_size_map: Sample sizes for each context
        
        Returns:
            Adjusted distribution with credibility-weighted correlation
        """
        if prior_context not in correlation_map:
            return current_q_dist
        
        raw_conditional_dist = correlation_map[prior_context]
        n = sample_size_map.get(prior_context, 0)
        
        # LAPLACE SMOOTHING: Add pseudocount to prevent zeros
        all_scores = set(current_q_dist.keys())
        smoothed_conditional_dist = {}
        
        raw_total = sum(raw_conditional_dist.values())
        vocab_size = len(all_scores)
        smoothed_total = raw_total + self.LAPLACE_ALPHA * vocab_size
        
        for score in all_scores:
            raw_count = raw_conditional_dist.get(score, 0.0)
            # Add pseudocount: ensures no score has exactly zero probability
            smoothed_prob = (raw_count + self.LAPLACE_ALPHA) / smoothed_total
            smoothed_conditional_dist[score] = smoothed_prob
        
        # Bayesian credibility weight
        credibility = n / (n + self.CREDIBILITY_K)
        
        # Blend with credibility weighting
        adjusted_dist = {}
        
        for score in all_scores:
            base_prob = current_q_dist[score]
            cond_prob = smoothed_conditional_dist[score]
            
            # Credibility-weighted blend
            # Now cond_prob is NEVER exactly zero (minimum is alpha / smoothed_total)
            adjusted_prob = (1 - credibility) * base_prob + credibility * cond_prob
            
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
    
    def simulate_game_with_correlation(self, pregame_spread, total, n_sims=100000, 
                                      home_favored=True, include_q4_garbage=True):
        """
        Simulate full game with quarter correlations and optional Q4 garbage time effects.
        Uses conditional distributions for Q2, Q3, Q4 with Bayesian credibility weighting.
        """
        # Predict base Q1 distribution
        q1_base = self.predict_standardized_probability(pregame_spread, total, 1)
        
        # Initialize result storage
        all_simulations = []
        
        for _ in range(n_sims):
            # Sample Q1
            q1_scores = list(q1_base.keys())
            q1_probs = list(q1_base.values())
            q1_score = np.random.choice(q1_scores, p=q1_probs)
            
            # Get Q2 with correlation
            q1_context = self._categorize_quarter_outcome(q1_score)
            q2_base = self.predict_standardized_probability(pregame_spread, total, 2)
            q2_dist = self.adjust_quarter_for_correlation(
                q2_base, q1_context,
                self.conditional_q2_given_q1,
                self.conditional_q2_sample_sizes
            )
            
            q2_scores = list(q2_dist.keys())
            q2_probs = list(q2_dist.values())
            q2_score = np.random.choice(q2_scores, p=q2_probs)
            
            # Calculate H1 for Q3 conditioning
            q1_fav, q1_dog = map(int, q1_score.split('-'))
            q2_fav, q2_dog = map(int, q2_score.split('-'))
            h1_fav = q1_fav + q2_fav
            h1_dog = q1_dog + q2_dog
            h1_score = f"{h1_fav}-{h1_dog}"
            
            # Get Q3 with H1 correlation
            h1_context = self._categorize_halftime_outcome(h1_score)
            q3_base = self.predict_standardized_probability(pregame_spread, total, 3)
            q3_dist = self.adjust_quarter_for_correlation(
                q3_base, h1_context,
                self.conditional_q3_given_h1,
                self.conditional_q3_sample_sizes
            )
            
            q3_scores = list(q3_dist.keys())
            q3_probs = list(q3_dist.values())
            q3_score = np.random.choice(q3_scores, p=q3_probs)
            
            # Get Q4 with optional garbage time blending
            if include_q4_garbage:
                # Calculate entering Q4 margin for garbage time
                q3_fav, q3_dog = map(int, q3_score.split('-'))
                h1_margin = h1_fav - h1_dog
                q3_margin = q3_fav - q3_dog
                entering_margin = h1_margin + q3_margin
                
                # Get garbage probability
                garbage_prob = self.predict_garbage_probability(entering_margin)
                
                # Create context
                entering_total = h1_fav + h1_dog + q3_fav + q3_dog
                
                if entering_total <= 28:
                    total_cat = 'low'
                elif entering_total <= 56:
                    total_cat = 'med'
                else:
                    total_cat = 'high'
                
                abs_margin = abs(entering_margin)
                if abs_margin < 4:
                    margin_cat = 'close'
                elif abs_margin < 8:
                    margin_cat = 'onescore'
                elif abs_margin < 12:
                    margin_cat = 'onescore_plus'
                elif abs_margin < 17:
                    margin_cat = 'twoscore'
                elif abs_margin < 24:
                    margin_cat = 'threescore'
                else:
                    margin_cat = 'blowout'
                
                if entering_margin > 0:
                    margin_cat += '_fav'
                elif entering_margin < 0:
                    margin_cat += '_dog'
                
                context = f"{total_cat}_{margin_cat}"
                
                # Get base Q4
                q4_base = self.predict_standardized_probability(pregame_spread, total, 4)
                
                # Blend with competitive distribution
                competitive_dist = self.conditional_q4_competitive.get(context, {})
                competitive_n = self.conditional_q4_sample_sizes_competitive.get(context, 0)
                competitive_credibility = competitive_n / (competitive_n + self.CREDIBILITY_K)
                
                competitive_adjusted = {}
                all_scores = set(q4_base.keys()) | set(competitive_dist.keys())
                
                for score in all_scores:
                    base_prob = q4_base.get(score, 0)
                    cond_prob = competitive_dist.get(score, base_prob)
                    competitive_adjusted[score] = (1 - competitive_credibility) * base_prob + competitive_credibility * cond_prob
                
                # Blend with garbage distribution
                garbage_dist = self.conditional_q4_garbage.get(context, {})
                garbage_n = self.conditional_q4_sample_sizes_garbage.get(context, 0)
                garbage_credibility = garbage_n / (garbage_n + self.CREDIBILITY_K)
                
                garbage_adjusted = {}
                for score in all_scores:
                    base_prob = q4_base.get(score, 0)
                    cond_prob = garbage_dist.get(score, base_prob)
                    garbage_adjusted[score] = (1 - garbage_credibility) * base_prob + garbage_credibility * cond_prob
                
                # Smooth mixture based on garbage_prob
                q4_dist = {}
                for score in all_scores:
                    comp_prob = competitive_adjusted.get(score, 0)
                    garb_prob = garbage_adjusted.get(score, 0)
                    q4_dist[score] = (1 - garbage_prob) * comp_prob + garbage_prob * garb_prob
                
                # Normalize
                total_prob = sum(q4_dist.values())
                if total_prob > 0:
                    q4_dist = {score: prob/total_prob for score, prob in q4_dist.items()}
            else:
                # Use old method without garbage time
                h1_context = self._categorize_halftime_outcome(h1_score)
                q3_context = self._categorize_quarter_outcome(q3_score)
                combined_context = f"{h1_context}|{q3_context}"
                
                q4_base = self.predict_standardized_probability(pregame_spread, total, 4)
                q4_dist = self.adjust_quarter_for_correlation(
                    q4_base, combined_context,
                    self.conditional_q4_given_h1_q3,
                    self.conditional_q4_sample_sizes
                )
            
            q4_scores = list(q4_dist.keys())
            q4_probs = list(q4_dist.values())
            q4_score = np.random.choice(q4_scores, p=q4_probs)
            
            # Parse all quarters
            q1_fav, q1_dog = map(int, q1_score.split('-'))
            q2_fav, q2_dog = map(int, q2_score.split('-'))
            q3_fav, q3_dog = map(int, q3_score.split('-'))
            q4_fav, q4_dog = map(int, q4_score.split('-'))
            
            # Calculate game total
            game_fav = q1_fav + q2_fav + q3_fav + q4_fav
            game_dog = q1_dog + q2_dog + q3_dog + q4_dog
            
            all_simulations.append({
                'q1_fav': q1_fav, 'q1_dog': q1_dog,
                'q2_fav': q2_fav, 'q2_dog': q2_dog,
                'q3_fav': q3_fav, 'q3_dog': q3_dog,
                'q4_fav': q4_fav, 'q4_dog': q4_dog,
                'game_fav': game_fav, 'game_dog': game_dog
            })
        
        return pd.DataFrame(all_simulations)
    
    def convolve_quarters(self, dist1, dist2):
        """
        Convolve two quarter distributions to get combined distribution.
        Used for creating full-game distributions from quarter predictions.
        """
        combined = {}
        
        for score1, prob1 in dist1.items():
            fav1, dog1 = map(int, score1.split('-'))
            
            for score2, prob2 in dist2.items():
                fav2, dog2 = map(int, score2.split('-'))
                
                # Combined score
                combined_fav = fav1 + fav2
                combined_dog = dog1 + dog2
                combined_score = f"{combined_fav}-{combined_dog}"
                
                # Combined probability
                combined_prob = prob1 * prob2
                
                if combined_score not in combined:
                    combined[combined_score] = 0
                combined[combined_score] += combined_prob
        
        return combined
    
    def calibrate_full_game_distribution(self, pregame_spread, total, max_iterations=10, n_sims=5000):
        """
        Iteratively calibrate quarter distributions to match pregame lines.
        UPDATED: Targets 50% Win Probability (Median) instead of Expected Points (Mean).
        """
        # Get base predictions for each quarter
        q1_base = self.predict_standardized_probability(pregame_spread, total, 1)
        q2_base = self.predict_standardized_probability(pregame_spread, total, 2)
        q3_base = self.predict_standardized_probability(pregame_spread, total, 3)
        q4_base = self.predict_standardized_probability(pregame_spread, total, 4)
        
        # Initialize working distributions
        q1_dist = q1_base.copy()
        q2_dist = q2_base.copy()
        q3_dist = q3_base.copy()
        q4_dist = q4_base.copy()
        
        home_favored = pregame_spread < 0
        target_spread_value = abs(pregame_spread)
        
        # Calculate OT adjustment
        if abs(pregame_spread) <= 3:
            ot_prob = 0.06
        elif abs(pregame_spread) <= 7:
            ot_prob = 0.04
        else:
            ot_prob = 0.02
        expected_ot_points = ot_prob * 14
        target_total_value = total - expected_ot_points
        
        print(f"\nCalibrating to 50% Probability: Spread={target_spread_value:.1f}, Reg Total={target_total_value:.1f}")
        
        # Learning rates for probability calibration (higher than point-based)
        spread_learning_rate = 0.20
        total_learning_rate = 0.15
        
        # Define logit transform functions
        def prob_to_logit(prob):
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            return np.log(prob / (1 - prob))
        
        def logit_to_prob(logit):
            return 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
        
        def calculate_probabilities(dist):
            """Calculate Cover % and Over % from distribution."""
            cover_prob = 0.0
            over_prob = 0.0
            
            for score, prob in dist.items():
                fav_pts, dog_pts = map(int, score.split('-'))
                margin = fav_pts - dog_pts
                score_total = fav_pts + dog_pts
                
                # Spread Logic (Handle Pushes as 0.5)
                if margin > target_spread_value:
                    cover_prob += prob
                elif margin == target_spread_value:
                    cover_prob += (0.5 * prob)
                
                # Total Logic (Handle Pushes as 0.5)
                if score_total > target_total_value:
                    over_prob += prob
                elif score_total == target_total_value:
                    over_prob += (0.5 * prob)
            
            return cover_prob, over_prob
        
        for iteration in range(max_iterations):
            # Convolve all quarters
            h1_dist = self.convolve_quarters(q1_dist, q2_dist)
            h2_dist = self.convolve_quarters(q3_dist, q4_dist)
            game_dist = self.convolve_quarters(h1_dist, h2_dist)
            
            # Calculate current probabilities
            current_cover_pct, current_over_pct = calculate_probabilities(game_dist)
            
            # Error is deviation from 50%
            spread_error = current_cover_pct - 0.50
            total_error = current_over_pct - 0.50
            
            print(f"  Iteration {iteration+1}: Cover={current_cover_pct*100:.2f}% (Err: {spread_error*100:+.2f}%), Over={current_over_pct*100:.2f}% (Err: {total_error*100:+.2f}%)")
            
            # Check convergence (Within 49.5% - 50.5%)
            if abs(spread_error) < 0.005 and abs(total_error) < 0.005:
                print(f"  Converged after {iteration+1} iterations")
                break
            
            # Calculate adjustments in logit space
            for q, q_dist in [(1, q1_dist), (2, q2_dist), (3, q3_dist), (4, q4_dist)]:
                q_pct = getattr(self, f'Q{q}_PERCENTAGE_POINTS')
                
                # Convert to logit space
                logits = {}
                for score, prob in q_dist.items():
                    logits[score] = prob_to_logit(prob)
                
                # Adjust based on errors
                for score in q_dist.keys():
                    fav_pts, dog_pts = map(int, score.split('-'))
                    score_total = fav_pts + dog_pts
                    score_margin = fav_pts - dog_pts
                    
                    # Use score_margin/total as directional vector for probability shift
                    spread_adjustment = -spread_learning_rate * spread_error * score_margin * q_pct
                    total_adjustment = -total_learning_rate * total_error * score_total * q_pct
                    
                    # Clamp updates
                    total_update = spread_adjustment + total_adjustment
                    total_update = np.clip(total_update, -0.5, 0.5)
                    
                    logits[score] += total_update
                
                # Convert back to probabilities and normalize
                new_probs = {score: logit_to_prob(logit) for score, logit in logits.items()}
                total_prob = sum(new_probs.values())
                if total_prob > 0:
                    new_probs = {score: prob / total_prob for score, prob in new_probs.items()}
                
                # Update distribution
                if q == 1:
                    q1_dist = new_probs
                elif q == 2:
                    q2_dist = new_probs
                elif q == 3:
                    q3_dist = new_probs
                else:
                    q4_dist = new_probs
        
        print(f"  Final: Cover={current_cover_pct*100:.2f}%, Over={current_over_pct*100:.2f}%\n")
        
        print("  Running final correlated simulation with calibrated marginals...")
        sim_results = self.simulate_game_with_calibrated_marginals(
            q1_dist, q2_dist, q3_dist, q4_dist,
            pregame_spread, total, n_sims=n_sims
        )
        
        return sim_results
    
    def simulate_game_with_calibrated_marginals(self, q1_dist, q2_dist, q3_dist, q4_dist,
                                                pregame_spread, total, n_sims=10000):
        """
        Run simulation using CALIBRATED base distributions.
        Captures correlations and garbage time effects while maintaining market efficiency.
        """
        all_simulations = []
        
        # Pre-convert for faster sampling
        q1_scores = list(q1_dist.keys())
        q1_probs = list(q1_dist.values())
        
        for _ in range(n_sims):
            # Sample Q1 from calibrated distribution
            q1_score = np.random.choice(q1_scores, p=q1_probs)
            
            # Sample Q2 with Q1 correlation
            q1_context = self._categorize_quarter_outcome(q1_score)
            q2_corr_dist = self.adjust_quarter_for_correlation(
                q2_dist, q1_context,
                self.conditional_q2_given_q1,
                self.conditional_q2_sample_sizes
            )
            q2_scores = list(q2_corr_dist.keys())
            q2_probs = list(q2_corr_dist.values())
            q2_score = np.random.choice(q2_scores, p=q2_probs)
            
            # Calculate H1 for Q3 conditioning
            q1_fav, q1_dog = map(int, q1_score.split('-'))
            q2_fav, q2_dog = map(int, q2_score.split('-'))
            h1_fav = q1_fav + q2_fav
            h1_dog = q1_dog + q2_dog
            h1_score = f"{h1_fav}-{h1_dog}"
            
            # Sample Q3 with H1 correlation
            h1_context = self._categorize_halftime_outcome(h1_score)
            q3_corr_dist = self.adjust_quarter_for_correlation(
                q3_dist, h1_context,
                self.conditional_q3_given_h1,
                self.conditional_q3_sample_sizes
            )
            q3_scores = list(q3_corr_dist.keys())
            q3_probs = list(q3_corr_dist.values())
            q3_score = np.random.choice(q3_scores, p=q3_probs)
            
            # Sample Q4 with garbage time blending
            q3_fav, q3_dog = map(int, q3_score.split('-'))
            h1_margin = h1_fav - h1_dog
            q3_margin = q3_fav - q3_dog
            entering_margin = h1_margin + q3_margin
            
            # Get garbage probability
            garbage_prob = self.predict_garbage_probability(entering_margin)
            
            # Create context
            entering_total = h1_fav + h1_dog + q3_fav + q3_dog
            
            if entering_total <= 28:
                total_cat = 'low'
            elif entering_total <= 56:
                total_cat = 'med'
            else:
                total_cat = 'high'
            
            abs_margin = abs(entering_margin)
            if abs_margin < 4:
                margin_cat = 'close'
            elif abs_margin < 8:
                margin_cat = 'onescore'
            elif abs_margin < 12:
                margin_cat = 'onescore_plus'
            elif abs_margin < 17:
                margin_cat = 'twoscore'
            elif abs_margin < 24:
                margin_cat = 'threescore'
            else:
                margin_cat = 'blowout'
            
            if entering_margin > 0:
                margin_cat += '_fav'
            elif entering_margin < 0:
                margin_cat += '_dog'
            
            context = f"{total_cat}_{margin_cat}"
            
            # Blend with competitive distribution
            competitive_dist = self.conditional_q4_competitive.get(context, {})
            competitive_n = self.conditional_q4_sample_sizes_competitive.get(context, 0)
            competitive_credibility = competitive_n / (competitive_n + self.CREDIBILITY_K)
            
            competitive_adjusted = {}
            all_scores = set(q4_dist.keys()) | set(competitive_dist.keys())
            
            for score in all_scores:
                base_prob = q4_dist.get(score, 0)
                cond_prob = competitive_dist.get(score, base_prob)
                competitive_adjusted[score] = (1 - competitive_credibility) * base_prob + competitive_credibility * cond_prob
            
            # Blend with garbage distribution
            garbage_dist = self.conditional_q4_garbage.get(context, {})
            garbage_n = self.conditional_q4_sample_sizes_garbage.get(context, 0)
            garbage_credibility = garbage_n / (garbage_n + self.CREDIBILITY_K)
            
            garbage_adjusted = {}
            for score in all_scores:
                base_prob = q4_dist.get(score, 0)
                cond_prob = garbage_dist.get(score, base_prob)
                garbage_adjusted[score] = (1 - garbage_credibility) * base_prob + garbage_credibility * cond_prob
            
            # Smooth mixture based on garbage_prob
            q4_final_dist = {}
            for score in all_scores:
                comp_prob = competitive_adjusted.get(score, 0)
                garb_prob = garbage_adjusted.get(score, 0)
                q4_final_dist[score] = (1 - garbage_prob) * comp_prob + garbage_prob * garb_prob
            
            # Normalize
            total_prob = sum(q4_final_dist.values())
            if total_prob > 0:
                q4_final_dist = {score: prob/total_prob for score, prob in q4_final_dist.items()}
            
            q4_scores = list(q4_final_dist.keys())
            q4_probs = list(q4_final_dist.values())
            q4_score = np.random.choice(q4_scores, p=q4_probs)
            
            # Parse Q4
            q4_fav, q4_dog = map(int, q4_score.split('-'))
            
            # Calculate game totals
            game_fav = q1_fav + q2_fav + q3_fav + q4_fav
            game_dog = q1_dog + q2_dog + q3_dog + q4_dog
            
            all_simulations.append({
                'q1_fav': q1_fav, 'q1_dog': q1_dog,
                'q2_fav': q2_fav, 'q2_dog': q2_dog,
                'q3_fav': q3_fav, 'q3_dog': q3_dog,
                'q4_fav': q4_fav, 'q4_dog': q4_dog,
                'game_fav': game_fav, 'game_dog': game_dog
            })
        
        sim_df = pd.DataFrame(all_simulations)
        
        # Convert simulation results to probability distributions
        final_distributions = {
            'q1': self._df_to_dist(sim_df, 'q1_fav', 'q1_dog'),
            'q2': self._df_to_dist(sim_df, 'q2_fav', 'q2_dog'),
            'q3': self._df_to_dist(sim_df, 'q3_fav', 'q3_dog'),
            'q4': self._df_to_dist(sim_df, 'q4_fav', 'q4_dog'),
            'game': self._df_to_dist(sim_df, 'game_fav', 'game_dog')
        }
        
        # Calculate H1 and H2
        sim_df['h1_fav'] = sim_df['q1_fav'] + sim_df['q2_fav']
        sim_df['h1_dog'] = sim_df['q1_dog'] + sim_df['q2_dog']
        sim_df['h2_fav'] = sim_df['q3_fav'] + sim_df['q4_fav']
        sim_df['h2_dog'] = sim_df['q3_dog'] + sim_df['q4_dog']
        
        final_distributions['h1'] = self._df_to_dist(sim_df, 'h1_fav', 'h1_dog')
        final_distributions['h2'] = self._df_to_dist(sim_df, 'h2_fav', 'h2_dog')
        
        return final_distributions
    
    def _df_to_dist(self, df, fav_col, dog_col):
        """Convert simulation dataframe to probability distribution."""
        score_counts = df.groupby([fav_col, dog_col]).size()
        total_sims = len(df)
        
        dist = {}
        for (fav, dog), count in score_counts.items():
            score = f"{fav}-{dog}"
            dist[score] = count / total_sims
        
        return dist
    
    def predict(self, pregame_spread, total, debug=False, n_sims=5000, max_calibration_iterations=60):
        """
        Generate complete quarter-by-quarter predictions.
        
        Args:
            pregame_spread: Negative if home favored, positive if away favored
            total: Game total points
            debug: If True, print detailed information
            n_sims: Number of simulations for correlation analysis
            max_calibration_iterations: Maximum iterations for calibration
        
        Returns:
            Dictionary with quarter and game distributions
        """
        home_favored = pregame_spread < 0
        
        if debug:
            print(f"\nPrediction Request:")
            print(f"  Pregame Spread: {pregame_spread:+.1f} ({'Home' if home_favored else 'Away'} favored)")
            print(f"  Pregame Total: {total:.1f}")
        
        # Calibrate distributions
        calibrated = self.calibrate_full_game_distribution(
            pregame_spread, total,
            max_iterations=max_calibration_iterations,
            n_sims=n_sims
        )
        
        # Convert to home-away format
        result = {}
        for key in ['q1', 'q2', 'q3', 'q4', 'h1', 'h2', 'game']:
            result[key] = self.convert_to_home_away_format(calibrated[key], home_favored)
        
        if debug:
            print("\n" + "="*80)
            print("Q1 RESULTS")
            print("="*80)
            print("\nTop 10 Q1 Outcomes:")
            sorted_q1 = sorted(result['q1'].items(), key=lambda x: x[1], reverse=True)
            for i, (score, prob) in enumerate(sorted_q1[:10], 1):
                print(f"  {i:2d}. {score:>7}: {prob*100:>6.2f}%")
            
            # Calculate Q1 tie probability
            q1_tie_prob = sum(prob for score, prob in result['q1'].items() if score.split('-')[0] == score.split('-')[1])
            print(f"\nP(Q1 Tie): {q1_tie_prob*100:.2f}%")
            
            # Calculate Q1 over 12.5 (13+)
            q1_over_12_5 = sum(prob for score, prob in result['q1'].items() 
                               if int(score.split('-')[0]) + int(score.split('-')[1]) >= 13)
            print(f"P(Q1 Over 12.5): {q1_over_12_5*100:.2f}%")
            
            # Calculate specific tie scores
            q1_7_7 = result['q1'].get('7-7', 0)
            q1_14_14 = result['q1'].get('14-14', 0)
            print(f"\nP(Q1 = 7-7): {q1_7_7*100:.2f}%")
            print(f"P(Q1 = 14-14): {q1_14_14*100:.2f}%")
            print(f"P(Q1 = 7-7 or 14-14): {(q1_7_7 + q1_14_14)*100:.2f}%")
            
            print("\n" + "="*80)
            print("FULL GAME RESULTS")
            print("="*80)
            print("\nTop 10 Game Outcomes:")
            sorted_game = sorted(result['game'].items(), key=lambda x: x[1], reverse=True)
            for i, (score, prob) in enumerate(sorted_game[:10], 1):
                print(f"  {i:2d}. {score:>9}: {prob*100:>6.2f}%")
        
        return result

def main():
    """Interactive mode for predictions"""
    load_dotenv()
    
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }
    
    predictor = CFBAllQuartersPredictor(db_config)
    
    print("Loading historical data...")
    if not predictor.load_historical_data():
        print("Failed to load data")
        return
    
    print("\nCalculating empirical distributions...")
    predictor.calculate_empirical_distribution()
    
    print("\nFitting models...")
    predictor.fit_model()
    
    print("\n" + "="*80)
    print("CFB QUARTER-BY-QUARTER PREDICTION MODEL")
    print("="*80)
    print("\nModel ready for predictions!")
    
    while True:
        print("\n" + "-"*80)
        try:
            spread_input = input("\nEnter pregame spread (negative=home favored, or 'q' to quit): ").strip()
            if spread_input.lower() == 'q':
                print("Exiting...")
                break
            
            pregame_spread = float(spread_input)
            total = float(input("Enter pregame total: ").strip())
            
            print("\nGenerating predictions...")
            predictions = predictor.predict(
                pregame_spread=pregame_spread,
                total=total,
                debug=True,
                n_sims=10000
            )
            
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()