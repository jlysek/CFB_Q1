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
    Predicts all four quarter score probabilities using Q1.py's coefficient inheritance approach.
    
    Key features:
    - Four independent quarter models (Q1, Q2, Q3, Q4)
    - Each uses anchor scores + coefficient inheritance (same as Q1.py)
    - Orthogonal features: feature_total and feature_margin
    - Convolution for combining quarters
    - Calibration to ensure full game matches Vegas spread/total
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
        
        self.historical_data = None
        self.typical_game_score = None
        
        # Quarter-specific empirical percentages
        self.Q1_PERCENTAGE_POINTS = None
        self.Q2_PERCENTAGE_POINTS = None
        self.Q3_PERCENTAGE_POINTS = None
        self.Q4_PERCENTAGE_POINTS = None
        
        # Q1 spread interpolator (for Q1 predictions)
        self.q1_spread_interpolator = None
        self.Q1_PERCENTAGE_SPREAD = None
        
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
        Calculate Q1 spread interpolator for proper Q1 modeling.
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
            
            # Calculate Q1 statistics (for Q1 spread interpolator)
            df['q1_total'] = df['q1_home_score'] + df['q1_away_score']
            df['q1_margin'] = df['q1_home_score'] - df['q1_away_score']
            df['abs_q1_margin'] = df['q1_margin'].abs()
            
            # Q1 percentage of game points
            df['q1_points_pct'] = df['q1_total'] / df['game_total'].replace(0, np.nan)
            q1_pct_points = df['q1_points_pct'].mean()
            
            # Q1 percentage of game spread (non-tie games only)
            non_tie_games = df[df['abs_game_margin'] > 0].copy()
            non_tie_games['q1_spread_pct'] = non_tie_games['abs_q1_margin'] / non_tie_games['abs_game_margin']
            q1_pct_spread = non_tie_games['q1_spread_pct'].mean()
            
            # Create smooth interpolator for Q1 spread percentage (same as Q1.py)
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
                df[f'q{q}_total'] = df[f'q{q}_home_score'] + df[f'q{q}_away_score']
                df[f'q{q}_points_pct'] = df[f'q{q}_total'] / df['game_total'].replace(0, np.nan)
                q_pct_points = df[f'q{q}_points_pct'].mean()
                setattr(self, f'Q{q}_PERCENTAGE_POINTS', q_pct_points)
                print(f"  Q{q} points as % of game total: {q_pct_points*100:.2f}%")
                print(f"  Mean Q{q} total: {df[f'q{q}_total'].mean():.2f} points")
            print()
            
            # Store typical game score (same as Q1.py)
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
            
            self.historical_data = df
            print("Data standardized to favored-underdog format for all quarters")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_empirical_distribution(self):
        """Calculate empirical distributions for all quarters."""
        if self.historical_data is None:
            return
        
        total_games = len(self.historical_data)
        
        for q in [1, 2, 3, 4]:
            # Home-away format
            raw_counts = self.historical_data[f'q{q}_raw_score_combo'].value_counts()
            raw_dist = {score: count / total_games for score, count in raw_counts.items()}
            setattr(self, f'empirical_distribution_q{q}', raw_dist)
            
            # Favored-underdog format
            std_counts = self.historical_data[f'q{q}_score_combo'].value_counts()
            std_dist = {score: count / total_games for score, count in std_counts.items()}
            setattr(self, f'standardized_empirical_dist_q{q}', std_dist)
            
            print(f"Q{q}: {len(std_dist)} unique favored-underdog scores")
    
    def get_score_pattern(self, score_str):
        """Classify score into pattern (same as Q1.py)."""
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
        """Find nearest anchor score by pattern and total points (same as Q1.py)."""
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
        """Get bounds for total and margin coefficients (same as Q1.py)."""
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
        """Fit model for specific quarter using Q1.py anchor + inheritance approach."""
        if self.historical_data is None:
            return
        
        print(f"\n{'='*80}")
        print(f"FITTING MODELS FOR QUARTER {quarter}")
        print(f"{'='*80}")
        
        # Get score counts for this quarter
        score_col = f'q{quarter}_score_combo'
        score_counts = self.historical_data[score_col].value_counts()
        
        # Define anchor threshold (same as Q1.py)
        anchor_threshold = 75
        anchor_scores = score_counts[score_counts >= anchor_threshold].index.tolist()
        rare_scores = score_counts[score_counts < anchor_threshold].index.tolist()
        
        # Store anchor scores
        setattr(self, f'anchor_scores_q{quarter}', anchor_scores)
        
        print(f"Anchor scores (>= {anchor_threshold} occurrences): {len(anchor_scores)}")
        print(f"Rare scores (< {anchor_threshold} occurrences): {len(rare_scores)}")
        
        # Prepare feature matrix (same as Q1.py)
        abs_spread = self.historical_data['abs_spread']
        total = self.historical_data['pregame_total']
        
        implied_fav_total = (total + abs_spread) / 2
        implied_dog_total = (total - abs_spread) / 2
        
        norm_fav = implied_fav_total / self.typical_game_score
        norm_dog = implied_dog_total / self.typical_game_score
        
        feature_total = norm_fav + norm_dog
        feature_margin = norm_fav - norm_dog
        
        X = np.column_stack([
            np.ones(len(self.historical_data)),
            feature_total,
            feature_margin
        ])
        
        print(f"\nFeature matrix: {X.shape[0]} games × {X.shape[1]} features")
        print(f"Features: [intercept, total, margin]")
        
        model_params = {}
        
        # PHASE 1: Fit anchor models
        print(f"\nPHASE 1: Fitting Anchor Models...")
        anchor_fits = 0
        
        for score_combo in anchor_scores:
            try:
                y = (self.historical_data[score_col] == score_combo).astype(int).values
                n_occurrences = np.sum(y)
                
                # Prior probability
                prior_prob = score_counts[score_combo] / len(self.historical_data)
                prior_prob_clipped = np.clip(prior_prob, 1e-6, 1 - 1e-6)
                prior_logit = np.log(prior_prob_clipped / (1 - prior_prob_clipped))
                
                # Get bounds
                total_bounds, margin_bounds = self.get_coefficient_bounds(score_combo, n_occurrences)
                
                # Define loss function with regularization
                def negative_log_likelihood(params):
                    logits = X @ params
                    logits_clipped = np.clip(logits, -50, 50)
                    probs = 1 / (1 + np.exp(-logits_clipped))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    
                    log_likelihood = np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                    nll = -log_likelihood
                    
                    # Light regularization for anchor models
                    alpha = 0.005 * (1 + 1/max(n_occurrences, 10))
                    l1_ratio = 0.3
                    
                    prior_means = np.array([prior_logit, 0, 0])
                    l1_penalty = alpha * l1_ratio * np.sum(np.abs(params - prior_means))
                    l2_penalty = alpha * (1 - l1_ratio) * np.sum((params - prior_means)**2)
                    
                    total_loss = nll + l1_penalty + l2_penalty
                    return total_loss if np.isfinite(total_loss) else 1e10
                
                # Initial parameters
                initial_params = np.array([prior_logit, 0, 0]) + np.random.normal(0, 0.01, 3)
                
                # Parameter bounds
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
        
        print(f"Successfully fit {anchor_fits}/{len(anchor_scores)} anchor models")
        
        # PHASE 2: Fit rare score models with coefficient inheritance
        print(f"\nPHASE 2: Fitting Rare Scores with Coefficient Inheritance...")
        rare_fits = 0
        
        for score_combo in rare_scores:
            try:
                y = (self.historical_data[score_col] == score_combo).astype(int).values
                n_occurrences = np.sum(y)
                
                # Skip if too few occurrences
                if n_occurrences < 5:
                    continue
                
                # Find nearest anchor
                nearest_anchor = self.find_nearest_anchor(score_combo, anchor_scores)
                anchor_params = model_params[nearest_anchor]
                
                # Prior probability
                prior_prob = score_counts[score_combo] / len(self.historical_data)
                prior_prob_clipped = np.clip(prior_prob, 1e-6, 1 - 1e-6)
                prior_logit = np.log(prior_prob_clipped / (1 - prior_prob_clipped))
                
                # Define loss with STRONG regularization toward anchor
                def negative_log_likelihood_inherited(params):
                    logits = X @ params
                    logits_clipped = np.clip(logits, -50, 50)
                    probs = 1 / (1 + np.exp(-logits_clipped))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    
                    log_likelihood = np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                    nll = -log_likelihood
                    
                    # STRONG regularization toward anchor coefficients
                    alpha_intercept = 0.05
                    alpha_coefs = 0.2 * (1 + 50/max(n_occurrences, 10))
                    l1_ratio = 0.3
                    
                    # Regularize intercept toward prior_logit
                    intercept_l1 = alpha_intercept * l1_ratio * abs(params[0] - prior_logit)
                    intercept_l2 = alpha_intercept * (1 - l1_ratio) * (params[0] - prior_logit)**2
                    
                    # Regularize total and margin coefficients toward anchor
                    coef_l1 = alpha_coefs * l1_ratio * (abs(params[1] - anchor_params[1]) + abs(params[2] - anchor_params[2]))
                    coef_l2 = alpha_coefs * (1 - l1_ratio) * ((params[1] - anchor_params[1])**2 + (params[2] - anchor_params[2])**2)
                    
                    total_loss = nll + intercept_l1 + intercept_l2 + coef_l1 + coef_l2
                    return total_loss if np.isfinite(total_loss) else 1e10
                
                # Initialize from anchor parameters but adjust intercept
                initial_params = anchor_params.copy()
                initial_params[0] = prior_logit
                
                # Bounds centered on anchor coefficients
                anchor_total_coef = anchor_params[1]
                anchor_margin_coef = anchor_params[2]
                
                # Allow ±0.5 movement from anchor
                total_bounds_tight = (anchor_total_coef - 0.5, anchor_total_coef + 0.5)
                margin_bounds_tight = (anchor_margin_coef - 0.5, anchor_margin_coef + 0.5)
                
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
        Uses same approach as Q1.py.
        """
        empirical_dist = getattr(self, f'standardized_empirical_dist_q{quarter}')
        model_params = getattr(self, f'model_params_q{quarter}')
        
        if not empirical_dist:
            return {}
        
        # Calculate features (same as Q1.py)
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
    
    def convolve_quarters(self, dist1, dist2):
        """
        Convolve two quarter distributions.
        Uses independence assumption: P(A and B) = P(A) * P(B)
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
    
    def calibrate_to_spread_and_total(self, full_game_dist, target_spread, target_total, max_iterations=50):
        """
        Calibrate full game distribution to match Vegas spread and total.
        Uses iterative adjustment of probabilities.
        
        Target: 50% favorite covers, 50% over
        Tolerance: 46.5% to 53.5%
        """
        abs_spread = abs(target_spread)
        
        # Calculate current probabilities
        def calculate_metrics(dist):
            prob_fav_cover = 0.0
            prob_over = 0.0
            
            for score, prob in dist.items():
                fav_score, dog_score = map(int, score.split('-'))
                margin = fav_score - dog_score
                game_total = fav_score + dog_score
                
                if margin > abs_spread:
                    prob_fav_cover += prob
                if game_total > target_total:
                    prob_over += prob
            
            return prob_fav_cover, prob_over
        
        current_dist = full_game_dist.copy()
        
        for iteration in range(max_iterations):
            prob_fav_cover, prob_over = calculate_metrics(current_dist)
            
            # Check if within tolerance
            if (0.465 <= prob_fav_cover <= 0.535) and (0.465 <= prob_over <= 0.535):
                print(f"Calibration converged after {iteration+1} iterations")
                print(f"  Favorite covers: {prob_fav_cover*100:.2f}%")
                print(f"  Over {target_total}: {prob_over*100:.2f}%")
                return current_dist
            
            # Adjust probabilities based on errors
            adjustment_factors = {}
            
            for score, prob in current_dist.items():
                fav_score, dog_score = map(int, score.split('-'))
                margin = fav_score - dog_score
                game_total = fav_score + dog_score
                
                # Start with factor of 1.0
                factor = 1.0
                
                # Adjust based on spread calibration
                if margin > abs_spread:
                    # Favorite covers
                    if prob_fav_cover > 0.535:
                        factor *= 0.98  # Decrease these outcomes
                    elif prob_fav_cover < 0.465:
                        factor *= 1.02  # Increase these outcomes
                else:
                    # Underdog covers
                    if prob_fav_cover < 0.465:
                        factor *= 0.98  # Decrease these outcomes
                    elif prob_fav_cover > 0.535:
                        factor *= 1.02  # Increase these outcomes
                
                # Adjust based on total calibration
                if game_total > target_total:
                    # Over
                    if prob_over > 0.535:
                        factor *= 0.98
                    elif prob_over < 0.465:
                        factor *= 1.02
                else:
                    # Under
                    if prob_over < 0.465:
                        factor *= 0.98
                    elif prob_over > 0.535:
                        factor *= 1.02
                
                adjustment_factors[score] = factor
            
            # Apply adjustments
            adjusted_dist = {}
            for score, prob in current_dist.items():
                adjusted_dist[score] = prob * adjustment_factors[score]
            
            # Renormalize
            total_prob = sum(adjusted_dist.values())
            current_dist = {score: prob / total_prob for score, prob in adjusted_dist.items()}
        
        # Max iterations reached
        prob_fav_cover, prob_over = calculate_metrics(current_dist)
        print(f"Calibration reached max iterations ({max_iterations})")
        print(f"  Favorite covers: {prob_fav_cover*100:.2f}%")
        print(f"  Over {target_total}: {prob_over*100:.2f}%")
        
        return current_dist
    
    def predict(self, pregame_spread, total, debug=False):
        """
        Generate predictions for all quarters and combine with calibration.
        
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
            # Generate distributions for all 4 quarters (favored-underdog format)
            dist_q1 = self.predict_standardized_probability(pregame_spread, total, 1)
            dist_q2 = self.predict_standardized_probability(pregame_spread, total, 2)
            dist_q3 = self.predict_standardized_probability(pregame_spread, total, 3)
            dist_q4 = self.predict_standardized_probability(pregame_spread, total, 4)
            
            # Convolve to get first half and second half
            dist_h1 = self.convolve_quarters(dist_q1, dist_q2)
            dist_h2 = self.convolve_quarters(dist_q3, dist_q4)
            
            # Convolve to get full game
            dist_full_game_raw = self.convolve_quarters(dist_h1, dist_h2)
            
            # Calibrate to match Vegas spread and total
            dist_full_game = self.calibrate_to_spread_and_total(
                dist_full_game_raw, pregame_spread, total
            )
            
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
    print("CFB ALL QUARTERS PREDICTOR - Q1 APPROACH FOR ALL QUARTERS")
    print("="*80)
    print("Features:")
    print("  ✓ Anchor scores + coefficient inheritance for each quarter")
    print("  ✓ Orthogonal features: feature_total and feature_margin")
    print("  ✓ Convolution for combining quarters")
    print("  ✓ Calibration to match Vegas spread/total")
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
                sorted_scores = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\nQuarter {q} - Top 5:")
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
            print(f"\nFull Game (Calibrated) - Top 10:")
            for score, prob in sorted_fg:
                print(f"  {score:<10} {prob*100:>6.2f}%")
            
        except ValueError:
            print("Enter valid numbers")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()