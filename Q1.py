import os
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

class CFBQuarterScorePredictor:
    """
    Predicts Q1 score probabilities using coefficient inheritance.
    
    Key improvements:
    - Anchor models for common scores (n >= 100)
    - Coefficient inheritance for rare scores from nearest anchor
    - Strong regularization toward anchor coefficients
    - No hardcoded buckets - every score gets individual treatment
    """
    
    def __init__(self, db_config):
        """Initialize with database config and empty containers for data and models."""
        self.db_config = db_config
        self.empirical_distribution = {}
        self.standardized_empirical_dist = {}
        self.model_params = {}
        self.anchor_scores = []
        self.inheritance_log = []
        self.historical_data = None
        self.Q1_PERCENTAGE_POINTS = None
        self.Q1_PERCENTAGE_SPREAD = None
        self.q1_spread_interpolator = None
        
    def connect_to_database(self):
        """Connect to MySQL database."""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            return None
    
    def load_historical_data(self):
        """Load Q1 scores from database and standardize to favored-underdog orientation."""
        connection = self.connect_to_database()
        if not connection:
            return False
            
        try:
            query = """
            SELECT 
                g.game_id,
                g.pregame_spread,
                g.pregame_total,
                q1_home.home_score as q1_home_score,
                q1_home.away_score as q1_away_score,
                g.home_points,
                g.away_points
            FROM cfb.games g
            JOIN cfb.quarter_scoring q1_home ON g.game_id = q1_home.game_id 
            WHERE q1_home.quarter = 1
            AND g.pregame_spread IS NOT NULL
            AND g.pregame_total IS NOT NULL
            AND g.pregame_spread BETWEEN -70 AND 70
            AND g.pregame_total BETWEEN 20 AND 90
            AND (g.home_classification = 'fbs' OR g.away_classification = 'fbs')
            AND g.home_points IS NOT NULL
            AND g.away_points IS NOT NULL
            ORDER BY g.game_id
            """
            
            df = pd.read_sql(query, connection)
            print(f"Loaded {len(df)} historical games")
            
            # Calculate Q1 statistics
            df['q1_total'] = df['q1_home_score'] + df['q1_away_score']
            df['game_total'] = df['home_points'] + df['away_points']
            df['q1_margin'] = df['q1_home_score'] - df['q1_away_score']
            df['game_margin'] = df['home_points'] - df['away_points']
            df['abs_q1_margin'] = df['q1_margin'].abs()
            df['abs_game_margin'] = df['game_margin'].abs()
            df['abs_spread'] = df['pregame_spread'].abs()
            
            # Calculate Q1 percentage of game points
            df['q1_points_pct'] = df['q1_total'] / df['game_total'].replace(0, np.nan)
            q1_pct_points = df['q1_points_pct'].mean()
            
            # Calculate Q1 percentage of game spread (non-tie games only)
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
            
            print(f"\nEmpirical Q1 Statistics:")
            print(f"  Q1 points as % of game total: {q1_pct_points*100:.2f}%")
            print(f"  Q1 spread as % of game margin: {q1_pct_spread*100:.2f}% (overall)")
            print(f"  Mean Q1 total: {df['q1_total'].mean():.2f} points")
            print(f"  Mean game total: {df['game_total'].mean():.2f} points")
            
            # Standardize scores to favored-underdog orientation
            df['raw_score_combination'] = (
                df['q1_home_score'].astype(str) + '-' + 
                df['q1_away_score'].astype(str)
            )
            
            df['favored_score'] = np.where(
                df['pregame_spread'] < 0,
                df['q1_home_score'],
                df['q1_away_score']
            )
            
            df['underdog_score'] = np.where(
                df['pregame_spread'] < 0,
                df['q1_away_score'],
                df['q1_home_score']
            )
            
            df['score_combination'] = (
                df['favored_score'].astype(str) + '-' + 
                df['underdog_score'].astype(str)
            )
            
            self.historical_data = df
            print("Games standardized to favored-underdog orientation for modeling")
            return True
            
        except Exception as e:
            print(f"Data loading error: {e}")
            return False
        finally:
            connection.close()
    
    def calculate_empirical_distribution(self):
        """Calculate baseline probabilities for both home-away and standardized formats."""
        if self.historical_data is None:
            return
            
        total_games = len(self.historical_data)
        
        # Home-away format
        score_counts = self.historical_data['raw_score_combination'].value_counts()
        self.empirical_distribution = {}
        for score_combo, count in score_counts.items():
            self.empirical_distribution[score_combo] = count / total_games
        
        # Favored-underdog format
        standardized_counts = self.historical_data['score_combination'].value_counts()
        self.standardized_empirical_dist = {}
        for score_combo, count in standardized_counts.items():
            self.standardized_empirical_dist[score_combo] = count / total_games
        
        print(f"\nEmpirical Distribution - Top 40 Scores (Favored-Underdog format):")
        print(f"{'Score':<10} {'Count':<8} {'Probability':<12} {'Percentage'}")
        print("-" * 60)
        sorted_std = sorted(standardized_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (score, count) in enumerate(sorted_std[:40]):
            prob = count / total_games
            print(f"{score:<10} {count:<8} {prob:<12.6f} {prob*100:.2f}%")

    def get_score_pattern(self, score_str):
        """Categorize score into pattern for finding similar scores."""
        fav_score, dog_score = map(int, score_str.split('-'))
        margin = fav_score - dog_score
        score_total = fav_score + dog_score
        
        # Define pattern categories
        if fav_score == dog_score:
            return 'tie', score_total
        elif dog_score == 0:
            return 'fav_shutout', score_total
        elif fav_score == 0:
            return 'dog_shutout', score_total
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
        
        # If no same-pattern anchor, find closest by total regardless of pattern
        # but prefer similar score characteristics
        all_anchors_with_totals = []
        for anchor in anchor_scores:
            anchor_pattern, anchor_total = self.get_score_pattern(anchor)
            all_anchors_with_totals.append((anchor, anchor_total))
        
        all_anchors_with_totals.sort(key=lambda x: abs(x[1] - target_total))
        return all_anchors_with_totals[0][0]

    def get_coefficient_bounds(self, score_str, n_occurrences):
        """
        Get bounds for total and margin coefficients based on score characteristics.
        Uses flexible bounds that allow model to learn from data.
        """
        fav_score, dog_score = map(int, score_str.split('-'))
        margin = fav_score - dog_score
        score_total = fav_score + dog_score
        
        # For common scores (n >= 100), use wide bounds
        # For rare scores, use tighter bounds to stay close to anchor
        bound_width = 3 if n_occurrences >= 100 else 1.5
        
        if fav_score == dog_score:
            # Ties: total coef depends on scoring level
            if score_total == 0:
                # 0-0 decreases with total
                total_bounds = (-bound_width, 0)
            elif score_total <= 14:
                # Low-mid ties: flexible
                total_bounds = (-bound_width, bound_width)
            else:
                # High-scoring ties: increase with total
                total_bounds = (0, bound_width)
            margin_bounds = (-1, 1)  # Ties happen at any spread
            
        elif dog_score == 0:
            # Favorite shutouts
            if fav_score <= 10:
                # Low-scoring shutouts: defensive games
                total_bounds = (-bound_width, 1)
            elif fav_score <= 17:
                # Mid shutouts: flexible
                total_bounds = (-bound_width, bound_width)
            else:
                # High shutouts: offensive blowouts
                total_bounds = (0, bound_width)
            margin_bounds = (0, bound_width)
            
        elif fav_score == 0:
            # Underdog shutouts (mirror of favorite shutouts)
            if dog_score <= 10:
                total_bounds = (-bound_width, 1)
            elif dog_score <= 17:
                total_bounds = (-bound_width, bound_width)
            else:
                total_bounds = (0, bound_width)
            margin_bounds = (-bound_width, 0)
            
        elif margin >= 14:
            # Large margins: more common in high-scoring games
            total_bounds = (0, bound_width)
            margin_bounds = (0 if margin > 0 else -bound_width, bound_width if margin > 0 else 0)
            
        elif score_total >= 21:
            # High-scoring games: increase with total
            total_bounds = (0, bound_width)
            margin_bounds = (-bound_width, bound_width)
            
        else:
            # Default: flexible bounds
            total_bounds = (-bound_width, bound_width)
            margin_bounds = (-bound_width, bound_width)
        
        return total_bounds, margin_bounds

    def fit_model(self):
        """Fit models using coefficient inheritance from anchor scores."""
        if self.historical_data is None:
            return
        
        print(f"\n{'='*80}")
        print(f"FITTING MODELS WITH COEFFICIENT INHERITANCE")
        print(f"{'='*80}")
        
        # Get score counts
        score_counts = self.historical_data['score_combination'].value_counts()
        

        # Scores below this will inherit from these reliable anchors
        anchor_threshold = 75
        self.anchor_scores = score_counts[score_counts >= anchor_threshold].index.tolist()
        rare_scores = score_counts[score_counts < anchor_threshold].index.tolist()
        
        print(f"\nAnchor scores (>= {anchor_threshold} occurrences): {len(self.anchor_scores)}")
        print(f"Rare scores (< {anchor_threshold} occurrences): {len(rare_scores)}")
        
        # Prepare feature matrix
        abs_spread = self.historical_data['abs_spread']
        total = self.historical_data['pregame_total']
        
        implied_fav_total = (total + abs_spread) / 2
        implied_dog_total = (total - abs_spread) / 2
        
        typical_game_score = self.historical_data['game_total'].mean() / 2
        norm_fav = implied_fav_total / typical_game_score
        norm_dog = implied_dog_total / typical_game_score
        
        feature_total = norm_fav + norm_dog
        feature_margin = norm_fav - norm_dog
        
        X = np.column_stack([
            np.ones(len(self.historical_data)),
            feature_total,
            feature_margin
        ])
        
        print(f"\nFeature matrix: {X.shape[0]} games × {X.shape[1]} features")
        print(f"Features: [intercept, total, margin]")
        
        # PHASE 1: Fit anchor models
        print(f"\n{'='*80}")
        print(f"PHASE 1: Fitting Anchor Models")
        print(f"{'='*80}")
        
        anchor_fits = 0
        for score_combo in self.anchor_scores:
            try:
                fav_score, dog_score = map(int, score_combo.split('-'))
                y = (self.historical_data['score_combination'] == score_combo).astype(int).values
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
                    self.model_params[score_combo] = result.x
                    anchor_fits += 1
                    
            except Exception as e:
                print(f"Error fitting anchor {score_combo}: {e}")
                continue
        
        print(f"Successfully fit {anchor_fits}/{len(self.anchor_scores)} anchor models")
        
        # PHASE 2: Fit rare score models with coefficient inheritance
        print(f"\n{'='*80}")
        print(f"PHASE 2: Fitting Rare Scores with Coefficient Inheritance")
        print(f"{'='*80}")
        
        rare_fits = 0
        inheritance_log = []
        
        for score_combo in rare_scores:
            try:
                y = (self.historical_data['score_combination'] == score_combo).astype(int).values
                n_occurrences = np.sum(y)
                
                # Skip if too few occurrences
                if n_occurrences < 5:
                    continue
                
                # Find nearest anchor
                nearest_anchor = self.find_nearest_anchor(score_combo, self.anchor_scores)
                anchor_params = self.model_params[nearest_anchor]
                
                inheritance_log.append((score_combo, nearest_anchor, n_occurrences))
                
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
                    # Intercept can vary (different base rates), but total/margin stay close
                    alpha_intercept = 0.05
                    alpha_coefs = 0.2 * (1 + 50/max(n_occurrences, 10))  # Stronger for rarer scores
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
                initial_params[0] = prior_logit  # Use score's own base rate
                
                # Bounds centered ONLY on anchor coefficients
                # This is the fix: no conflicting logic from get_coefficient_bounds
                anchor_total_coef = anchor_params[1]
                anchor_margin_coef = anchor_params[2]
                
                # Allow ±0.5 movement from anchor for rare scores
                total_bounds_tight = (anchor_total_coef - 0.5, anchor_total_coef + 0.5)
                margin_bounds_tight = (anchor_margin_coef - 0.5, anchor_margin_coef + 0.5)
                
                param_bounds = [
                    (prior_logit - 5, prior_logit + 5),  # Intercept is free
                    total_bounds_tight,                   # Total anchored
                    margin_bounds_tight                   # Margin anchored
                ]
                
                # Optimize
                result = minimize(negative_log_likelihood_inherited, initial_params,
                                method='L-BFGS-B', bounds=param_bounds)
                
                if result.success:
                    self.model_params[score_combo] = result.x
                    rare_fits += 1
                    
            except Exception as e:
                print(f"Error fitting rare score {score_combo}: {e}")
                continue
        
        print(f"Successfully fit {rare_fits}/{len(rare_scores)} rare score models")
        
        # Show some inheritance examples
        print(f"\nCoefficient Inheritance Examples:")
        print(f"{'Rare Score':<12} {'Anchor':<12} {'Occurrences':<12} {'Rare Total Coef':<18} {'Anchor Total Coef'}")
        print("-" * 80)
        for rare, anchor, n_occ in inheritance_log[:10]:
            if rare in self.model_params and anchor in self.model_params:
                rare_coef = self.model_params[rare][1]
                anchor_coef = self.model_params[anchor][1]
                print(f"{rare:<12} {anchor:<12} {n_occ:<12} {rare_coef:>+.4f} ({rare_coef:+.2f})  {anchor_coef:>+.4f} ({anchor_coef:+.2f})")
        
        # Store inheritance log for debugging
        self.inheritance_log = inheritance_log
        
        print(f"{'='*80}\n")

    def predict(self, pregame_spread, total, debug=False):
        """Generate probability distribution for Q1 scores."""
        home_favored = (pregame_spread < 0)
        abs_spread = abs(pregame_spread)
        
        debug_lines = []
        
        if debug:
            debug_lines.append("="*80)
            debug_lines.append("DETAILED DEBUG OUTPUT")
            debug_lines.append("="*80)
            debug_lines.append(f"Pregame spread: {pregame_spread:+.1f}")
            debug_lines.append(f"Favorite: {'Home' if home_favored else 'Away'}")
            debug_lines.append(f"Spread magnitude: {abs_spread:.1f}")
            debug_lines.append(f"Total: {total:.1f}")
            debug_lines.append("="*80)
            debug_lines.append("")
        
        print(f"\n{'='*80}")
        print(f"PREDICTION FOR QUARTER 1")
        print(f"{'='*80}")
        print(f"Spread: {pregame_spread:+.1f} ({'Home' if home_favored else 'Away'} favored by {abs_spread:.1f})")
        print(f"Total: {total:.1f}")
        print(f"{'='*80}\n")
        
        predictions = {}
        debug_info = {}
        
        # Calculate features
        implied_fav_total = (total + abs_spread) / 2
        implied_dog_total = (total - abs_spread) / 2
        
        typical_game_score = self.historical_data['game_total'].mean() / 2
        norm_fav = implied_fav_total / typical_game_score
        norm_dog = implied_dog_total / typical_game_score
        
        feature_total = norm_fav + norm_dog
        feature_margin = norm_fav - norm_dog
        
        features_full = np.array([1.0, feature_total, feature_margin])
        
        print(f"Game-level implied totals:")
        print(f"  Favorite: {implied_fav_total:.2f} points (full game)")
        print(f"  Underdog: {implied_dog_total:.2f} points (full game)")
        print(f"  Model learned Q1 is ~{self.Q1_PERCENTAGE_POINTS*100:.1f}% of these")
        print(f"\nOrthogonal features:")
        print(f"  feature_total  = {feature_total:.4f}")
        print(f"  feature_margin = {feature_margin:.4f}\n")
        
        if debug:
            expected_q1_fav = implied_fav_total * self.Q1_PERCENTAGE_POINTS
            expected_q1_dog = implied_dog_total * self.Q1_PERCENTAGE_POINTS
            expected_q1_total = expected_q1_fav + expected_q1_dog
            
            debug_lines.append(f"GAME-LEVEL IMPLIED TOTALS:")
            debug_lines.append(f"  Favorite: {implied_fav_total:.2f} points (full game)")
            debug_lines.append(f"  Underdog: {implied_dog_total:.2f} points (full game)")
            debug_lines.append(f"  Expected Q1: Fav {expected_q1_fav:.2f}, Dog {expected_q1_dog:.2f}")
            debug_lines.append(f"  Total expected Q1: {expected_q1_total:.2f} points")
            debug_lines.append("")
            debug_lines.append("ORTHOGONAL FEATURES:")
            debug_lines.append(f"  feature_total  = {feature_total:.4f}")
            debug_lines.append(f"  feature_margin = {feature_margin:.4f}")
            debug_lines.append("")
        
        # Generate predictions for all scores
        all_scores = list(self.empirical_distribution.keys())
        
        for home_away_score in all_scores:
            try:
                home_score, away_score = map(int, home_away_score.split('-'))
                base_prob = self.empirical_distribution[home_away_score]
                
                # Convert to favored-underdog orientation
                if home_favored:
                    favored_score = home_score
                    underdog_score = away_score
                else:
                    favored_score = away_score
                    underdog_score = home_score
                
                standardized_score = f"{favored_score}-{underdog_score}"
                
                # Use model if available, otherwise use empirical
                if standardized_score in self.model_params:
                    params = self.model_params[standardized_score]
                    
                    logit = features_full @ params
                    logit_clipped = np.clip(logit, -50, 50)
                    model_prob = 1 / (1 + np.exp(-logit_clipped))
                    
                    predicted_prob = model_prob
                    
                    if debug:
                        debug_info[home_away_score] = {
                            'empirical': base_prob,
                            'standardized_score': standardized_score,
                            'model_params': params,
                            'logit': logit,
                            'model_prob': model_prob,
                            'final_prob': predicted_prob,
                            'method': 'model'
                        }
                else:
                    predicted_prob = base_prob
                    
                    if debug:
                        debug_info[home_away_score] = {
                            'empirical': base_prob,
                            'standardized_score': standardized_score,
                            'final_prob': predicted_prob,
                            'method': 'empirical_only'
                        }
                
                predictions[home_away_score] = predicted_prob
                
            except Exception as e:
                predictions[home_away_score] = self.empirical_distribution.get(home_away_score, 0)
                continue
        
        # Normalize
        total_prob_before_norm = sum(predictions.values())
        if total_prob_before_norm > 0:
            predictions = {score: prob / total_prob_before_norm for score, prob in predictions.items()}
        
        print(f"Total probability before normalization: {total_prob_before_norm:.6f}\n")
        
        # Debug output
        if debug:
            debug_lines.append("="*80)
            debug_lines.append("MODEL DIAGNOSTICS")
            debug_lines.append("="*80)
            debug_lines.append("")
            
            # Probability mass by bins
            debug_lines.append("PROBABILITY MASS BY SCORE TOTAL:")
            score_bins = {
                '0-6 pts': (0, 6),
                '7-13 pts': (7, 13),
                '14-20 pts': (14, 20),
                '21-27 pts': (21, 27),
                '28+ pts': (28, 999)
            }
            
            for bin_name, (min_pts, max_pts) in score_bins.items():
                bin_prob = 0
                bin_scores = []
                for score, info in debug_info.items():
                    h, a = map(int, score.split('-'))
                    total_pts = h + a
                    if min_pts <= total_pts <= max_pts:
                        bin_prob += info['final_prob']
                        if info['final_prob'] >= 0.01:
                            bin_scores.append((score, info['final_prob']))
                
                debug_lines.append(f"  {bin_name:<12}: {bin_prob:.4f} ({bin_prob*100:.1f}%)")
                if bin_scores and bin_name in ['14-20 pts', '21-27 pts', '28+ pts']:
                    bin_scores.sort(key=lambda x: x[1], reverse=True)
                    top_contributors = ', '.join([f"{s}:{p*100:.1f}%" for s, p in bin_scores[:3]])
                    debug_lines.append(f"               Top: {top_contributors}")
            
            debug_lines.append("")
            debug_lines.append(f"Total probability before normalization: {total_prob_before_norm:.6f}")
            debug_lines.append("")
            
            # Detailed analysis of specific problematic scores
            debug_lines.append("="*80)
            debug_lines.append("DETAILED SCORE ANALYSIS - PROBLEMATIC CASES")
            debug_lines.append("="*80)
            debug_lines.append("")
            
            problematic_scores = ['0-0', '7-7', '14-14', '10-0', '10-3', '10-6', '6-3', '0-10']
            
            for score_to_check in problematic_scores:
                # Find the score in either home-away or away-home format
                found_score = None
                for home_away_score in all_scores:
                    home_score, away_score = map(int, home_away_score.split('-'))
                    
                    if home_favored:
                        standardized = f"{home_score}-{away_score}"
                    else:
                        standardized = f"{away_score}-{home_score}"
                    
                    if standardized == score_to_check or home_away_score == score_to_check:
                        found_score = home_away_score
                        break
                
                if not found_score or found_score not in debug_info:
                    continue
                
                info = debug_info[found_score]
                fav, dog = map(int, info['standardized_score'].split('-'))
                score_total = fav + dog
                score_margin = abs(fav - dog)
                
                debug_lines.append(f"Score: {info['standardized_score']} (Displayed as {found_score})")
                debug_lines.append("-"*80)
                debug_lines.append(f"  Score total: {score_total} points")
                debug_lines.append(f"  Score margin: {score_margin} points")
                debug_lines.append(f"  Expected Q1 total: {expected_q1_total:.2f} points")
                debug_lines.append(f"  Ratio (score/expected): {score_total/expected_q1_total:.2f}x")
                debug_lines.append(f"  Empirical probability: {info['empirical']:.6f} ({info['empirical']*100:.2f}%)")
                debug_lines.append(f"  Model probability: {info['final_prob']:.6f} ({info['final_prob']*100:.2f}%)")
                debug_lines.append(f"  Change: {(info['final_prob'] - info['empirical'])*100:+.2f} percentage points")
                debug_lines.append("")
                
                if info['method'] == 'model':
                    params = info['model_params']
                    pattern, pattern_total = self.get_score_pattern(info['standardized_score'])
                    
                    debug_lines.append(f"  Pattern: {pattern}")
                    debug_lines.append(f"  Model coefficients:")
                    debug_lines.append(f"    Intercept: {params[0]:>+8.4f}")
                    debug_lines.append(f"    Total:     {params[1]:>+8.4f}")
                    debug_lines.append(f"    Margin:    {params[2]:>+8.4f}")
                    debug_lines.append("")
                    
                    # Check if this is an anchor or inherited
                    is_anchor = info['standardized_score'] in self.anchor_scores
                    
                    if is_anchor:
                        debug_lines.append(f"  Model Type: ANCHOR (common score with n >= 100)")
                        score_count = self.historical_data['score_combination'].value_counts().get(info['standardized_score'], 0)
                        debug_lines.append(f"  Occurrences: {score_count}")
                    else:
                        debug_lines.append(f"  Model Type: INHERITED (rare score)")
                        
                        # Find inheritance info
                        anchor_used = None
                        n_occ = 0
                        for rare, anchor, n in self.inheritance_log:
                            if rare == info['standardized_score']:
                                anchor_used = anchor
                                n_occ = n
                                break
                        
                        if anchor_used:
                            debug_lines.append(f"  Occurrences: {n_occ}")
                            debug_lines.append(f"  Inherited from: {anchor_used}")
                            
                            if anchor_used in self.model_params:
                                anchor_params = self.model_params[anchor_used]
                                anchor_pattern, anchor_pattern_total = self.get_score_pattern(anchor_used)
                                
                                debug_lines.append(f"  Anchor pattern: {anchor_pattern} ({anchor_pattern_total} pts)")
                                debug_lines.append(f"  Anchor coefficients:")
                                debug_lines.append(f"    Intercept: {anchor_params[0]:>+8.4f}")
                                debug_lines.append(f"    Total:     {anchor_params[1]:>+8.4f}  (score uses: {params[1]:>+8.4f}, diff: {params[1]-anchor_params[1]:>+.4f})")
                                debug_lines.append(f"    Margin:    {anchor_params[2]:>+8.4f}  (score uses: {params[2]:>+8.4f}, diff: {params[2]-anchor_params[2]:>+.4f})")
                                debug_lines.append("")
                                debug_lines.append(f"  INHERITANCE ANALYSIS:")
                                
                                # Analyze if inheritance makes sense
                                total_deviation = abs(params[1] - anchor_params[1])
                                margin_deviation = abs(params[2] - anchor_params[2])
                                
                                debug_lines.append(f"    Total coef deviation from anchor: {total_deviation:.4f}")
                                debug_lines.append(f"    Margin coef deviation from anchor: {margin_deviation:.4f}")
                                debug_lines.append(f"    Max allowed deviation: 0.5000 (set by bounds)")
                                
                                if total_deviation > 0.4:
                                    debug_lines.append(f"    WARNING: Large deviation in total coefficient")
                                
                                # Context analysis
                                if score_total > expected_q1_total * 2:
                                    debug_lines.append(f"    CONTEXT WARNING: Score total is {score_total/expected_q1_total:.2f}x expected")
                                    debug_lines.append(f"    This score is very unlikely in this game total")
                                    if params[1] > 0:
                                        debug_lines.append(f"    PROBLEM: Positive total coef ({params[1]:+.4f}) increases probability")
                                        debug_lines.append(f"    SOLUTION NEEDED: Context-dependent scaling for extreme scores")
                    
                    debug_lines.append("")
                    debug_lines.append(f"  Logit calculation for this game:")
                    debug_lines.append(f"    = {params[0]:.4f} + {params[1]:.4f}×{feature_total:.4f} + {params[2]:.4f}×{feature_margin:.4f}")
                    debug_lines.append(f"    = {params[0]:.4f} + {params[1]*feature_total:.4f} + {params[2]*feature_margin:.4f}")
                    debug_lines.append(f"    = {info['logit']:.4f}")
                    debug_lines.append(f"    → Probability = 1/(1+e^(-{info['logit']:.4f})) = {info['model_prob']:.6f}")
                    
                    # Show what happens at different totals
                    debug_lines.append("")
                    debug_lines.append(f"  SENSITIVITY ANALYSIS - How probability changes with total:")
                    test_totals = [40, 45, 50, 55, 60, 65]
                    for test_total in test_totals:
                        test_implied_fav = (test_total + abs_spread) / 2
                        test_implied_dog = (test_total - abs_spread) / 2
                        test_norm_fav = test_implied_fav / typical_game_score
                        test_norm_dog = test_implied_dog / typical_game_score
                        test_feature_total = test_norm_fav + test_norm_dog
                        
                        test_logit = params[0] + params[1] * test_feature_total + params[2] * feature_margin
                        test_prob = 1 / (1 + np.exp(-np.clip(test_logit, -50, 50)))
                        
                        debug_lines.append(f"    Total {test_total}: {test_prob:.6f} ({test_prob*100:.2f}%)")
                    
                else:
                    debug_lines.append(f"  Model Type: EMPIRICAL ONLY (no model fitted)")
                
                debug_lines.append("")
                debug_lines.append("")
            
            debug_lines.append("="*80)
            
            with open('debug_output.txt', 'w') as f:
                f.write('\n'.join(debug_lines))
            
            print("Debug output saved to: debug_output.txt\n")
        
        # Display results
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        filtered_predictions = [(score, prob) for score, prob in sorted_predictions if prob >= 0.0005]
        
        print(f"Scores with Probability > 0.05%:")
        print(f"{'Rank':<6} {'Score':<10} {'Probability':<12} {'Percentage'}")
        print("-" * 70)
        
        cumulative_prob = 0
        for rank, (score, prob) in enumerate(filtered_predictions, 1):
            cumulative_prob += prob
            print(f"{rank:<6} {score:<10} {prob:<12.6f} {prob*100:>6.2f}%")
        
        print(f"\nDisplayed {len(filtered_predictions)} scores")
        print(f"Cumulative probability: {cumulative_prob*100:.2f}%")

        print(f"Probability of draw and Over 12.5 points: {predictions.get('7-7', 0) + predictions.get('10-10', 0) + predictions.get('14-14', 0):.4f}")
        
        return {
            'home_away_format': predictions,
            'abs_spread': abs_spread,
            'home_favored': home_favored,
            'debug_info': debug_info if debug else None
        }

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
    
    predictor = CFBQuarterScorePredictor(db_config)
    
    print("="*80)
    print("CFB QUARTER 1 SCORE PREDICTOR - COEFFICIENT INHERITANCE")
    print("="*80)
    print("Improvements:")
    print("  ✓ Anchor models for common scores (n >= 100)")
    print("  ✓ Coefficient inheritance for rare scores")
    print("  ✓ Strong regularization toward anchor coefficients")
    print("  ✓ No hardcoded buckets - intelligent borrowing")
    print("="*80)
    
    print("\nLoading historical data...")
    if not predictor.load_historical_data():
        print("Failed to load data")
        return
    
    print("\nCalculating empirical distribution...")
    predictor.calculate_empirical_distribution()
    
    print("\nFitting models...")
    predictor.fit_model()

    print("\n" + "="*80)
    print("MODEL READY")
    print("="*80)
    print("Enter signed spread and total for predictions")
    print("Format: spread total (e.g., -3.5 58.5)")
    print("Add 'debug' for detailed output")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            print("\n(Ex: Home favored by 3.5 with total 50.5 -> enter: -3.5 50.5)")
            print("(Ex: Away favored by 6.5 with total 54.5 -> enter: 6.5 54.5)")
            user_input = input("Enter spread and total: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            debug_mode = 'debug' in user_input.lower()
            if debug_mode:
                user_input = user_input.lower().replace('debug', '').strip()
                
            parts = user_input.split()
            if len(parts) != 2:
                print("Enter two numbers: spread and total")
                continue
                
            pregame_spread = float(parts[0])
            total = float(parts[1])
            
            if not (-50 <= pregame_spread <= 50) or not (30 <= total <= 90):
                print("Use realistic values: spread -50 to 50, total 30 to 90")
                continue
                
            predictor.predict(pregame_spread, total, debug=debug_mode)
            
        except ValueError:
            print("Enter valid numbers")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()