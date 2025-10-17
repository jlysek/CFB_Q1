import os
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class CFBQuarterScorePredictor:
    """
    Main class for predicting college football first quarter scores.
    Uses historical game data to build probabilistic models that can predict
    the likelihood of different score combinations based on pregame betting lines.
    """
    
    def __init__(self, db_config):
        """
        Initialize the predictor with database configuration.
        Sets up empty containers for storing empirical distributions, model parameters, and historical data.
        """
        self.db_config = db_config
        self.empirical_distribution = {}  # Will store baseline probability of each score combo
        self.model_params = {}  # Will store logistic regression coefficients for each score
        self.historical_data = None  # Will store all historical games data
        
    def connect_to_database(self):
        """
        Establish connection to MySQL database using the provided configuration.
        Returns connection object if successful, None if there is an error.
        """
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            return None
    
    def load_historical_data(self):
        """
        Load historical first quarter scoring data from the database.
        Joins the games table with quarter_scoring table to get Q1 scores alongside pregame betting lines.
        Only includes games with valid pregame spread and total data.
        Creates a 'score_combination' field like '7-3' for easy grouping.
        """
        connection = self.connect_to_database()
        if not connection:
            return False
            
        try:
            # Query pulls game_id, pregame betting lines, and Q1 scores for both teams
            # Filters for reasonable spread and total values to exclude outliers
            query = """
            SELECT 
                g.game_id,
                g.pregame_spread,
                g.pregame_total,
                q1_home.home_score as q1_home_score,
                q1_home.away_score as q1_away_score
            FROM cfb.games g
            JOIN cfb.quarter_scoring q1_home ON g.game_id = q1_home.game_id 
            WHERE q1_home.quarter = 1
            AND g.pregame_spread IS NOT NULL
            AND g.pregame_total IS NOT NULL
            AND g.pregame_spread BETWEEN -70 AND 70
            AND g.pregame_total BETWEEN 20 AND 90
            AND (g.home_classification = 'fbs' OR g.away_classification = 'fbs')
            ORDER BY g.game_id
            """
            
            # Load query results into pandas DataFrame
            self.historical_data = pd.read_sql(query, connection)
            
            # Create combined score string for easier grouping and probability calculation
            # Example: home=7, away=3 becomes '7-3'
            self.historical_data['score_combination'] = (
                self.historical_data['q1_home_score'].astype(str) + '-' + 
                self.historical_data['q1_away_score'].astype(str)
            )
            
            print(f"Loaded {len(self.historical_data)} historical games")
            
            return True
            
        except Exception as e:
            print(f"Data loading error: {e}")
            return False
        finally:
            connection.close()
    
    def calculate_empirical_distribution(self):
        """
        Calculate baseline probability distribution of all score combinations.
        This represents the unconditional probability of each score without considering betting lines.
        Essentially answers: "What percentage of all Q1s end 7-7, 7-3, etc?"
        """
        if self.historical_data is None:
            return
            
        # Count how many times each score combination occurred
        score_counts = self.historical_data['score_combination'].value_counts()
        total_games = len(self.historical_data)
        
        # Convert counts to probabilities by dividing by total games
        self.empirical_distribution = {}
        for score_combo, count in score_counts.items():
            self.empirical_distribution[score_combo] = count / total_games
        
        # Display the most common scores for verification
        print(f"\nEmpirical Distribution - Top 20 Scores:")
        print(f"{'Score':<10} {'Count':<8} {'Probability':<12} {'Percentage'}")
        print("-" * 60)
        sorted_dist = sorted(self.empirical_distribution.items(), key=lambda x: x[1], reverse=True)
        for i, (score, prob) in enumerate(sorted_dist[:20]):
            count = int(prob * total_games)
            print(f"{score:<10} {count:<8} {prob:<12.6f} {prob*100:.2f}%")
    def load_calibration_data(self, calibration_folder):
        """
        Load market calibration data from CSV files in the specified folder.
        Each file should be named <spread>_<total>.csv with columns: Home Score, Away Score, Fair
        
        This data represents the market's probability distribution for specific spread/total combinations,
        which we'll use to calibrate our model's predictions.
        
        Args:
            calibration_folder: path to folder containing calibration CSV files
        """
        import glob
        import re
        
        self.calibration_data = {}
        
        # Find all CSV files in the calibration folder
        csv_files = glob.glob(os.path.join(calibration_folder, '*.csv'))
        
        if not csv_files:
            print(f"Warning: No calibration files found in {calibration_folder}")
            return False
        
        print(f"\nLoading calibration data from {len(csv_files)} files...")
        
        for csv_file in csv_files:
            try:
                # Extract spread and total from filename
                # Handle both formats: "3_49.csv" and "7.5 45.5.csv"
                filename = os.path.basename(csv_file)
                # Remove .csv extension
                filename_no_ext = filename.replace('.csv', '')
                # Replace spaces with underscores for consistent parsing
                filename_no_ext = filename_no_ext.replace(' ', '_')
                # Split on underscore
                parts = filename_no_ext.split('_')
                
                if len(parts) != 2:
                    print(f"Skipping {filename}: invalid filename format")
                    continue
                
                spread = float(parts[0])
                total = float(parts[1])
                
                # Load the calibration data
                cal_df = pd.read_csv(csv_file)
                
                # Verify required columns exist
                required_cols = ['Home Score', 'Away Score', 'Fair']
                if not all(col in cal_df.columns for col in required_cols):
                    print(f"Skipping {filename}: missing required columns")
                    continue
                
                # Store calibration probabilities as a dictionary keyed by score combination
                cal_probs = {}
                for _, row in cal_df.iterrows():
                    home_score = int(row['Home Score'])
                    away_score = int(row['Away Score'])
                    fair_prob = float(row['Fair'])
                    
                    score_combo = f"{home_score}-{away_score}"
                    cal_probs[score_combo] = fair_prob
                
                # Store this calibration data indexed by (spread, total)
                self.calibration_data[(spread, total)] = cal_probs
                
                print(f"  Loaded: Spread {spread:+.1f}, Total {total:.1f} ({len(cal_probs)} scores)")
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.calibration_data)} calibration datasets")
        return len(self.calibration_data) > 0

    def interpolate_calibration(self, spread, total, score_combo):
        """
        Interpolate/extrapolate calibration factor for a specific spread, total, and score combination.
        
        Uses inverse distance weighting (IDW) to combine calibration data from multiple nearby
        spread/total combinations. Scores closer in (spread, total) space have more influence.
        
        Args:
            spread: target spread value
            total: target total value
            score_combo: score combination string like "7-3"
            
        Returns:
            Interpolated calibration probability, or None if score not in any calibration data
        """
        if not hasattr(self, 'calibration_data') or not self.calibration_data:
            return None
        
        # Collect all calibration points that contain this score combination
        calibration_points = []
        for (cal_spread, cal_total), cal_probs in self.calibration_data.items():
            if score_combo in cal_probs:
                # Calculate distance in normalized (spread, total) space
                # Normalize spread by 10 and total by 20 to weight them appropriately
                spread_dist = (spread - cal_spread) / 10.0
                total_dist = (total - cal_total) / 20.0
                distance = np.sqrt(spread_dist**2 + total_dist**2)
                
                calibration_points.append({
                    'spread': cal_spread,
                    'total': cal_total,
                    'prob': cal_probs[score_combo],
                    'distance': distance
                })
        
        # If no calibration data contains this score, return None
        if not calibration_points:
            return None
        
        # If we have an exact match (distance = 0), return that probability
        exact_matches = [p for p in calibration_points if p['distance'] < 1e-6]
        if exact_matches:
            return exact_matches[0]['prob']
        
        # Use inverse distance weighting (IDW) for interpolation
        # Weight = 1 / distance^power, where power controls how quickly influence drops off
        power = 2.0
        
        total_weight = 0.0
        weighted_prob = 0.0
        
        for point in calibration_points:
            # Add small epsilon to prevent division by zero
            weight = 1.0 / (point['distance']**power + 1e-10)
            weighted_prob += weight * point['prob']
            total_weight += weight
        
        # Return weighted average
        if total_weight > 0:
            return weighted_prob / total_weight
        else:
            return None

    def apply_calibration(self, model_probs, spread, total):
        """
        Apply market calibration to model predictions using Bayesian updating.
        
        For each score, we:
        1. Get model's predicted probability
        2. Get interpolated market probability from calibration data
        3. Blend them using a weighted average (more weight to market for well-calibrated scores)
        4. Normalize to ensure probabilities sum to 1
        
        The calibration strength is adaptive:
        - Stronger calibration for scores with nearby market data
        - Weaker calibration for scores far from any market data
        
        Args:
            model_probs: dictionary of model-predicted probabilities
            spread: game spread
            total: game total
            
        Returns:
            Calibrated probability distribution
        """
        if not hasattr(self, 'calibration_data') or not self.calibration_data:
            # No calibration data available, return model predictions as-is
            return model_probs
        
        calibrated_probs = {}
        
        # Determine how much to trust calibration vs model
        # We'll calculate this adaptively based on proximity to calibration data
        for score_combo, model_prob in model_probs.items():
            # Get interpolated market probability for this score
            market_prob = self.interpolate_calibration(spread, total, score_combo)
            
            if market_prob is not None:
                # We have market data for this score, blend model and market
                # Calculate distance to nearest calibration point for this score
                min_distance = float('inf')
                for (cal_spread, cal_total), cal_probs in self.calibration_data.items():
                    if score_combo in cal_probs:
                        spread_dist = (spread - cal_spread) / 10.0
                        total_dist = (total - cal_total) / 20.0
                        distance = np.sqrt(spread_dist**2 + total_dist**2)
                        min_distance = min(min_distance, distance)
                
                # Calibration weight decreases with distance
                # At distance=0 (exact match), weight=0.7 (70% market, 30% model)
                # At distance=2, weight=0.35 (35% market, 65% model)
                # At distance=5+, weight approaches 0 (mostly model)
                max_calibration_weight = 0.7
                distance_decay = np.exp(-min_distance / 2.0)
                calibration_weight = max_calibration_weight * distance_decay
                
                # Blend model and market probabilities
                blended_prob = (calibration_weight * market_prob + 
                            (1 - calibration_weight) * model_prob)
                calibrated_probs[score_combo] = blended_prob
            else:
                # No market data for this score, use model prediction
                calibrated_probs[score_combo] = model_prob
        
        # Normalize to ensure probabilities sum to 1
        total_prob = sum(calibrated_probs.values())
        if total_prob > 0:
            calibrated_probs = {k: v / total_prob for k, v in calibrated_probs.items()}
        
        return calibrated_probs
    def fit_model(self):
        """
        Fit logistic regression models for each common score combination.
        For each score, we model P(score | spread, total) using a logistic function.
        This allows us to adjust probabilities based on game-specific betting lines.
        
        Model features:
        - Intercept: baseline log-odds for this score
        - Spread: linear effect of point spread (Note: spread is from home team's perspective and is negative if home is favored, positive if away is favored
                                                 Example: if spread is -7, home team is favored by 7 points)
        - Total: linear effect of total points line
        - Spread^2: captures non-linear spread effects
        - Total^2: captures non-linear total effects
        """
        if self.historical_data is None or not self.empirical_distribution:
            return
            
        # Only model scores that occur frequently enough for reliable estimation
        # Minimum threshold scales with dataset size
        min_occurrences = max(3, len(self.historical_data) // 1000)
        common_scores = {k: v for k, v in self.empirical_distribution.items() 
                        if v >= min_occurrences / len(self.historical_data)}
        
        print(f"\nFitting models for {len(common_scores)} common score combinations")
        print(f"Minimum occurrences threshold: {min_occurrences}")
        
        # Build feature matrix for all games with common scores
        features = []
        score_labels = []
        
        for _, row in self.historical_data.iterrows():
            if row['score_combination'] in common_scores:
                # Normalize spread and total to improve optimization
                # Spread normalized by dividing by 10
                norm_spread = row['pregame_spread'] / 10.0
                # Total normalized by centering at 50 and dividing by 20
                norm_total = (row['pregame_total'] - 50) / 20.0
                
                # Feature vector: [1, spread, total, spread^2, total^2]
                # The 1 is the intercept term
                feature_vector = [
                    1.0,
                    norm_spread,
                    norm_total,
                    norm_spread**2,
                    norm_total**2
                ]
                
                features.append(feature_vector)
                score_labels.append(row['score_combination'])
        
        # Convert to numpy arrays for efficient computation
        self.X = np.array(features)
        self.y = score_labels
        self.score_list = list(common_scores.keys())
        self.score_to_idx = {score: idx for idx, score in enumerate(self.score_list)}
        
        print(f"Feature matrix shape: {self.X.shape[0]} games x {self.X.shape[1]} features")
        print(f"\nSample feature statistics:")
        print(f"  Spread (normalized): mean={np.mean(self.X[:, 1]):.3f}, std={np.std(self.X[:, 1]):.3f}")
        print(f"  Total (normalized): mean={np.mean(self.X[:, 2]):.3f}, std={np.std(self.X[:, 2]):.3f}")
        
        self.model_params = {}
        successful_fits = 0
        
        print(f"\nFitting individual logistic regression models...")
        
        # Fit a separate binary logistic regression for each score combination
        # This is a "one-vs-all" approach where each model predicts P(this_score | features)
        # Note that score is in terms of home team but unlike spread it is positive if home team is winning. For example 7-0 means home is up by 7.
        for i, score in enumerate(self.score_list):
            # Create binary target: 1 if this score, 0 otherwise
            y_binary = np.array([1 if label == score else 0 for label in self.y])
            
            # Calculate prior probability and convert to log-odds (logit) space
            # This serves as a starting point and regularization anchor
            prior_prob = self.empirical_distribution[score]
            prior_prob_clipped = np.clip(prior_prob, 1e-6, 1 - 1e-6)
            prior_logit = np.log(prior_prob_clipped / (1 - prior_prob_clipped))
            
            def objective(params):
                """
                Objective function for optimization: negative log-likelihood with L2 regularization.
                We minimize this to find the best model parameters.
                """
                # Calculate predicted log-odds for each game
                logits = self.X @ params
                # Clip to prevent numerical overflow in exp()
                logits_clipped = np.clip(logits, -50, 50)
                # Convert log-odds to probabilities using sigmoid function
                probs = 1 / (1 + np.exp(-logits_clipped))
                # Clip probabilities away from 0 and 1 to prevent log(0)
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                
                # Calculate log-likelihood: sum of log P(y|x) across all games
                log_likelihood = np.sum(y_binary * np.log(probs) + (1 - y_binary) * np.log(1 - probs))
                # Negative log-likelihood (we minimize this)
                nll = -log_likelihood
                
                # L2 regularization: penalize parameters that drift far from prior
                # This prevents overfitting, especially for rare scores
                prior_means = np.array([prior_logit, 0, 0, 0, 0])
                # Regularization strength increases for rarer scores
                reg_strength = 0.1 * (1 + 1/max(prior_prob * len(self.historical_data), 5))
                l2_penalty = reg_strength * np.sum((params - prior_means)**2)
                
                # Total loss to minimize
                total_loss = nll + l2_penalty
                return total_loss if np.isfinite(total_loss) else 1e10
            
            # Initialize parameters near prior with small random noise
            init_params = np.array([prior_logit, 0, 0, 0, 0]) + np.random.normal(0, 0.01, 5)
            
            # Set bounds for optimization to keep parameters reasonable
            # Intercept can vary within 5 units of prior logit
            # Spread/Total linear effects bounded to [-2, 2]
            # Quadratic effects bounded to [-1, 1]
            param_bounds = [
                (prior_logit - 5, prior_logit + 5),
                (-2, 2),
                (-2, 2),
                (-1, 1),
                (-1, 1)
            ]
            
            optimization_success = False
            
            # Try L-BFGS-B optimization first (gradient-based, efficient)
            try:
                result = minimize(objective, init_params, method='L-BFGS-B', bounds=param_bounds)
                if result.success and np.isfinite(result.fun):
                    self.model_params[score] = result.x
                    optimization_success = True
                    successful_fits += 1
            except:
                pass
            
            # If L-BFGS-B fails, try Nelder-Mead (gradient-free, more robust)
            if not optimization_success:
                try:
                    result = minimize(objective, init_params, method='Nelder-Mead')
                    if result.success and np.isfinite(result.fun) and np.all(np.abs(result.x) < 10):
                        self.model_params[score] = result.x
                        optimization_success = True
                        successful_fits += 1
                except:
                    pass
            
            # If both methods fail, raise error
            if not optimization_success:
                raise RuntimeError(f"Optimization failed for score {score}")
        
        print(f"Successfully fitted {successful_fits}/{len(self.score_list)} models")
        
        # Display sample coefficients to verify reasonable values
        # Would assume that the more favored the home team is the more the home team scores and less the away team scores and vice versa when away team is favored. 
        # Ex we should see more probability of 7-0, 14-0 as spread gets more negative
        print(f"\nSample model coefficients (first 5 scores):")
        print(f"{'Score':<10} {'Intercept':<10} {'Spread':<10} {'Total':<10} {'Spread^2':<10} {'Total^2':<10}")
        print("-" * 70)
        for score in list(self.model_params.keys())[:5]:
            params = self.model_params[score]
            print(f"{score:<10} {params[0]:<10.3f} {params[1]:<10.3f} {params[2]:<10.3f} {params[3]:<10.3f} {params[4]:<10.3f}")
    
    def predict_score_probabilities(self, spread, total):
        """
        Predict probability distribution over all possible scores given a spread and total.
        Now includes calibration layer that adjusts predictions based on market data.
        
        Process:
        1. For scores with fitted models: use logistic regression to adjust empirical probability
        2. For rare scores without models: use heuristic adjustment based on distance from expected outcome
        3. Apply market calibration using interpolated data from calibration files
        4. Normalize all probabilities to sum to 1
        
        Args:
            spread: pregame point spread (negative = home favored)
            total: pregame total points line
            
        Returns:
            Dictionary mapping score combinations to probabilities
        """
        if not self.model_params:
            return {}
        
        # Normalize inputs the same way as during training
        norm_spread = spread / 10.0
        norm_total = (total - 50) / 20.0
        x_new = np.array([1.0, norm_spread, norm_total, norm_spread**2, norm_total**2])
        
        updated_probs = {}
        
        # For common scores with fitted models: calculate adjusted probability
        for score in self.score_list:
            params = self.model_params[score]
            # Calculate log-odds using model parameters
            logit = x_new @ params
            # Clip to prevent overflow
            logit_clipped = np.clip(logit, -50, 50)
            # Convert to probability using sigmoid
            prob = 1 / (1 + np.exp(-logit_clipped))
            updated_probs[score] = prob
        
        # For rare scores without models: use distance-based heuristic
        # Scores closer to expected outcome get higher adjusted probability
        for score, emp_prob in self.empirical_distribution.items():
            if score not in updated_probs:
                # Parse the score combination
                home_score, away_score = map(int, score.split('-'))
                actual_margin = home_score - away_score
                actual_total = home_score + away_score
                
                # Expected margin: opposite of spread (spread is home's disadvantage)
                # Expected Q1 total: 25% of game total
                expected_margin = -spread
                expected_q1_total = total * 0.25
                
                # Calculate how far this score is from expectations
                margin_diff = abs(actual_margin - expected_margin)
                total_diff = abs(actual_total - expected_q1_total)
                
                # Apply exponential decay: scores far from expectation get reduced probability
                margin_factor = np.exp(-margin_diff / 10.0)
                total_factor = np.exp(-total_diff / 5.0)
                
                # Adjust empirical probability based on distance factors
                adjusted_prob = emp_prob * margin_factor * total_factor
                updated_probs[score] = adjusted_prob
        
        # Normalize before calibration
        total_prob = sum(updated_probs.values())
        if total_prob > 0:
            updated_probs = {score: prob / total_prob for score, prob in updated_probs.items()}
        
        # Apply market calibration if available
        calibrated_probs = self.apply_calibration(updated_probs, spread, total)
        
        return calibrated_probs
    
    def calculate_betting_markets(self, all_probs, spread, total):
        """
        Derive various betting market probabilities and odds from the score distribution.
        
        Calculates:
        - Moneyline (2-way and 3-way)
        - Spread lines at multiple points
        - Total lines at multiple points
        - Special draw markets
        
        Args:
            all_probs: complete probability distribution over scores
            spread: game spread (used to initialize spread line search)
            total: game total (used to initialize total line search)
            
        Returns:
            Dictionary with all betting markets and their probabilities/odds
        """
        # Calculate moneyline probabilities by summing over relevant scores
        home_win_prob = 0.0
        away_win_prob = 0.0
        tie_prob = 0.0
        
        for score_combo, prob in all_probs.items():
            try:
                home_score, away_score = map(int, score_combo.split('-'))
                if home_score > away_score:
                    home_win_prob += prob
                elif away_score > home_score:
                    away_win_prob += prob
                else:
                    tie_prob += prob
            except:
                continue
        
        # Calculate 2-way moneyline (ties are void/push)
        total_decisive = home_win_prob + away_win_prob
        if total_decisive > 0:
            home_ml_2way = home_win_prob / total_decisive
            away_ml_2way = away_win_prob / total_decisive
        else:
            home_ml_2way = away_ml_2way = 0.5
        
        def prob_to_american_odds(prob):
            """
            Convert probability to American odds format.
            Probability >= 0.5 becomes negative odds (favorite)
            Probability < 0.5 becomes positive odds (underdog)
            """
            if prob <= 0 or prob >= 1:
                return 0
            if prob >= 0.5:
                return int(-100 * prob / (1 - prob))
            else:
                return int(100 * (1 - prob) / prob)
        
        def calculate_spread_prob(line):
            """
            Calculate probability that home team covers a given spread line.
            Line is the number of points home team must win by.
            Negative line = home gets points
            Positive line = home must win by more than line
            """
            cover_prob = 0.0
            for score_combo, prob in all_probs.items():
                try:
                    home_score, away_score = map(int, score_combo.split('-'))
                    margin = home_score - away_score
                    # Home covers if their margin is greater than the line
                    if margin > line:
                        cover_prob += prob
                except:
                    continue
            return cover_prob
        
        def round_to_half(value):
            """
            Round to nearest 0.5, but avoid whole numbers (always add 0.5 to whole numbers).
            This ensures no ties/pushes on spread and total bets.
            """
            rounded = round(value * 2) / 2
            if rounded == int(rounded):
                return rounded + 0.5
            return rounded
        
        # Find optimal Q1 spread line that makes both sides closest to 50%
        game_spread = spread
        # Initial guess: quarter of the game spread
        start_line = round_to_half(game_spread / 4)
        
        tested_lines = {}
        tested_lines[start_line] = calculate_spread_prob(start_line)
        
        best_line = start_line
        best_distance = abs(tested_lines[start_line] - 0.5)
        
        # Search in direction that moves closer to 50%
        direction = 1 if tested_lines[start_line] > 0.5 else -1
        current_line = start_line + direction
        
        # Keep searching while improving and within reasonable bounds
        while -20.5 <= current_line <= 20.5:
            prob = calculate_spread_prob(current_line)
            tested_lines[current_line] = prob
            distance = abs(prob - 0.5)
            
            if distance < best_distance:
                best_distance = distance
                best_line = current_line
                current_line += direction
            else:
                break
        
        # Also search in opposite direction in case we started on wrong side
        opposite_direction = -direction
        current_line = start_line + opposite_direction
        
        while -20.5 <= current_line <= 20.5:
            prob = calculate_spread_prob(current_line)
            tested_lines[current_line] = prob
            distance = abs(prob - 0.5)
            
            if distance < best_distance:
                best_distance = distance
                best_line = current_line
                current_line += opposite_direction
            else:
                break
        
        # The line closest to 50% is our median/fair spread
        median_spread = best_line
        
        # Generate spread lines at different offsets from median for display
        spread_lines = {}
        for offset in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
            line = median_spread + offset
            if line not in tested_lines:
                tested_lines[line] = calculate_spread_prob(line)
            spread_lines[line] = {
                'home_cover': tested_lines[line],
                'away_cover': 1 - tested_lines[line],
                'home_odds': prob_to_american_odds(tested_lines[line]),
                'away_odds': prob_to_american_odds(1 - tested_lines[line])
            }

        def calculate_total_prob(line):
            """
            Calculate probability that combined score goes over a given total line.
            """
            over_prob = 0.0
            for score_combo, prob in all_probs.items():
                try:
                    home_score, away_score = map(int, score_combo.split('-'))
                    actual_total = home_score + away_score
                    if actual_total > line:
                        over_prob += prob
                except:
                    continue
            return over_prob
        
        # Find optimal Q1 total line (similar process to spread)
        game_total = total
        # Initial guess: 20% of game total (Q1 tends to be slightly lower scoring)
        start_total_line = round_to_half(game_total / 5)
        
        tested_total_lines = {}
        tested_total_lines[start_total_line] = calculate_total_prob(start_total_line)
        
        best_total_line = start_total_line
        best_total_distance = abs(tested_total_lines[start_total_line] - 0.5)
        
        # Search for line closest to 50%
        total_direction = 1 if tested_total_lines[start_total_line] > 0.5 else -1
        current_total_line = start_total_line + total_direction
        
        while 0.5 <= current_total_line <= 35.5:
            prob = calculate_total_prob(current_total_line)
            tested_total_lines[current_total_line] = prob
            distance = abs(prob - 0.5)
            
            if distance < best_total_distance:
                best_total_distance = distance
                best_total_line = current_total_line
                current_total_line += total_direction
            else:
                break
        
        # Search opposite direction
        opposite_total_direction = -total_direction
        current_total_line = start_total_line + opposite_total_direction
        
        while 0.5 <= current_total_line <= 35.5:
            prob = calculate_total_prob(current_total_line)
            tested_total_lines[current_total_line] = prob
            distance = abs(prob - 0.5)
            
            if distance < best_total_distance:
                best_total_distance = distance
                best_total_line = current_total_line
                current_total_line += opposite_total_direction
            else:
                break
        
        median_total = best_total_line

        # Generate total lines at different offsets from median
        total_lines = {}
        for offset in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            line = median_total + offset
            if line > 0:
                if line not in tested_total_lines:
                    tested_total_lines[line] = calculate_total_prob(line)
                total_lines[line] = {
                    'over': tested_total_lines[line],
                    'under': 1 - tested_total_lines[line],
                    'over_odds': prob_to_american_odds(tested_total_lines[line]),
                    'under_odds': prob_to_american_odds(1 - tested_total_lines[line])
                }
        
        # Special markets: probability of draw with over certain totals
        # Useful for parlay constructions
        draw_over_10_5 = 0.0
        draw_over_12_5 = 0.0
        for score_combo, prob in all_probs.items():
            try:
                home_score, away_score = map(int, score_combo.split('-'))
                if home_score == away_score:
                    total_pts = home_score + away_score
                    if total_pts > 10.5:
                        draw_over_10_5 += prob
                    if total_pts > 12.5:
                        draw_over_12_5 += prob
            except:
                continue
        
        # Return all betting markets
        return {
            'moneyline_2way': {
                'home': home_ml_2way,
                'away': away_ml_2way,
                'home_odds': prob_to_american_odds(home_ml_2way),
                'away_odds': prob_to_american_odds(away_ml_2way)
            },
            'moneyline_3way': {
                'home': home_win_prob,
                'away': away_win_prob,
                'draw': tie_prob,
                'home_odds': prob_to_american_odds(home_win_prob),
                'away_odds': prob_to_american_odds(away_win_prob),
                'draw_odds': prob_to_american_odds(tie_prob)
            },
            'spread': spread_lines,
            'total': total_lines,
            'draw_and_over_10_5': draw_over_10_5,
            'draw_and_over_12_5': draw_over_12_5
        }
    
    def predict(self, spread, total):
        """
        Main prediction function: generates full probability distribution and all betting markets.
        Displays detailed output showing:
        - How probabilities were adjusted from empirical baseline
        - All scores with meaningful probability
        - All betting markets with odds
        
        Args:
            spread: pregame point spread
            total: pregame total points line
        """
        # Get probability distribution over all scores
        all_probs = self.predict_score_probabilities(spread, total)
        
        print(f"\n{'='*80}")
        print(f"PREDICTION: Spread {spread:+.1f}, Total {total:.1f}")
        print(f"{'='*80}")
        
        # Show how model adjusted probabilities from baseline
        print(f"\nProbability Adjustment Analysis (Top 15 Scores):")
        print(f"{'Score':<10} {'Empirical':<12} {'Adjusted':<12} {'Change':<12} {'Final %'}")
        print("-" * 70)
        
        sorted_final = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:15]
        for score, final_prob in sorted_final:
            emp_prob = self.empirical_distribution.get(score, 0)
            change = final_prob - emp_prob
            print(f"{score:<10} {emp_prob:<12.6f} {final_prob:<12.6f} {change:+12.6f} {final_prob*100:.2f}%")
        
        # Display all scores with non-trivial probability
        print(f"\n{'='*80}")
        print(f"SCORES WITH PROBABILITY >= 0.5%")
        print(f"{'='*80}")
        
        filtered_probs = {k: v for k, v in all_probs.items() if v >= 0.005}
        sorted_scores = sorted(filtered_probs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Score':<10} {'Probability':<15} {'Percentage'}")
        print("-" * 50)
        for score, prob in sorted_scores:
            print(f"{score:<10} {prob:<15.6f} {prob*100:.2f}%")
        
        # Calculate all betting markets
        markets = self.calculate_betting_markets(all_probs, spread, total)
        
        print(f"\n{'='*80}")
        print(f"BETTING MARKETS")
        print(f"{'='*80}")
        
        # Display moneyline markets
        ml2 = markets['moneyline_2way']
        print(f"\n2-Way Moneyline (Draws Void):")
        print(f"  Home: {ml2['home']:.4f} ({ml2['home']*100:.2f}%) - Odds: {ml2['home_odds']:+d}")
        print(f"  Away: {ml2['away']:.4f} ({ml2['away']*100:.2f}%) - Odds: {ml2['away_odds']:+d}")
        
        ml3 = markets['moneyline_3way']
        print(f"\n3-Way Moneyline:")
        print(f"  Home: {ml3['home']:.4f} ({ml3['home']*100:.2f}%) - Odds: {ml3['home_odds']:+d}")
        print(f"  Draw: {ml3['draw']:.4f} ({ml3['draw']*100:.2f}%) - Odds: {ml3['draw_odds']:+d}")
        print(f"  Away: {ml3['away']:.4f} ({ml3['away']*100:.2f}%) - Odds: {ml3['away_odds']:+d}")
        
        # Display spread markets at various lines
        print(f"\nSpread Markets:")
        for line in sorted(markets['spread'].keys()):
            sp = markets['spread'][line]
            print(f"  Line {-line:+.1f}: Home {sp['home_cover']:.4f} ({sp['home_cover']*100:.2f}%, {sp['home_odds']:+d}) | Away {sp['away_cover']:.4f} ({sp['away_cover']*100:.2f}%, {sp['away_odds']:+d})")
        
        # Display total markets at various lines
        print(f"\nTotal Markets:")
        for line in sorted(markets['total'].keys()):
            tot = markets['total'][line]
            print(f"  Line {line:.1f}: Over {tot['over']:.4f} ({tot['over']*100:.2f}%, {tot['over_odds']:+d}) | Under {tot['under']:.4f} ({tot['under']*100:.2f}%, {tot['under_odds']:+d})")
        
        # Display special draw markets
        print(f"\nSpecial Markets:")
        print(f"  Draw and Over 10.5: {markets['draw_and_over_10_5']:.4f} ({markets['draw_and_over_10_5']*100:.2f}%)")
        print(f"  Draw and Over 12.5: {markets['draw_and_over_12_5']:.4f} ({markets['draw_and_over_12_5']*100:.2f}%)")
    
    def calculate_parlay_probability(self, all_probs, selections):
        """
        Calculate probability of a parlay (multiple bets that all must win) using full score distribution.
        
        Works by iterating through all possible scores and checking if each score satisfies
        ALL conditions in the parlay. Sums probabilities of qualifying scores.
        
        Args:
            all_probs: complete probability distribution over scores
            selections: list of bet dictionaries, each containing:
                - type: 'spread', 'total', 'moneyline', or 'moneyline_3way'
                - team/side: which side of bet
                - line: the line value (for spread/total)
        
        Example selections:
        [
            {'type': 'spread', 'team': 'home', 'line': -3.5},
            {'type': 'total', 'side': 'over', 'line': 9.5},
            {'type': 'moneyline', 'team': 'away'}
        ]
        
        Returns:
            Probability that all selections win (parlay hits)
        """
        parlay_prob = 0.0
        
        # Iterate through all possible score combinations
        for score_combo, prob in all_probs.items():
            try:
                home_score, away_score = map(int, score_combo.split('-'))
                
                # Check if this score satisfies ALL selections in the parlay
                all_conditions_met = True
                
                for selection in selections:
                    # Check spread bets
                    if selection['type'] == 'spread':
                        margin = home_score - away_score
                        line = selection['line']
                        
                        if selection['team'] == 'home':
                            # Home covers if margin beats the spread
                            # If line is -2.5 (home favored), margin must be > 2.5
                            # If line is +2.5 (home underdog), margin must be > -2.5
                            if margin <= -line:
                                all_conditions_met = False
                                break
                        else:  # away
                            # Away covers if they beat the spread from their perspective
                            # If line is -2.5 (home favored, away gets +2.5), away margin must be < -2.5
                            # If line is +2.5 (away favored), away margin must be < 2.5
                            if margin >= -line:
                                all_conditions_met = False
                                break
                    
                    # Check total bets
                    elif selection['type'] == 'total':
                        total = home_score + away_score
                        if selection['side'] == 'over':
                            if total <= selection['line']:
                                all_conditions_met = False
                                break
                        else:  # under
                            if total >= selection['line']:
                                all_conditions_met = False
                                break
                    
                    # Check 2-way moneyline bets (ties void)
                    elif selection['type'] == 'moneyline':
                        if selection['team'] == 'home':
                            if home_score <= away_score:
                                all_conditions_met = False
                                break
                        else:  # away
                            if away_score <= home_score:
                                all_conditions_met = False
                                break
                    
                    # Check 3-way moneyline bets (ties are a separate outcome)
                    elif selection['type'] == 'moneyline_3way':
                        if selection['team'] == 'home':
                            if home_score <= away_score:
                                all_conditions_met = False
                                break
                        elif selection['team'] == 'away':
                            if away_score <= home_score:
                                all_conditions_met = False
                                break
                        elif selection['team'] == 'draw':
                            if home_score != away_score:
                                all_conditions_met = False
                                break
                
                # If this score satisfies all conditions, add its probability to parlay total
                if all_conditions_met:
                    parlay_prob += prob
                    
            except:
                continue
        
        return parlay_prob

    def prob_to_american_odds(self, prob):
        """
        Convert probability to American odds format.
        
        American odds format:
        - Negative odds (e.g. -150): amount you must bet to win $100
        - Positive odds (e.g. +200): amount you win if you bet $100
        
        Favorites (prob >= 0.5) get negative odds
        Underdogs (prob < 0.5) get positive odds
        """
        if prob <= 0 or prob >= 1:
            return 0
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)

    def calculate_parlay_payout(self, prob):
        """
        Calculate parlay payout from true probability.
        Converts probability to American odds to show potential payout.
        
        Args:
            prob: true probability of parlay hitting
            
        Returns:
            American odds representing the fair payout
        """
        if prob <= 0 or prob >= 1:
            return 0
        # Convert to decimal odds
        decimal_odds = 1 / prob
        # Return American odds
        return self.prob_to_american_odds(prob)

# Load environment variables from .env file for database credentials
load_dotenv()

def main():
    """
    Main execution function.
    
    Steps:
    1. Load database configuration from environment variables
    2. Initialize predictor and load historical data
    3. Calculate empirical distribution
    4. Fit statistical models
    5. Enter interactive loop for predictions
    """
    # Database configuration from environment variables
    # Uses defaults if environment variables not set
    db_config = {
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'cfb')
    }
    
    # Initialize the predictor
    predictor = CFBQuarterScorePredictor(db_config)
    
    print("="*80)
    print("CFB QUARTER 1 SCORE PREDICTOR")
    print("="*80)
    
    # Load all historical first quarter data
    print("\nLoading historical data...")
    if not predictor.load_historical_data():
        print("Failed to load data")
        return
    
    # Calculate baseline probability distribution
    print("\nCalculating empirical distribution...")
    predictor.calculate_empirical_distribution()
    
    # Fit logistic regression models for each score
    print("\nFitting model...")
    predictor.fit_model()

    # Load calibration data from market
    calibration_folder = '/Users/jarrettlysek/Documents/PythonWork/CFB/Q1/Calibration'
    print("\nLoading market calibration data...")
    predictor.load_calibration_data(calibration_folder)

    print("\n" + "="*80)
    print("MODEL READY")
    print("="*80)
    print("Enter spread and total for predictions")
    print("Format: spread total (e.g., -3.5 58.5)")
    print("Type 'quit' to exit\n")
    
    # Interactive prediction loop
    while True:
        try:
            # Get user input
            user_input = input("Enter spread and total (Ex Home team favored by 3.5 (if home team is favored use negative sign for spread) and total of 50.5 should be entered as: -3.5 50.5 ): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            # Parse spread and total from input
            parts = user_input.split()
            if len(parts) != 2:
                print("Enter two numbers: spread and total")
                continue
                
            spread = float(parts[0])
            total = float(parts[1])
            
            # Validate input ranges
            if not (-50 <= spread <= 50) or not (30 <= total <= 90):
                print("Use realistic values: spread -50 to 50, total 30 to 90")
                continue
                
            # Generate and display prediction
            predictor.predict(spread, total)
            
        except ValueError:
            print("Enter valid numbers")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()