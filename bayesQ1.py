import os
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')



class CFBQuarterScorePredictor:
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.empirical_distribution = {}
        self.model_params = {}
        self.historical_data = None
        
    def connect_to_database(self):
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            return None
    
    def load_historical_data(self):
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
                q1_home.away_score as q1_away_score
            FROM cfb.games g
            JOIN cfb.quarter_scoring q1_home ON g.game_id = q1_home.game_id 
            WHERE q1_home.quarter = 1
            AND g.pregame_spread IS NOT NULL
            AND g.pregame_total IS NOT NULL
            AND g.pregame_spread BETWEEN -70 AND 70
            AND g.pregame_total BETWEEN 20 AND 90
            ORDER BY g.game_id
            """
            
            self.historical_data = pd.read_sql(query, connection)
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
        if self.historical_data is None:
            return
            
        score_counts = self.historical_data['score_combination'].value_counts()
        total_games = len(self.historical_data)
        
        self.empirical_distribution = {}
        for score_combo, count in score_counts.items():
            self.empirical_distribution[score_combo] = count / total_games
        
        print(f"\nEmpirical Distribution - Top 20 Scores:")
        print(f"{'Score':<10} {'Count':<8} {'Probability':<12} {'Percentage'}")
        print("-" * 60)
        sorted_dist = sorted(self.empirical_distribution.items(), key=lambda x: x[1], reverse=True)
        for i, (score, prob) in enumerate(sorted_dist[:20]):
            count = int(prob * total_games)
            print(f"{score:<10} {count:<8} {prob:<12.6f} {prob*100:.2f}%")
    
    def fit_bayesian_model(self):
        if self.historical_data is None or not self.empirical_distribution:
            return
            
        min_occurrences = max(3, len(self.historical_data) // 1000)
        common_scores = {k: v for k, v in self.empirical_distribution.items() 
                        if v >= min_occurrences / len(self.historical_data)}
        
        print(f"\nFitting Bayesian models for {len(common_scores)} common score combinations")
        print(f"Minimum occurrences threshold: {min_occurrences}")
        
        features = []
        score_labels = []
        
        for _, row in self.historical_data.iterrows():
            if row['score_combination'] in common_scores:
                norm_spread = row['pregame_spread'] / 10.0
                norm_total = (row['pregame_total'] - 50) / 20.0
                
                feature_vector = [
                    1.0,
                    norm_spread,
                    norm_total,
                    norm_spread**2,
                    norm_total**2
                ]
                
                features.append(feature_vector)
                score_labels.append(row['score_combination'])
        
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
        
        for i, score in enumerate(self.score_list):
            y_binary = np.array([1 if label == score else 0 for label in self.y])
            
            prior_prob = self.empirical_distribution[score]
            prior_prob_clipped = np.clip(prior_prob, 1e-6, 1 - 1e-6)
            prior_logit = np.log(prior_prob_clipped / (1 - prior_prob_clipped))
            
            def objective(params):
                logits = self.X @ params
                logits_clipped = np.clip(logits, -50, 50)
                probs = 1 / (1 + np.exp(-logits_clipped))
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                
                log_likelihood = np.sum(y_binary * np.log(probs) + (1 - y_binary) * np.log(1 - probs))
                nll = -log_likelihood
                
                prior_means = np.array([prior_logit, 0, 0, 0, 0])
                reg_strength = 0.1 * (1 + 1/max(prior_prob * len(self.historical_data), 5))
                l2_penalty = reg_strength * np.sum((params - prior_means)**2)
                
                total_loss = nll + l2_penalty
                return total_loss if np.isfinite(total_loss) else 1e10
            
            init_params = np.array([prior_logit, 0, 0, 0, 0]) + np.random.normal(0, 0.01, 5)
            
            param_bounds = [
                (prior_logit - 5, prior_logit + 5),
                (-2, 2),
                (-2, 2),
                (-1, 1),
                (-1, 1)
            ]
            
            optimization_success = False
            
            try:
                result = minimize(objective, init_params, method='L-BFGS-B', bounds=param_bounds)
                if result.success and np.isfinite(result.fun):
                    self.model_params[score] = result.x
                    optimization_success = True
                    successful_fits += 1
            except:
                pass
            
            if not optimization_success:
                try:
                    result = minimize(objective, init_params, method='Nelder-Mead')
                    if result.success and np.isfinite(result.fun) and np.all(np.abs(result.x) < 10):
                        self.model_params[score] = result.x
                        optimization_success = True
                        successful_fits += 1
                except:
                    pass
            
            if not optimization_success:
                raise RuntimeError(f"Optimization failed for score {score}")
        
        print(f"Successfully fitted {successful_fits}/{len(self.score_list)} models")
        
        print(f"\nSample model coefficients (first 5 scores):")
        print(f"{'Score':<10} {'Intercept':<10} {'Spread':<10} {'Total':<10} {'Spread^2':<10} {'Total^2':<10}")
        print("-" * 70)
        for score in list(self.model_params.keys())[:5]:
            params = self.model_params[score]
            print(f"{score:<10} {params[0]:<10.3f} {params[1]:<10.3f} {params[2]:<10.3f} {params[3]:<10.3f} {params[4]:<10.3f}")
    
    def predict_score_probabilities(self, spread, total):
        if not self.model_params:
            return {}
        
        norm_spread = spread / 10.0
        norm_total = (total - 50) / 20.0
        x_new = np.array([1.0, norm_spread, norm_total, norm_spread**2, norm_total**2])
        
        updated_probs = {}
        
        for score in self.score_list:
            params = self.model_params[score]
            logit = x_new @ params
            logit_clipped = np.clip(logit, -50, 50)
            prob = 1 / (1 + np.exp(-logit_clipped))
            updated_probs[score] = prob
        
        for score, emp_prob in self.empirical_distribution.items():
            if score not in updated_probs:
                home_score, away_score = map(int, score.split('-'))
                actual_margin = home_score - away_score
                actual_total = home_score + away_score
                
                expected_margin = -spread
                expected_q1_total = total * 0.25
                
                margin_diff = abs(actual_margin - expected_margin)
                total_diff = abs(actual_total - expected_q1_total)
                
                margin_factor = np.exp(-margin_diff / 10.0)
                total_factor = np.exp(-total_diff / 5.0)
                
                adjusted_prob = emp_prob * margin_factor * total_factor
                updated_probs[score] = adjusted_prob
        
        total_prob = sum(updated_probs.values())
        if total_prob > 0:
            normalized_probs = {score: prob / total_prob for score, prob in updated_probs.items()}
        else:
            normalized_probs = self.empirical_distribution.copy()
        
        return normalized_probs
    
    def calculate_betting_markets(self, all_probs, spread, total):
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
        
        total_decisive = home_win_prob + away_win_prob
        if total_decisive > 0:
            home_ml_2way = home_win_prob / total_decisive
            away_ml_2way = away_win_prob / total_decisive
        else:
            home_ml_2way = away_ml_2way = 0.5
        
        def prob_to_american_odds(prob):
            if prob <= 0 or prob >= 1:
                return 0
            if prob >= 0.5:
                return int(-100 * prob / (1 - prob))
            else:
                return int(100 * (1 - prob) / prob)
        
        def calculate_spread_prob(line):
            cover_prob = 0.0
            for score_combo, prob in all_probs.items():
                try:
                    home_score, away_score = map(int, score_combo.split('-'))
                    margin = home_score - away_score
                    if margin > line:
                        cover_prob += prob
                except:
                    continue
            return cover_prob
        
        def round_to_half(value):
            rounded = round(value * 2) / 2
            if rounded == int(rounded):
                return rounded + 0.5
            return rounded
        
        game_spread = spread
        start_line = round_to_half(game_spread / 4)
        
        tested_lines = {}
        tested_lines[start_line] = calculate_spread_prob(start_line)
        
        best_line = start_line
        best_distance = abs(tested_lines[start_line] - 0.5)
        
        direction = 1 if tested_lines[start_line] > 0.5 else -1
        current_line = start_line + direction
        
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
        
        median_spread = best_line
        
        spread_lines = {}
        for offset in [-1.0, 0.0, 1.0]:
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
        
        game_total = total
        start_total_line = round_to_half(game_total / 5)
        
        tested_total_lines = {}
        tested_total_lines[start_total_line] = calculate_total_prob(start_total_line)
        
        best_total_line = start_total_line
        best_total_distance = abs(tested_total_lines[start_total_line] - 0.5)
        
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
        
        total_lines = {}
        for offset in [-1.0, 0.0, 1.0]:
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
        
        draw_over_11_5 = 0.0
        draw_over_12_5 = 0.0
        for score_combo, prob in all_probs.items():
            try:
                home_score, away_score = map(int, score_combo.split('-'))
                if home_score == away_score:
                    total_pts = home_score + away_score
                    if total_pts > 11.5:
                        draw_over_11_5 += prob
                    if total_pts > 12.5:
                        draw_over_12_5 += prob
            except:
                continue
        
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
            'draw_and_over_11_5': draw_over_11_5,
            'draw_and_over_12_5': draw_over_12_5
        }
    
    def predict(self, spread, total):
        all_probs = self.predict_score_probabilities(spread, total)
        
        print(f"\n{'='*80}")
        print(f"PREDICTION: Spread {spread:+.1f}, Total {total:.1f}")
        print(f"{'='*80}")
        
        print(f"\nProbability Adjustment Analysis (Top 15 Scores):")
        print(f"{'Score':<10} {'Empirical':<12} {'Adjusted':<12} {'Change':<12} {'Final %'}")
        print("-" * 70)
        
        sorted_final = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:15]
        for score, final_prob in sorted_final:
            emp_prob = self.empirical_distribution.get(score, 0)
            change = final_prob - emp_prob
            print(f"{score:<10} {emp_prob:<12.6f} {final_prob:<12.6f} {change:+12.6f} {final_prob*100:.2f}%")
        
        print(f"\n{'='*80}")
        print(f"SCORES WITH PROBABILITY >= 0.5%")
        print(f"{'='*80}")
        
        filtered_probs = {k: v for k, v in all_probs.items() if v >= 0.005}
        sorted_scores = sorted(filtered_probs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Score':<10} {'Probability':<15} {'Percentage'}")
        print("-" * 50)
        for score, prob in sorted_scores:
            print(f"{score:<10} {prob:<15.6f} {prob*100:.2f}%")
        
        markets = self.calculate_betting_markets(all_probs, spread, total)
        
        print(f"\n{'='*80}")
        print(f"BETTING MARKETS")
        print(f"{'='*80}")
        
        ml2 = markets['moneyline_2way']
        print(f"\n2-Way Moneyline (Draws Void):")
        print(f"  Home: {ml2['home']:.4f} ({ml2['home']*100:.2f}%) - Odds: {ml2['home_odds']:+d}")
        print(f"  Away: {ml2['away']:.4f} ({ml2['away']*100:.2f}%) - Odds: {ml2['away_odds']:+d}")
        
        ml3 = markets['moneyline_3way']
        print(f"\n3-Way Moneyline:")
        print(f"  Home: {ml3['home']:.4f} ({ml3['home']*100:.2f}%) - Odds: {ml3['home_odds']:+d}")
        print(f"  Draw: {ml3['draw']:.4f} ({ml3['draw']*100:.2f}%) - Odds: {ml3['draw_odds']:+d}")
        print(f"  Away: {ml3['away']:.4f} ({ml3['away']*100:.2f}%) - Odds: {ml3['away_odds']:+d}")
        
        print(f"\nSpread Markets:")
        for line in sorted(markets['spread'].keys()):
            sp = markets['spread'][line]
            print(f"  Line {-line:+.1f}: Home {sp['home_cover']:.4f} ({sp['home_cover']*100:.2f}%, {sp['home_odds']:+d}) | Away {sp['away_cover']:.4f} ({sp['away_cover']*100:.2f}%, {sp['away_odds']:+d})")
        
        print(f"\nTotal Markets:")
        for line in sorted(markets['total'].keys()):
            tot = markets['total'][line]
            print(f"  Line {line:.1f}: Over {tot['over']:.4f} ({tot['over']*100:.2f}%, {tot['over_odds']:+d}) | Under {tot['under']:.4f} ({tot['under']*100:.2f}%, {tot['under_odds']:+d})")
        
        print(f"\nSpecial Markets:")
        print(f"  Draw and Over 11.5: {markets['draw_and_over_11_5']:.4f} ({markets['draw_and_over_11_5']*100:.2f}%)")
        print(f"  Draw and Over 12.5: {markets['draw_and_over_12_5']:.4f} ({markets['draw_and_over_12_5']*100:.2f}%)")

# Load environment variables from .env file
load_dotenv()
def main():
    db_config = {
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'cfb')
    }
    
    predictor = CFBQuarterScorePredictor(db_config)
    
    print("="*80)
    print("CFB QUARTER 1 SCORE PREDICTOR")
    print("="*80)
    
    print("\nLoading historical data...")
    if not predictor.load_historical_data():
        print("Failed to load data")
        return
    
    print("\nCalculating empirical distribution...")
    predictor.calculate_empirical_distribution()
    
    print("\nFitting Bayesian model...")
    predictor.fit_bayesian_model()
    
    print("\n" + "="*80)
    print("MODEL READY")
    print("="*80)
    print("Enter spread and total for predictions")
    print("Format: spread total (e.g., -3.5 58.5)")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter spread and total: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            parts = user_input.split()
            if len(parts) != 2:
                print("Enter two numbers: spread and total")
                continue
                
            spread = float(parts[0])
            total = float(parts[1])
            
            if not (-50 <= spread <= 50) or not (30 <= total <= 90):
                print("Use realistic values: spread -50 to 50, total 30 to 90")
                continue
                
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