# College Football Quarter Scoring Prediction Model

A Bayesian statistical model that predicts the probability distribution of first quarter scores in college football games using historical data and pregame betting markets.

## How It Works

### The Bayesian Framework

**Goal:** Calculate P(Q1 Score | Pregame Spread & Total)

**Data:** College football games including a FBS team from 2014 to present along with pregame closing spread and totals 

**Bayesian Equation:**
```
P(Q1 Score | Market Info) = [P(Q1 Score) × P(Market Info | Q1 Score)] / P(Market Info)
```

### Three Components

#### 1. The Prior: P(Q1 Score)
The empirical distribution of all historical Q1 scores. This represents our baseline expectation when we know nothing about a specific game. 

#### 2. The Likelihood: P(Market Info | Q1 Score)
Found using multinomial logistic regression. For each possible Q1 score, we model what pregame spread and total we would expect to see. Each score gets its own model with 5 learned coefficients:

```
logit(p) = β₀ + β₁·spread + β₂·total + β₃·spread² + β₄·total²
```

**Coefficient Meanings:**
- **β₀ (Intercept):** Baseline probability when spread and total are zero
- **β₁ (Spread):** How a 1-point change in spread affects this score's probability
- **β₂ (Total):** How a 1-point change in total affects this score's probability
- **β₃ (Spread²):** Captures non-linear spread effects (move from 3 to 6 matters more than 30 to 33)
- **β₄ (Total²):** Captures non-linear total effects

**Regularization:** We apply L2 penalty to prevent overfitting, especially for rare scores:
```
Loss = -LogLikelihood + λ·||β - β_prior||²
```
Rare scores get stronger regularization (higher λ) to keep predictions reasonable.

#### 3. The Evidence: P(Market Info)
Normalizes probabilities so they sum to 1 across all possible scores.

## Project Structure

### Core Files

**bayesQ1.py**
- Main Bayesian model implementation
- Generates Q1 score probability distributions

**scraper.py**
- Pulls data from collegefootballdata.com API
- Populates MySQL database with games, scores, and betting lines
- Maintains updated historical dataset

**prediction_server.py & CFBinterface.html**
- Web-based GUI for model interaction

## Database Schema

### cfb.games
Stores every FBS game since 2014 with final scores and betting information.

| Column | Type | Description |
|--------|------|-------------|
| game_id | Integer | Unique identifier from CFBD API |
| season | Integer | Year of season |
| week | Integer | Week number |
| home_team | String | Home team name |
| away_team | String | Away team name |
| home_points | Integer | Final home score |
| away_points | Integer | Final away score |
| neutral_site | Integer | 1 if neutral site, 0 otherwise |
| conference_game | Integer | 1 if conference game, 0 otherwise |
| pregame_spread | Float | Closing spread (negative = home favored) |
| pregame_total | Float | Closing total (over/under) |
| home_ml_prob | Float | Home moneyline probability (2021+) |
| away_ml_prob | Float | Away moneyline probability (2021+) |
| betting_provider | String | Source of betting line |

### cfb.quarter_scoring
Stores quarter-by-quarter scoring for every game (4 rows per game).

| Column | Type | Description |
|--------|------|-------------|
| game_id | Integer | Links to games table |
| quarter | Integer | Quarter number (1-4) |
| home_score | Integer | Home points scored in this quarter only |
| away_score | Integer | Away points scored in this quarter only |
| total_score | Integer | Computed (home_score + away_score) |

## Installation & Usage

### Requirements
```
Python 3.8+
MySQL database
Required packages: numpy, pandas, scikit-learn, scipy, mysql-connector-python, flask
```

## Future Development

### Next Steps
1. **Quantify first possession value** - Analyze how much getting the ball first matters based on spread and total
2. **Coach configurations** - Incorporate coaching tendencies into predictions
3. **Expand to other quarters** - Extend model to Q2, Q3, and Q4 predictions


## Contributing

This is an active research project. Questions, suggestions, and bug reports are welcome via GitHub issues.

**Repository:** https://github.com/jlysek/CFB_Q1

**License:** To be determined

**Academic Use:** Please cite appropriately if used in research

## Technical Notes

### Data Normalization
Spread and total are normalized before regression to improve numerical stability during optimization. This prevents large raw values from dominating the coefficient learning process.

### Handling Rare Scores
For extremely rare score combinations with minimal historical data, the model uses distance-based smoothing, borrowing information from similar scores to make reasonable predictions.

### Model Validation
Predictions are validated against holdout test sets and compared to actual market prices where available. The model's probability distributions should align with implied probabilities from betting markets when properly calibrated.

---
