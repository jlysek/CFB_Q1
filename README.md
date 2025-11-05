# College Football Quarter 1 Score Prediction

A Bayesian statistical model that predicts first quarter score probability distributions in college football games using pregame betting lines. The model uses orthogonal feature decomposition and logistic regression with adaptive regularization to estimate the likelihood of every possible Q1 score outcome.

## Table of Contents

- [Overview](#overview)
- [Statistical Method](#statistical-method)
- [Database Schema](#database-schema)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)

## Overview

**Input:** Pregame spread and total  
**Output:** Probability distribution over all Q1 score outcomes

**Example:**
```
Spread: -7.5 (Home favored by 7.5)
Total: 58.5

Top Predictions:
7-0  → 9.2%
7-7  → 6.8%
14-0 → 5.4%
0-0  → 4.1%
...
```

The model trains on 7000+ FBS games from 2014-present with quarter-by-quarter scoring and pregame betting lines.

## Statistical Method

### Feature Engineering

The model uses two orthogonal features to eliminate multicollinearity:

```
implied_fav_points = (total + |spread|) / 2
implied_dog_points = (total - |spread|) / 2

feature_total  = (implied_fav_points + implied_dog_points) / typical_team_score
feature_margin = (implied_fav_points - implied_dog_points) / typical_team_score
```

**Why orthogonal?** Traditional features (spread, total) are correlated. These derived features are uncorrelated, yielding stable coefficients:
- `feature_total` captures overall scoring level
- `feature_margin` captures expected margin

### Model Architecture

**Two-tier logistic regression:**

1. **Common Scores** (≥10 occurrences): Individual binary classifiers
   ```
   logit(score) = β₀ + β₁×feature_total + β₂×feature_margin
   ```

2. **Rare Scores** (<10 occurrences): Bucket models
   - Scores grouped by outcome type (e.g., "favored blowout", "underdog win", "tie")
   - Model predicts bucket probability, then distributes among scores proportionally

### Regularization

**Elastic Net regularization** (L1 + L2) with adaptive penalties:

```python
α = 0.1 × (1 + 1/max(n_occurrences, 5))
penalty = α × [0.3 × |β - β_prior| + 0.7 × (β - β_prior)²]
```

- Rare scores → strong regularization → stay near baseline
- Common scores → weak regularization → learn from data
- L1 component performs feature selection

### Empirical Calibration

The model automatically learns Q1 scoring percentages from data:

```
Q1 Points = 23.5% of full game total
Q1 Margin = f(|spread|)  [smooth cubic spline interpolation]
```

The Q1 margin percentage varies by spread magnitude (closer games have proportionally larger Q1 margins).

### Normalization

Raw probabilities are normalized to form a valid distribution:

```python
P_final(score) = P_model(score) / Σ P_model(all_scores)
```

## Database Schema

### `cfb.games`

Every FBS game since 2014 with betting lines and final scores.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | Integer | Unique identifier |
| `season` | Integer | Year |
| `week` | Integer | Week number |
| `home_team` | String | Home team |
| `away_team` | String | Away team |
| `home_points` | Integer | Final home score |
| `away_points` | Integer | Final away score |
| `pregame_spread` | Float | Closing spread (negative = home favored) |
| `pregame_total` | Float | Closing total |
| `home_classification` | String | Division (fbs/fcs) |
| `away_classification` | String | Division (fbs/fcs) |

### `cfb.quarter_scoring`

Quarter-by-quarter scoring (4 rows per game).

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | Integer | Links to games table |
| `quarter` | Integer | Quarter number (1-4) |
| `home_score` | Integer | Home points this quarter |
| `away_score` | Integer | Away points this quarter |

### `cfb.coach_first_possession`

*Planned future feature - not yet implemented*

## Installation

### Prerequisites

- Python 3.8+
- MySQL 8.0+
- College Football Data API key ([collegefootballdata.com](https://collegefootballdata.com))

### Setup

**1. Clone repository**
```bash
git clone https://github.com/jlysek/CFB_Q1.git
cd CFB_Q1
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create MySQL database**
```sql
CREATE DATABASE cfb;
```

**4. Configure environment variables**

Create `.env` file:
```bash
CFBD_API_KEY=your_api_key_here
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=cfb
```

**5. Load historical data**
```bash
python scraper.py
```

This will take 10-15 minutes to fetch all games, scores, and betting lines since 2014.

**6. Run the model**
```bash
python Q1.py
```

## Usage

### Command Line

```bash
python Q1.py
```

**Interactive mode:**
```
Enter spread and total: -7.5 58.5
```

The model outputs:
- Top score probabilities (scores >0.05%)
- Cumulative probability coverage
- Spread/total feature values
- Orthogonality diagnostics

**Debug mode:**
```
Enter spread and total: -7.5 58.5 debug
```

Creates `debug_output.txt` with detailed breakdowns:
- Parameter values for each score
- Logit calculations
- Probability adjustments from baseline
- Model vs empirical comparisons

### Web Interface

**Start Flask server:**
```bash
python prediction_server.py
```

**Access interface:**
```
http://localhost:5000/interface
```

Features:
- Load weekly games from CFBD API
- Manual prediction input
- SGP (same game parlay) builder
- Betting market calculators

### API Endpoints

```bash
# Generate predictions
curl -X POST http://localhost:5000/api/predict-quarter-scores \
  -H "Content-Type: application/json" \
  -d '{"spread": -7.5, "total": 58.5}'

# Health check
curl http://localhost:5000/api/health

# Model status
curl http://localhost:5000/api/model-status
```

## Project Structure

```
CFB_Q1/
├── Q1.py                    # Core prediction model
├── scraper.py               # Data collection from CFBD API
├── prediction_server.py     # Flask API server
├── CFBInterface.html        # Web interface
├── scatter.py               # Q1 score distribution visualization
├── requirements.txt         # Dependencies
├── .env                     # Configuration (create this)
└── README.md               # Documentation
```

## Technical Details

### Model Training

For each score, we minimize regularized negative log-likelihood:

```python
L = -Σ[y×log(p) + (1-y)×log(1-p)] 
    + α×[0.3×|β-β₀| + 0.7×(β-β₀)²]
```

Where:
- `y` = binary indicator (1 if this score, 0 otherwise)
- `p` = sigmoid(β₀ + β₁×feature_total + β₂×feature_margin)
- `β₀` = prior logit from empirical frequency
- `α` = adaptive penalty strength

**Logical constraints** on coefficients prevent nonsensical predictions:
- 0-0 score: negative total coefficient (less likely with high totals)
- Blowout scores: positive margin coefficient
- Underdog wins: negative margin coefficient

### Bucket Categories

Rare scores are grouped into buckets:

| Bucket | Description | Example Scores |
|--------|-------------|----------------|
| `other_fav_blowout` | Favored team wins by 7+ | 14-0, 21-7, 28-14 |
| `other_fav_close` | Favored team wins by 1-6 | 7-3, 10-7, 14-10 |
| `other_dog_win` | Underdog team wins | 0-7, 3-10, 7-14 |
| `other_tie` | Tied scores | 3-3, 14-14, 17-17 |

### Optimization

**L-BFGS-B algorithm** with bounded parameter search:
- Typical convergence: 20-50 iterations per score
- Bounds prevent extreme coefficients
- Robust to local minima via regularization

### Probability Distribution Properties

The final distribution is:
- **Valid:** All probabilities ≥0, sum to 1
- **Smooth:** Similar inputs produce similar outputs
- **Calibrated:** Matches empirical Q1 scoring rates
- **Interpretable:** Coefficients have logical signs

## Future Development

1. **First possession data** - Adjust Q1/Q3 based on team receiving/deferring
2. **Pace and tempo adjustments** - Account for offensive/defensive play style as not all games with -2.5 spread and 50.5 totals are the same 

## Contributing

Questions, suggestions, and bug reports are appreciated 

**Contact:** lysek.jarrett@gmail.com  
**GitHub:** [github.com/jlysek/CFB_Q1](https://github.com/jlysek/CFB_Q1)
