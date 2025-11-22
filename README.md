# College Football Quarter-by-Quarter Score Prediction

A statistical model that predicts quarter-by-quarter scoring probability distributions in college football games using pregame betting lines. Modeling.md will provide a 
more in depth explanation of the modeling used. 

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Database Schema](#database-schema)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)

## Overview

**Input:** Pregame spread and total  
**Output:** Probability distributions for all four quarters, both halves, and the full game

**Example:**
```
Spread: -7.5 (Home favored by 7.5)
Total: 58.5

Quarter 1 Top Predictions:
7-0  → 9.2%
7-7  → 6.8%
14-0 → 5.4%
0-0  → 4.1%

Full Game Top Predictions:
31-24 → 3.1%
28-21 → 2.8%
34-27 → 2.6%
...
```

The model trains on 7000+ FBS games from 2014-present with quarter-by-quarter scoring and pregame betting lines.

## How It Works

### Converting Betting Lines to Features

Betting lines give us two numbers:
- **Spread:** How much better one team is (e.g., -7.5 means favorite wins by 7.5 more than 7.5 ~50% of the time)
- **Total:** How many total points we expect (e.g., ~50% of the time there are more than 58.5 points)

We convert these to what each team is expected to score:
```
If spread = -7.5 and total = 58.5:
  Favorite expected: (58.5 + 7.5) ÷ 2 = 33 points
  Underdog expected: (58.5 - 7.5) ÷ 2 = 26 points
```

Then we create two special features:
- **feature_total:** Measures overall scoring level (high = lots of points)
- **feature_margin:** Measures how lopsided the game is (high = big blowout)

**Why these features?** They're "orthogonal" - meaning they don't overlap. Total tells you about overall scoring, margin tells you about the difference, and they work independently. 

### Predicting Each Quarter

For each quarter, we train separate models for every possible score:
- **Common scores** (appearing 50+ times): Get their own individual model
  - Examples: 7-0, 7-7, 14-0, 0-0, 14-7
  - Each learns how feature_total and feature_margin affect its probability
  
- **Rare scores** (appearing <50 times): Grouped into categories
  - "Big favorite wins" (21-0, 28-7, etc.)
  - "Close games" (10-7, 14-10, etc.)
  - "Underdog wins" (0-7, 3-10, etc.)
  - "Ties" (3-3, 14-14, etc.)

**The model learns logical patterns:**
- Score 0-0 becomes LESS likely when feature_total is high (high scoring games rarely stay 0-0)
- Score 14-0 becomes MORE likely when feature_margin is high (blowouts start early)
- Score 7-7 becomes MORE likely with high feature_total but LESS likely with high feature_margin (ties need scoring but not dominance)

### Step 4: Accounting for Quarter Correlation

This is where our model gets smarter than simple multiplication. Instead of treating quarters independently, we use **conditional probability**:

**P(Q2 | Q1)** = "Probability of Q2 score GIVEN what happened in Q1"

**Example:**
```
If Q1 was 14-0 (high-scoring, favored blowout):
  → Q2 is more likely to be high-scoring too
  → Q2 is more likely to favor the same team
  
If Q1 was 0-0 (defensive struggle):
  → Q2 is more likely to be low-scoring too
  → Both teams likely to struggle
```

We learn these patterns from history by categorizing Q1 outcomes:
- Low scoring (0-7 points), medium (8-20 points), high (21+ points)
- Tie, close game, or blowout
- Favorite ahead or underdog ahead

Then for Q3, we use **P(Q3 | H1)** where H1 = Q1 + Q2, and for Q4 we use **P(Q4 | H1, Q3)**.

### Step 5: Monte Carlo Simulation

To get the full game distribution, we run 5,000 simulations:

```
For each simulation:
  1. Sample Q1 from its distribution
  2. Based on Q1, sample Q2 from P(Q2|Q1)
  3. Based on first half, sample Q3 from P(Q3|H1)
  4. Based on everything, sample Q4 from P(Q4|H1,Q3)
  5. Add up all four quarters
```

This captures how scoring flows through the game naturally.

### Step 6: Calibration

The simulated full game might not match the betting lines exactly (remember, we started with -7.5 spread and 58.5 total). So we **iteratively adjust** the quarter distributions:

**Iteration 1:** Generate full game → Check if it matches spread/total → Adjust quarters slightly  
**Iteration 2:** Generate new full game → Check again → Adjust more  
...  
**Iteration 10:** Full game matches betting lines within 1%

**How do we adjust?** We figure out which scores are "responsible" for being too high or too low:
- If the full game is scoring too many points, we reduce probability of high-scoring quarters
- If the favorite is covering the spread too often, we reduce probability of favorite-dominant quarters
- We adjust all four quarters proportionally

**Example:**
```
Full game is 2% over on total → Need to reduce scoring

Q1 score 14-7 (21 points, high):
  - This score contributes to excess points
  - Reduce its probability by 2% × responsibility factor
  
Q1 score 0-0 (0 points, low):
  - This score helps with the problem
  - Increase its probability slightly
```

After 10 iterations, the full game distribution perfectly matches Vegas while maintaining realistic quarter-by-quarter correlations.

## Database Schema

### `cfb.games`

Every FBS game since 2014 with betting lines and final scores.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | Integer | Unique identifier |
| `season` | Integer | Year |
| `week` | Integer | Week number |
| `home_team` | String | Home team name |
| `away_team` | String | Away team name |
| `home_points` | Integer | Final home score |
| `away_points` | Integer | Final away score |
| `pregame_spread` | Float | Closing spread (negative = home favored) |
| `pregame_total` | Float | Closing total |
| `home_classification` | String | Division (fbs/fcs) |
| `away_classification` | String | Division (fbs/fcs) |
| `betting_provider` | String | Source of betting lines |

### `cfb.quarter_scoring`

Quarter-by-quarter scoring (4 rows per game).

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | Integer | Links to games table |
| `quarter` | Integer | Quarter number (1-4) |
| `home_score` | Integer | Home points this quarter |
| `away_score` | Integer | Away points this quarter |

## Installation

### Prerequisites

- Python 3.8+
- MySQL 8.0+
- College Football Data API key ([collegefootballdata.com](https://collegefootballdata.com))

### Setup

**1. Clone repository**
```bash
git clone https://github.com/yourusername/cfb-quarters.git
cd cfb-quarters
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
python all.py
```

## Usage

### Command Line

```bash
python all.py
```

**Interactive mode:**
```
Enter spread and total: -7.5 58.5
```

The model outputs:
- Top 10 predictions for each quarter
- Top 10 predictions for first half
- Top 10 predictions for full game (calibrated)
- Probabilities account for quarter correlation

**Example output:**
```
Quarter 1 - Top 10:
  7-0        10.23%
  7-7         6.81%
  14-0        5.43%
  0-0         4.15%
  3-0         3.99%
  ...

Full Game (Calibrated with Correlation) - Top 10:
  31-24       3.12%
  28-21       2.84%
  34-27       2.67%
  ...
```

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
# Generate predictions for all quarters
curl -X POST http://localhost:5000/api/predict-all-quarters \
  -H "Content-Type: application/json" \
  -d '{"spread": -7.5, "total": 58.5}'

# Response includes:
# - q1_home_away, q2_home_away, q3_home_away, q4_home_away
# - h1_home_away, h2_home_away
# - full_game_home_away (calibrated)
```

## Project Structure

```
cfb-quarters/
├── all.py                  # All-quarters predictor with correlation
├── scraper.py              # Data collection from CFBD API
├── scraper_incremental.py  # Weekly data updates
├── prediction_server.py    # Flask API server
├── CFBInterface.html       # Web interface
├── requirements.txt        # Dependencies
├── .env                    # Configuration (create this)
├── README.md              # This file
└── Modeling.md            # Detailed methodology
```

## Technical Details

### Model Architecture

**Quarter Models:**
- Individual logistic regression for common scores (≥50 occurrences)
- Bucket models for rare scores (<50 occurrences)
- Coefficients bounded by logical constraints

**Correlation Modeling:**
- Conditional probabilities learned from historical data
- P(Q2|Q1), P(Q3|H1), P(Q4|H1,Q3)
- Weighted blend: 70% base model + 30% conditional adjustment

**Calibration:**
- Iterative gradient descent on quarter distributions
- Target: 49-51% on spread and total
- Learning rate: 0.12, clamped updates: ±6% per iteration
- Convergence: 10 iterations or 1% tolerance

### Adaptive Anchor Threshold

Scores become "anchors" (individual models) if they appear in:
- At least 0.75% of games, OR
- At least 50 games (statistical stability floor)

Example with 7,000 games:
- 0.75% threshold = 52.5 games
- Selected threshold: max(50, 52.5) = 52.5 games

### Optimization

**L-BFGS-B algorithm** with:
- Bounded parameter search
- L2 regularization (0.01 weight)
- Convergence: typically 20-50 iterations per score

### Probability Distribution Properties

The final distributions are:
- **Valid:** All probabilities ≥0, sum to 1
- **Smooth:** Similar inputs produce similar outputs
- **Calibrated:** Match Vegas spread and total exactly
- **Correlated:** Capture quarter-to-quarter dependencies
- **Interpretable:** Coefficients have logical signs

## Future Development

1. **First possession data** - Adjust Q1/Q3 based on team receiving/deferring
2. **Pace and tempo adjustments** - Account for offensive/defensive play style
3. **Weather conditions** - Wind, rain, temperature effects
4. **Coaching tendencies** - Fourth down decisions, risk tolerance
5. **Conference championship simulations** - Use quarter models for season projections

## Contributing

Questions, suggestions, and bug reports are appreciated!

**Contact:** lysek.jarrett@gmail.com  
**GitHub:** [github.com/jlysek](https://github.com/jlysek)
