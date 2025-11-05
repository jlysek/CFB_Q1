# Statistical Methodology

This document explains the statistical methods used in the College Football Quarter 1 Score Prediction Model. The approach uses orthogonal feature decomposition and two-tier logistic regression to predict Q1 score probability distributions.

## Table of Contents

- [Overview](#overview)
- [Understanding the Problem](#understanding-the-problem)
- [Feature Engineering](#feature-engineering)
- [Two-Tier Model Architecture](#two-tier-model-architecture)
- [Empirical Q1 Calibration](#empirical-q1-calibration)
- [Training Process](#training-process)
- [Making Predictions](#making-predictions)
- [Worked Example](#worked-example)
- [Key Takeaways](#key-takeaways)

---

## Overview

**Goal:** Given pregame spread and total, predict the probability distribution over all possible Q1 scores.

**Input:**
- Pregame spread (negative = home favored)
- Pregame total (expected full game points)

**Output:**
- Probability for every possible Q1 score (0-0, 7-0, 7-7, 14-0, etc.)
- Valid probability distribution (all probabilities ≥ 0, sum to 1)

**Method:** Independent binary logistic regression with orthogonal features and adaptive regularization.

### Why Orthogonal Features?

Traditional approach uses spread and total directly as features:
```
Problem: spread and total are CORRELATED
- High total games often have smaller spreads (both teams score)
- Blowout games often have lower totals (one team dominates)
- Correlation ≈ -0.3 to -0.4 in CFB data
```

This correlation causes **multicollinearity**, leading to:
- Unstable coefficient estimates
- Difficult interpretation (effects are entangled)
- Poor generalization to new data

**Solution:** Transform features to eliminate correlation.

---

## Understanding the Problem

### Probability Distributions

A probability distribution assigns a probability to every possible outcome. All probabilities must sum to 1.

**Example: Rolling a die**

| Outcome | Probability |
|---------|-------------|
| 1 | 1/6 ≈ 16.7% |
| 2 | 1/6 ≈ 16.7% |
| 3 | 1/6 ≈ 16.7% |
| 4 | 1/6 ≈ 16.7% |
| 5 | 1/6 ≈ 16.7% |
| 6 | 1/6 ≈ 16.7% |
| **Sum** | **1.0 (100%)** |

### Our Application: Q1 Scores

From 7000+ games, empirical distribution:

| Score | Probability |
|-------|-------------|
| 7-0 | 11.0% |
| 7-7 | 7.6% |
| 0-7 | 7.1% |
| 0-0 | 7.1% |
| 14-0 | 6.6% |
| ... | ... |
| **Sum** | **100%** |

### Conditional Probability

We want: `P(Q1 Score | Spread, Total)`

"What is the probability of each Q1 score GIVEN the pregame betting lines?"

**Example:**
- Given spread = -7.5, total = 58.5
- What is P(7-0)? P(7-7)? P(14-0)?

---

## Feature Engineering

The key innovation is transforming correlated inputs into orthogonal (uncorrelated) features.

### Step 1: Decompose Spread and Total

The spread and total implicitly tell us each team's expected points:

```
Spread = Expected home points - Expected away points
Total = Expected home points + Expected away points

Solve for individual team expectations:
  Favorite expected points = (Total + |Spread|) / 2
  Underdog expected points = (Total - |Spread|) / 2
```

**Example:**
```
Spread = -7.5 (home favored by 7.5)
Total = 58.5

Favorite (home) expected: (58.5 + 7.5) / 2 = 33.0 points
Underdog (away) expected: (58.5 - 7.5) / 2 = 26.0 points
```

### Step 2: Normalize by Typical Team Score

Normalize to make features scale-invariant:

```python
typical_team_score = 27.5  # Average team score in CFB (learned from data)

norm_fav = implied_fav_points / typical_team_score
norm_dog = implied_dog_points / typical_team_score
```

**Example continued:**
```
norm_fav = 33.0 / 27.5 = 1.200
norm_dog = 26.0 / 27.5 = 0.945
```

### Step 3: Create Orthogonal Features

Transform to orthogonal basis:

```python
feature_total = norm_fav + norm_dog   # Sum captures overall scoring level
feature_margin = norm_fav - norm_dog  # Difference captures expected margin
```

**Example completed:**
```
feature_total = 1.200 + 0.945 = 2.145  (high-scoring game)
feature_margin = 1.200 - 0.945 = 0.255  (moderate favorite)
```

### Why This Works

**Mathematical property:**
```
Correlation(feature_total, feature_margin) ≈ 0
```

By construction, these features are uncorrelated when spread and total vary independently.

**Interpretation:**
- `feature_total`: How much scoring to expect (drives total points)
- `feature_margin`: How lopsided the scoring will be (drives margin)

**Visual intuition:**
```
High total, small spread     → High feature_total, low feature_margin
High total, large spread     → High feature_total, high feature_margin  
Low total, small spread      → Low feature_total, low feature_margin
Low total, large spread      → Low feature_total, high feature_margin
```

The features cleanly separate "how much" from "how lopsided".

---

## Two-Tier Model Architecture

Not all scores have enough data for reliable individual models. We use a two-tier approach:

### Tier 1: Common Scores (≥10 occurrences)

Each score gets its own logistic regression model:

```
logit(score) = β₀ + β₁×feature_total + β₂×feature_margin
```

**Coefficients have logical constraints:**

| Score Type | β₁ (total) | β₂ (margin) | Logic |
|------------|-----------|-------------|-------|
| 0-0 | Negative | Near zero | Less likely with high scoring |
| 7-0 (fav) | Positive | Positive | More likely with scoring + favored |
| 14-0 (fav blowout) | Positive | Strong positive | Much more likely with large margin |
| 7-7 (tie) | Positive | Negative | More likely with scoring, less with margin |
| 0-7 (dog win) | Small | Negative | More likely when underdog strong |

These constraints are enforced via bounded optimization.

### Tier 2: Rare Scores (<10 occurrences)

Scores with insufficient data are grouped into buckets:

**Bucket categories:**

| Bucket | Description | Example Scores |
|--------|-------------|----------------|
| `other_fav_blowout` | Favored team wins by 7+ | 14-0, 21-7, 28-14, 21-0 |
| `other_fav_close` | Favored team wins by 1-6 | 10-7, 7-3, 14-10, 17-14 |
| `other_dog_win` | Underdog team wins | 0-7, 3-10, 7-14, 0-14 |
| `other_tie` | Tied scores | 3-3, 14-14, 17-17, 10-10 |

**Bucket model prediction:**

```
1. Train logistic regression for entire bucket
   P(any score in bucket) = sigmoid(β₀ + β₁×feature_total + β₂×feature_margin)

2. Distribute bucket probability among scores proportionally
   P(specific score) = P(bucket) × (empirical proportion of score in bucket)
```

**Example:**
```
Bucket "other_tie" has 15 rare tied scores
Empirical proportions: 3-3 → 40%, 14-14 → 30%, 17-17 → 20%, 10-10 → 10%

If P(other_tie bucket) = 0.05:
  P(3-3) = 0.05 × 0.40 = 0.020 (2.0%)
  P(14-14) = 0.05 × 0.30 = 0.015 (1.5%)
  P(17-17) = 0.05 × 0.20 = 0.010 (1.0%)
  P(10-10) = 0.05 × 0.10 = 0.005 (0.5%)
```

This allows rare scores to borrow information from similar scores while maintaining individual identities.

---

## Empirical Q1 Calibration

The model automatically learns Q1 scoring rates from historical data rather than assuming fixed percentages.

### Q1 as Percentage of Full Game

**Points percentage:**
```python
Q1_points = (Q1_home + Q1_away) / (Game_home + Game_away)
Empirical average: 23.5%
```

The intercept term in each model implicitly learns this ~23.5% factor.

**Margin percentage:**
```python
Q1_margin_pct = |Q1_margin| / |Game_margin|
```

This percentage varies by spread magnitude and is learned via **cubic spline interpolation**:

| Game Spread | Q1 Margin % | Games |
|-------------|-------------|-------|
| 3 | 24.1% | ~800 |
| 7 | 23.8% | ~1200 |
| 10 | 22.9% | ~900 |
| 14 | 22.3% | ~700 |
| 21 | 21.1% | ~400 |
| 28+ | 19.8% | ~200 |

**Pattern:** Closer games see proportionally more of the margin manifest in Q1. Blowouts spread margin more evenly across quarters.

The model incorporates this via a smooth interpolator rather than discrete buckets.

---

## Training Process

### Data Preparation

**1. Standardize to favored-underdog orientation:**

All scores are oriented relative to the favored team for modeling:
```
If home favored (spread < 0):
  favored_score = home_score
  underdog_score = away_score
  
If away favored (spread > 0):
  favored_score = away_score
  underdog_score = home_score
```

This allows the model to learn patterns like "favored team scores 14, underdog scores 0" rather than learning separate patterns for "home 14, away 0" and "home 0, away 14" when the favorite changes.

**2. Compute orthogonal features for all games:**

```python
for each game:
    abs_spread = |pregame_spread|
    total = pregame_total
    
    implied_fav = (total + abs_spread) / 2
    implied_dog = (total - abs_spread) / 2
    
    norm_fav = implied_fav / typical_team_score
    norm_dog = implied_dog / typical_team_score
    
    feature_total = norm_fav + norm_dog
    feature_margin = norm_fav - norm_dog
```

### Model Fitting

**For each common score:**

Minimize regularized negative log-likelihood:

```
Loss = -Σ[y×log(p) + (1-y)×log(1-p)] + Penalty

where:
  y = 1 if game had this score, 0 otherwise
  p = sigmoid(β₀ + β₁×feature_total + β₂×feature_margin)
```

**Elastic Net regularization:**

```python
α = 0.1 × (1 + 1/max(n_occurrences, 5))  # Adaptive strength
penalty = α × [0.3×|β - β_prior| + 0.7×(β - β_prior)²]
         = α × [L1_component + L2_component]
```

**Why Elastic Net?**
- **L1 (Lasso):** Encourages sparse solutions, can zero out features
- **L2 (Ridge):** Shrinks coefficients smoothly toward prior
- **Combination:** Gets benefits of both

**Prior specification:**

```python
β₀_prior = logit(empirical_frequency)
β₁_prior = 0  # No prior belief about total effect
β₂_prior = 0  # No prior belief about margin effect
```

**Optimization:**

Uses L-BFGS-B algorithm with bounded search:
- Intercept bounds: `[prior_logit - 5, prior_logit + 5]`
- Coefficient bounds: Set based on score type (see logical constraints above)
- Typical convergence: 20-50 iterations

**For each rare score bucket:**

Same process but:
- Binary target: 1 if score in bucket, 0 otherwise
- Larger sample size → more stable fit
- Distributes probability among bucket members

### Feature Selection

After fitting, the model reports which features were effectively dropped:

```
Feature 'total': Dropped in 15/120 models (12.5%)
Feature 'margin': Dropped in 8/120 models (6.7%)
```

A feature is "dropped" when Elastic Net shrinks its coefficient below 0.01.

---

## Making Predictions

### Prediction Pipeline

**Given:** New game with spread = -7.5, total = 58.5

**Step 1: Compute orthogonal features**

```python
abs_spread = 7.5
implied_fav = (58.5 + 7.5) / 2 = 33.0
implied_dog = (58.5 - 7.5) / 2 = 26.0

norm_fav = 33.0 / 27.5 = 1.200
norm_dog = 26.0 / 27.5 = 0.945

feature_total = 2.145
feature_margin = 0.255
```

**Step 2: Calculate logit for each score**

For each score with a trained model:
```python
logit = β₀ + β₁×2.145 + β₂×0.255
```

Convert to raw probability:
```python
raw_prob = 1 / (1 + exp(-logit))
```

**Step 3: Handle orientation**

Since models are trained in favored-underdog orientation:

```python
if home_favored:
    P(home=7, away=0) = P_model(fav=7, dog=0)
    P(home=0, away=7) = P_model(fav=0, dog=7)
else:  # away favored
    P(home=7, away=0) = P_model(fav=0, dog=7)
    P(home=0, away=7) = P_model(fav=7, dog=0)
```

**Step 4: Normalize to valid distribution**

```python
total = Σ raw_prob(all scores)
for each score:
    final_prob(score) = raw_prob(score) / total
```

This ensures probabilities sum to exactly 1.

### Output Format

```
Rank  Score     Probability  Percentage
1     7-0       0.092834     9.28%
2     7-7       0.068145     6.81%
3     14-0      0.054287     5.43%
4     0-0       0.041502     4.15%
5     3-0       0.039871     3.99%
...
```

---

## Worked Example

### Setup

**Game:** Alabama (-10.5) vs Auburn, Total 54.5

**Historical data:**
- Total games: 7042
- Score 14-0 occurred 460 times (6.53%)
- Score 7-7 occurred 528 times (7.50%)

### Step 1: Compute Features

```
abs_spread = 10.5
implied_fav = (54.5 + 10.5) / 2 = 32.5
implied_dog = (54.5 - 10.5) / 2 = 22.0

norm_fav = 32.5 / 27.5 = 1.182
norm_dog = 22.0 / 27.5 = 0.800

feature_total = 1.182 + 0.800 = 1.982
feature_margin = 1.182 - 0.800 = 0.382
```

**Interpretation:**
- Total of 1.982 → Moderate scoring game (typical is ~2.0)
- Margin of 0.382 → Solid favorite (positive margin expected)

### Step 2: Score 14-0 (Favored Blowout)

**Empirical probability:**
```
P(14-0) = 460 / 7042 = 0.0653
logit_prior = log(0.0653 / 0.9347) = -2.65
```

**Trained model:**
```
β₀ = -2.60   (intercept, slightly above prior)
β₁ = +0.83   (total effect: more likely with scoring)
β₂ = +1.12   (margin effect: much more likely with large margin)
```

**Calculate logit:**
```
logit = -2.60 + 0.83×1.982 + 1.12×0.382
      = -2.60 + 1.645 + 0.428
      = -0.527
```

**Convert to probability:**
```
raw_prob = 1 / (1 + exp(0.527))
         = 1 / 1.694
         = 0.371 (37.1% before normalization)
```

**After normalizing all scores:**
```
P(14-0) = 0.089 (8.9%)
```

### Step 3: Score 7-7 (Tie)

**Empirical probability:**
```
P(7-7) = 528 / 7042 = 0.0750
logit_prior = log(0.0750 / 0.9250) = -2.51
```

**Trained model:**
```
β₀ = -2.48   (close to prior)
β₁ = +0.31   (total effect: more likely with scoring)
β₂ = -0.52   (margin effect: less likely with large margin)
```

**Calculate logit:**
```
logit = -2.48 + 0.31×1.982 + (-0.52)×0.382
      = -2.48 + 0.614 - 0.199
      = -2.065
```

**Convert to probability:**
```
raw_prob = 1 / (1 + exp(2.065))
         = 1 / 8.88
         = 0.113 (11.3% before normalization)
```

**After normalizing all scores:**
```
P(7-7) = 0.064 (6.4%)
```

### Step 4: Interpretation

| Score | Empirical | Model | Change | Why |
|-------|-----------|-------|--------|-----|
| 14-0 | 6.53% | 8.90% | +2.37% | Large favorite makes blowout more likely |
| 7-7 | 7.50% | 6.40% | -1.10% | Large spread makes tie less likely |

**Feature effects:**

For 14-0:
- `feature_total` (1.982) → +1.65 logit points (moderate scoring helps)
- `feature_margin` (0.382) → +0.43 logit points (margin helps significantly)
- **Net effect:** Probability increases substantially

For 7-7:
- `feature_total` (1.982) → +0.61 logit points (scoring helps ties too)
- `feature_margin` (0.382) → -0.20 logit points (margin hurts ties)
- **Net effect:** Margin effect dominates, probability decreases

### Step 5: Full Distribution

Top 10 predictions for this game:

| Rank | Score | Probability | Interpretation |
|------|-------|-------------|----------------|
| 1 | 7-0 | 10.2% | Most common score, favorite scoring first |
| 2 | 14-0 | 8.9% | Large margin increases blowout probability |
| 3 | 7-7 | 6.4% | Moderate scoring allows tie despite margin |
| 4 | 10-0 | 5.1% | Field goal-touchdown combo |
| 5 | 0-0 | 4.8% | Defensive start |
| 6 | 3-0 | 4.2% | Field goal only |
| 7 | 14-7 | 3.9% | Both teams score, favorite ahead |
| 8 | 0-7 | 3.1% | Underdog wins Q1 (less likely given margin) |
| 9 | 21-0 | 2.8% | Extreme blowout start |
| 10 | 7-3 | 2.6% | Close game despite spread |

---

## Key Takeaways

### Why Orthogonal Features?

**Problem with raw features:**
```
spread and total are correlated → unstable coefficients
Hard to interpret: "Does total affect this score, or is it just correlated spread?"
```

**Solution with orthogonal features:**
```
feature_total and feature_margin are uncorrelated → stable coefficients
Clear interpretation:
  - feature_total controls overall scoring level
  - feature_margin controls expected margin
```

### Why Two-Tier Architecture?

**Common scores:** Enough data for individual models → precise predictions

**Rare scores:** Not enough data for stable individual models → group into buckets
- Borrow information from similar scores
- Maintain individual score identities via proportional distribution
- Avoid overfitting on sparse data

### Why Elastic Net Regularization?

**L1 component (30%):**
- Performs feature selection
- Can zero out irrelevant features
- Creates sparse models

**L2 component (70%):**
- Smooth shrinkage toward prior
- Prevents extreme coefficients
- Better for correlated features (though ours aren't!)

**Adaptive strength:**
- Rare scores → strong regularization → stay near baseline
- Common scores → weak regularization → learn from data

### Model Properties

**Interpretability:**
- Each coefficient has clear meaning
- Logical constraints ensure sensible predictions
- Easy to debug individual scores

**Robustness:**
- Regularization prevents overfitting
- Bucket models handle sparse data
- Normalization ensures valid distribution

**Flexibility:**
- Can add new features easily
- Per-score customization possible
- Extensible to other quarters

### Current Limitations

**Features:**
- No coaching tendency information
- No first possession data
- No weather/injuries/context

**Model:**
- Independence assumption (ignores score correlations)
- Linear effects only (no interactions)
- Only predicts Q1 (not sequential quarters)

**Data:**
- Limited to FBS games
- Requires pregame betting lines
- Historical data may not reflect current trends

---

## Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| `P(A)` | Probability of event A |
| `P(A\|B)` | Probability of A given B |
| `log(x)` | Natural logarithm |
| `exp(x)` | Exponential (e^x) |
| `sigmoid(x)` | 1 / (1 + exp(-x)) |
| `logit(p)` | log(p / (1-p)) |
| `β` | Regression coefficient |
| `α` | Regularization strength |
| `Σ` | Summation |
| `\|x\|` | Absolute value |

---

## Glossary

**Binary Classification:** Predicting one of two outcomes (yes/no, this score/not this score)

**Elastic Net:** Regularization combining L1 (Lasso) and L2 (Ridge) penalties

**Empirical Distribution:** Probability distribution from observed frequencies

**Feature Engineering:** Transforming raw inputs into useful model features

**Logistic Regression:** Model for binary outcomes using logit link function

**Multicollinearity:** High correlation between predictor variables causing unstable estimates

**One-vs-All:** Training separate binary models for each class

**Orthogonal:** Mathematically independent (zero correlation)

**Regularization:** Penalty preventing extreme coefficient values and overfitting

**Sigmoid:** Function mapping any number to probability between 0 and 1

---

*This methodology reflects the current model implementation. Future versions may incorporate additional features, alternative architectures, or improved calibration methods.*