# Statistical Methodology

This document breaks down the statistical methods used in the College Football Quarter Scoring Prediction Model.

---

## Table of Contents

- [Overview](#overview)
- [Understanding Probability Distributions](#understanding-probability-distributions)
  - [Empirical vs Model-Based Distributions](#empirical-vs-model-based-distributions)
- [Building the Empirical Distribution](#building-the-empirical-distribution)
- [Independent Binary Logistic Regression](#independent-binary-logistic-regression)
  - [Part 1: Understanding Logistic Functions](#part-1-understanding-logistic-functions)
  - [Part 2: Simple Logistic Regression](#part-2-simple-logistic-regression)
  - [Part 3: One-vs-All Binary Classification](#part-3-one-vs-all-binary-classification)
  - [Part 4: Making Predictions](#part-4-making-predictions)
  - [Part 5: Training the Model](#part-5-training-the-model)
  - [Part 6: Feature Normalization](#part-6-feature-normalization)
- [Regularization and Overfitting](#regularization-and-overfitting)
- [Market Calibration Layer](#market-calibration-layer)
- [Distance-Based Adjustment for Rare Scores](#distance-based-adjustment-for-rare-scores)
- [Putting It All Together](#putting-it-all-together)
- [Worked Example](#worked-example)

---

## Overview

**Goal:** Given pregame closing betting markets (spread and total), what is the probability of every possible first quarter score?

### Example

- **Spread:** Home team favored by 7 points (spread = -7)
- **Total:** 58.5 points expected in the full game
- **Question:** What's the probability Q1 ends 7-7? 14-7? 0-0? 7-3?

### Approach: Independent Binary Logistic Regression with Market Calibration

We use a **one-vs-all** approach where each score gets its own independent binary logistic regression model. Each model predicts `P(this_score | Spread, Total)` independently. The models are then normalized and calibrated against market data.

The pipeline consists of three layers:

1. **Model Layer:** Independent binary logistic regression for each score
2. **Normalization Layer:** Ensure probabilities sum to 1
3. **Calibration Layer:** Blend model predictions with market data

This approach allows for flexible per-score regularization and handles 300+ possible scores efficiently.

---

## Understanding Probability Distributions

A probability distribution assigns a probability to every possible outcome. Probabilities must sum to exactly 1 (or 100%).

### Simple Example - Rolling a Die

| Outcome | Probability |
|---------|-------------|
| 1 | 1/6 ≈ 0.167 |
| 2 | 1/6 ≈ 0.167 |
| 3 | 1/6 ≈ 0.167 |
| 4 | 1/6 ≈ 0.167 |
| 5 | 1/6 ≈ 0.167 |
| 6 | 1/6 ≈ 0.167 |
| **Total** | **1.0 (100%)** |

### Conditional Probability Example

`P(x > 4 | x is even)` = "Probability x is greater than 4 GIVEN x is even"

Since x is even, possible outcomes shrink to: `{2, 4, 6}`

`P(x > 4 | x is even) = P(6) / P(2,4,6) = (1/6) / (3/6) = 1/3`

> The "|" symbol means "given" in probability notation

### Our Application - Q1 Scores

| Score | Probability |
|-------|-------------|
| 0-0 | 0.071 |
| 7-0 | 0.110 |
| 0-7 | 0.071 |
| 7-7 | 0.076 |
| 14-0 | 0.066 |
| ... | ... |
| **Total** | **1.0 (100%)** |

---

## Empirical vs Model-Based Distributions

### Empirical Distribution: Built from real data
- Count how often each outcome happened historically
- Turn counts into percentages
- Represents unconditional baseline probabilities

### Model-Based Distribution: Conditional on game features
- Use statistical models to adjust probabilities based on game characteristics
- Incorporates pregame betting information (spread and total)
- Represents conditional probabilities: `P(score | spread, total)`

**We use both:** Start with empirical frequencies as a baseline, then train models that adjust predictions based on game-specific information.

---

## Building the Empirical Distribution

### Step 1: Collect Historical Data

Every FBS college football game since 2014 with:
- Quarter-by-quarter scoring
- Pregame closing spread and total
- Team classifications

### Step 2: Count Score Occurrences

From actual model output (approximately 7,000 games):

| Score | Count | Probability |
|-------|-------|-------------|
| 7-0 | 765 | 0.110 (11.0%) |
| 7-7 | 528 | 0.076 (7.6%) |
| 0-7 | 490 | 0.071 (7.1%) |
| 0-0 | 490 | 0.071 (7.1%) |
| 14-0 | 460 | 0.066 (6.6%) |
| 3-0 | 320 | 0.046 (4.6%) |
| ... | ... | ... |

### Step 3: Use as Foundation

This empirical distribution serves three purposes:

1. **Baseline:** Starting point for all predictions
2. **Regularization anchor:** Prevents wild predictions for rare scores
3. **Training target:** Models learn to adjust from this baseline

---

## Independent Binary Logistic Regression

This is the core of the prediction model. Unlike traditional multinomial logistic regression with softmax, we use a **one-vs-all** approach with independent binary models.

### Part 1: Understanding Logistic Functions

#### From Probability to Logit

Regression works best with numbers ranging from -∞ to +∞. Probabilities are constrained between 0 and 1.

We transform probabilities using the **logit function**:

```
logit(p) = log(p / (1-p))
```

**What this does:**

| Probability | Logit Value | Interpretation |
|-------------|-------------|----------------|
| 0.01 (1%) | -4.6 | Very unlikely |
| 0.10 (10%) | -2.2 | Unlikely |
| 0.50 (50%) | 0.0 | Even odds |
| 0.90 (90%) | +2.2 | Likely |
| 0.99 (99%) | +4.6 | Very likely |

Now we can work with unbounded numbers, which regression handles well.

#### From Logit Back to Probability

The inverse operation (sigmoid function):

```
p = 1 / (1 + exp(-logit))
```

This maps any number back to a probability between 0 and 1.

---

### Part 2: Simple Logistic Regression

Before understanding one-vs-all, consider regular binary logistic regression.

**Example Goal:** Predict if home team wins based on spread.

```
logit(p_win) = β₀ + β₁ × spread
```

**What coefficients mean:**
- `β₀` (intercept): Baseline log-odds when spread is 0
- `β₁` (slope): How spread affects win probability

**Example with learned values:**

```
logit(p_win) = 0.0 + 0.3 × spread
```

Note: Spread is negative when home is favored, positive when away is favored.

| Scenario | Calculation | Result |
|----------|-------------|--------|
| Spread = -7 (home favored) | logit = -2.1 | p_win = 0.89 (89%) |
| Spread = 0 (pick'em) | logit = 0.0 | p_win = 0.50 (50%) |
| Spread = +7 (home underdog) | logit = +2.1 | p_win = 0.11 (11%) |

---

### Part 3: One-vs-All Binary Classification

**One-vs-all** means we train a separate binary classifier for each possible Q1 score. Each model answers: "Given the spread and total, is THIS specific score likely?"

#### The Setup

For each of 300+ possible scores, we train an independent model:

```
logit(score) = β₀ + β₁ × spread + β₂ × total + β₃ × spread² + β₄ × total²
```

**5 coefficients per score:**

1. `β₀`: Baseline log-odds for this score at neutral game conditions
2. `β₁`: Spread effect - How spread impacts this score's likelihood
3. `β₂`: Total effect - How total impacts this score's likelihood
4. `β₃`: Non-linear spread effect - Captures diminishing or accelerating spread effects
5. `β₄`: Non-linear total effect - Captures non-linear total effects

#### Training Data Structure

For each score, we create a binary classification problem:

**Example: Training the "7-0" model**

| Game | Spread | Total | Actual Score | Binary Target |
|------|--------|-------|--------------|---------------|
| 1 | -7.5 | 58.5 | 7-0 | 1 |
| 2 | -3.0 | 52.0 | 3-0 | 0 |
| 3 | -10.5 | 61.0 | 7-0 | 1 |
| 4 | +2.5 | 48.5 | 0-7 | 0 |
| ... | ... | ... | ... | ... |

#### Why Quadratic Terms?

Relationships aren't always linear. Consider:

- **Spread effect on 14-0 scores:**
  - Spread -3 → -6: Moderate increase in 14-0 probability
  - Spread -28 → -31: Minimal additional increase (already very likely blowout)

- **Total effect on 7-7 scores:**
  - Total 45 → 55: Significant increase in 7-7 probability (more scoring)
  - Total 75 → 85: Diminishing effect (score becomes less balanced at extremes)

Quadratic terms capture these non-linear relationships.

---

### Part 4: Making Predictions

When predicting for a new game, each score's independent model produces a raw probability estimate.

**Step 1: Calculate logit for each score using its trained model**

```
logit_7-0 = β₀₁ + β₁₁ × spread + β₂₁ × total + β₃₁ × spread² + β₄₁ × total²
logit_7-7 = β₀₂ + β₁₂ × spread + β₂₂ × total + β₃₂ × spread² + β₄₂ × total²
logit_0-7 = β₀₃ + β₁₃ × spread + β₂₃ × total + β₃₃ × spread² + β₄₃ × total²
...
```

**Step 2: Convert each logit to raw probability using sigmoid**

```
raw_prob(7-0) = 1 / (1 + exp(-logit_7-0))
raw_prob(7-7) = 1 / (1 + exp(-logit_7-7))
raw_prob(0-7) = 1 / (1 + exp(-logit_0-7))
...
```

**Step 3: Normalize probabilities to sum to 1**

Since each model was trained independently, raw probabilities don't automatically sum to 1:

```
p(score_i) = raw_prob(score_i) / Σ(raw_prob(j) for all j)
```

This normalization ensures we have a valid probability distribution.

#### One-vs-All vs Softmax

**Our approach (One-vs-All):**
- Each score has its own independent binary model
- Models trained separately with individual regularization
- Raw probabilities normalized after prediction
- Allows flexible per-score parameters
- Efficient for 300+ classes

**Alternative approach (Softmax/True Multinomial):**
- All scores compete in a single joint model
- Probabilities naturally sum to 1 via softmax
- All classes trained simultaneously
- Single set of hyperparameters for all classes
- Computationally expensive for many classes

We use one-vs-all because:
1. Flexible regularization per score (adaptive to frequency)
2. More stable optimization for 300+ scores
3. Easier to debug and interpret individual score models
4. Can handle rare scores better with adaptive penalties

---

### Part 5: Training the Model

**Training objective:** Find β coefficients that best predict historical scores while avoiding overfitting.

#### Loss Function

For each score's binary model:

```
Loss = -LogLikelihood + L2_Regularization

Where:
LogLikelihood = Σ[y_i × log(p_i) + (1 - y_i) × log(1 - p_i)]
L2_Regularization = λ × Σ(β_j - β_prior_j)²
```

**Components:**

- `-LogLikelihood`: How well the model fits training data (lower is better)
- `λ`: Regularization strength (higher = more penalty for deviation)
- `(β - β_prior)²`: Squared distance from prior coefficients

#### Prior Coefficients

The prior is derived from the empirical distribution:

```
β₀_prior = logit(empirical_probability)
β₁_prior = 0  (no prior belief about spread effect)
β₂_prior = 0  (no prior belief about total effect)
β₃_prior = 0  (no prior belief about quadratic effects)
β₄_prior = 0  (no prior belief about quadratic effects)
```

#### Optimization Algorithm

We use L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds):

1. **Gradient-based optimization:** Efficiently finds coefficients that minimize loss
2. **Bounded search:** Constrains coefficients to reasonable ranges
3. **Fallback to Nelder-Mead:** If L-BFGS-B fails, use gradient-free method

**Coefficient bounds:**

```
β₀: [prior_logit - 5, prior_logit + 5]  (intercept can vary moderately)
β₁: [-2, +2]  (spread effect bounded)
β₂: [-2, +2]  (total effect bounded)
β₃: [-1, +1]  (quadratic effects smaller)
β₄: [-1, +1]  (quadratic effects smaller)
```

#### What the Model Learns

**Example learned patterns:**

- **7-0 scores:** Positive β₁ coefficient (more likely when home is favored)
- **0-7 scores:** Negative β₁ coefficient (more likely when away is favored)
- **14-0 scores:** Larger positive β₁ (even more sensitive to spread)
- **7-7 scores:** Near-zero β₁ (happens regardless of favorite)
- **High-scoring ties:** Positive β₂ (more likely in high-total games)

---

### Part 6: Feature Normalization

Before regression, we normalize inputs to put them on comparable scales:

```
normalized_spread = spread / 10.0
normalized_total = (total - 50) / 20.0
```

**Why normalize?**

- **Raw spread range:** -70 to +70 (but typically -35 to +35)
- **Raw total range:** 30 to 90 (but typically 40 to 70)
- Without normalization, larger-magnitude features dominate
- Normalization ensures both features contribute proportionally

**Example:**

| Feature | Raw | Normalization | Normalized |
|---------|-----|---------------|------------|
| Spread | -14.0 | / 10.0 | -1.40 |
| Total | 58.0 | (x - 50) / 20.0 | +0.40 |
| Spread² | 196.0 | (-1.40)² | 1.96 |
| Total² | 3364.0 | (0.40)² | 0.16 |

After normalization, both features typically range from -3 to +3.

---

## Regularization and Overfitting

### The Overfitting Problem

**Scenario:** Only 5 games in history ended Q1 at 24-17.

**Without regularization, the model might learn:**
- "24-17 only happens when total is exactly 67.5"
- "24-17 requires spread between -8.5 and -9.5"

**Problem:** This is likely random noise, not a real pattern. With only 5 examples, we can't reliably distinguish signal from noise.

**Result:** Wild, overconfident predictions that don't generalize.

---

### The Solution: Adaptive L2 Regularization

Add a penalty for coefficients that deviate from sensible priors:

```
Loss = -LogLikelihood + λ × Σ(β_j - β_prior_j)²
```

**How it works:**

- Pulls coefficients toward prior values (empirical baseline)
- Stronger pull for rare scores (high λ)
- Weaker pull for common scores (low λ)
- Prevents overfitting while allowing data-driven adjustments

### Adaptive Regularization Strength

Regularization strength varies by score frequency:

```
λ(score) = base_λ × (1 + 1 / max(count(score), 5))
```

Where:
- `base_λ = 0.1` (baseline regularization)
- `count(score)` = number of times this score appeared
- Minimum count of 5 prevents extreme penalties

**Example:**

| Score | Count | Calculation | λ Value | Strength |
|-------|-------|-------------|---------|----------|
| 7-0 | 765 | 0.1 × (1 + 1/765) | 0.100 | Very weak |
| 14-14 | 45 | 0.1 × (1 + 1/45) | 0.102 | Weak |
| 21-17 | 8 | 0.1 × (1 + 1/8) | 0.113 | Moderate |
| 24-17 | 5 | 0.1 × (1 + 1/5) | 0.120 | Strong |

### Effect of Regularization

**For common scores (weak regularization):**
- Model learns mostly from data
- Can deviate significantly from empirical baseline
- Captures true conditional patterns

**For rare scores (strong regularization):**
- Model stays closer to empirical baseline
- Limited deviation based on sparse data
- Prevents overfitting to noise

**Example: 7-0 score in a heavily favored home game**

```
Empirical: 11.0%
Model raw: 18.5% (learns home is heavily favored)
Regularized: 17.2% (allows adjustment but not too extreme)
```

**Example: 24-17 score (rare)**

```
Empirical: 0.08%
Model raw: 2.1% (might be noise from 5 games)
Regularized: 0.15% (stays closer to baseline)
```

---

## Market Calibration Layer

After the model produces predictions, we apply a calibration layer that blends model estimates with market probabilities derived from actual betting lines.

### Why Calibration?

Statistical models have systematic biases:
- May underestimate or overestimate certain scores
- May not capture all information in betting markets
- Markets aggregate information from many sources

**Solution:** Use market data (when available) to calibrate model predictions.

### Calibration Data Structure

Market probabilities are stored in CSV files named `<spread>_<total>.csv`:

**Example: `7.5_45.5.csv`**

| Home Score | Away Score | Fair |
|------------|------------|------|
| 0 | 0 | 0.0821 |
| 7 | 0 | 0.1243 |
| 0 | 7 | 0.0532 |
| 7 | 7 | 0.0687 |
| 14 | 0 | 0.0821 |
| ... | ... | ... |

These represent the market's implied probability distribution for that specific spread/total combination.

### 2D Interpolation

Since we can't have calibration data for every possible spread/total combination, we use **inverse distance weighting (IDW)** to interpolate.

**For a score at spread = -9.5, total = 52.0:**

1. Find all calibration files that contain this score
2. Calculate distance in normalized (spread, total) space
3. Weight each calibration point by inverse of distance
4. Compute weighted average

**Distance calculation:**

```
distance = sqrt((spread_diff / 10)² + (total_diff / 20)²)
```

**Inverse distance weighting:**

```
weight_i = 1 / (distance_i² + ε)

calibrated_prob = Σ(weight_i × market_prob_i) / Σ(weight_i)
```

### Blending Model and Market

For each score, we blend the model prediction with the interpolated market probability:

```
final_prob = α × market_prob + (1 - α) × model_prob
```

**Calibration weight (α) is adaptive:**

```
α = max_α × exp(-min_distance / 2.0)

Where:
max_α = 0.7  (maximum trust in market)
min_distance = distance to nearest calibration point with this score
```

**Effect:**

- **Exact match** (distance = 0): α = 0.70 (70% market, 30% model)
- **Distance = 1**: α = 0.42 (42% market, 58% model)
- **Distance = 2**: α = 0.26 (26% market, 74% model)
- **Distance = 5+**: α ≈ 0 (mostly model)

### Example Calibration

**Game: Spread = -7.5, Total = 58.5**
**Score: 7-0**

**Nearest calibration points:**

| File | Distance | Market Prob | Weight |
|------|----------|-------------|--------|
| 7.5_58.5 | 0.00 | 0.135 | ∞ (exact) |

Since we have an exact match:
- α = 0.70
- Model prob = 0.142
- Market prob = 0.135
- Final prob = 0.70 × 0.135 + 0.30 × 0.142 = 0.137

**Game: Spread = -9.5, Total = 56.0**
**Score: 7-7**

**Nearest calibration points:**

| File | Distance | Market Prob | Weight |
|------|----------|-------------|--------|
| 7.5_58.5 | 0.29 | 0.078 | 11.9 |
| 10.5_54.5 | 0.17 | 0.072 | 34.6 |

Interpolated market prob:
```
(11.9 × 0.078 + 34.6 × 0.072) / (11.9 + 34.6) = 0.074
```

Distance to nearest = 0.17, so α = 0.70 × exp(-0.17/2) ≈ 0.64

Model prob = 0.081

Final prob = 0.64 × 0.074 + 0.36 × 0.081 = 0.076

### Calibration Pipeline

```
1. Model produces raw probabilities
2. Normalize to sum to 1
3. For each score:
   a. Interpolate market probability (if available)
   b. Calculate calibration weight based on distance
   c. Blend model and market probabilities
4. Normalize final distribution to sum to 1
```

---

## Distance-Based Adjustment for Rare Scores

For scores that are too rare to have fitted models (occur fewer than ~10 times), we use a distance-based heuristic rather than logistic regression.

### The Approach

For rare scores without fitted models:

1. Start with empirical probability
2. Calculate "expected" Q1 outcome from game lines
3. Measure how far this score is from expectations
4. Apply exponential decay based on distance

### Expected Outcome Calculation

```
expected_margin = -spread
expected_q1_total = total × 0.25
```

**Examples:**

| Spread | Total | Expected Margin | Expected Q1 Total |
|--------|-------|-----------------|-------------------|
| -14.0 | 56.0 | +14.0 | 14.0 |
| +7.5 | 48.0 | -7.5 | 12.0 |
| -3.0 | 62.0 | +3.0 | 15.5 |

### Distance Calculation

For a score like 17-10:

```
actual_margin = 17 - 10 = 7
actual_total = 17 + 10 = 27

margin_distance = |7 - expected_margin|
total_distance = |27 - expected_q1_total|
```

### Exponential Decay Adjustment

```
margin_factor = exp(-margin_distance / 10.0)
total_factor = exp(-total_distance / 5.0)

adjusted_prob = empirical_prob × margin_factor × total_factor
```

**Decay rates:**
- Margin decays with half-life of ~7 points
- Total decays with half-life of ~3.5 points

### Example

**Game: Spread = -10.5, Total = 52.0**
**Score: 17-10 (empirical prob = 0.0012)**

```
Expected margin: +10.5
Expected Q1 total: 13.0

Actual margin: 7
Actual total: 27

Margin distance: |7 - 10.5| = 3.5
Total distance: |27 - 13| = 14.0

Margin factor: exp(-3.5 / 10) = 0.705
Total factor: exp(-14.0 / 5) = 0.061

Adjusted prob: 0.0012 × 0.705 × 0.061 = 0.000052 (0.0052%)
```

This score is far from expected total, so it gets heavily downweighted.

---

## Putting It All Together

### The Complete Prediction Pipeline

#### Step 1: Calculate Empirical Distribution

```
For each Q1 score in historical data:
    empirical_prob[score] = count[score] / total_games
```

#### Step 2: Train Binary Models with Adaptive Regularization

```
For each common Q1 score:
    # Calculate regularization strength
    λ = 0.1 × (1 + 1 / max(count(score), 5))
    
    # Calculate prior
    prior_logit = logit(empirical_prob[score])
    
    # Set up loss function
    Loss = -Σ[y × log(p) + (1-y) × log(1-p)] + λ × Σ(β - β_prior)²
    
    # Optimize to find best coefficients
    β = optimize(Loss)
    
    # Store trained model
    model[score] = β
```

#### Step 3: Load Market Calibration Data

```
For each CSV file in calibration folder:
    # Parse filename to get spread and total
    spread, total = parse_filename(file)
    
    # Load market probabilities
    market_data[(spread, total)] = load_csv(file)
```

#### Step 4: Make Predictions

```
Given: spread = -7.5, total = 58.5

# Phase 1: Model predictions
For each possible score:
    If score has fitted model:
        # Normalize inputs
        norm_spread = spread / 10.0
        norm_total = (total - 50) / 20.0
        
        # Calculate logit
        logit = β₀ + β₁×norm_spread + β₂×norm_total 
                + β₃×norm_spread² + β₄×norm_total²
        
        # Convert to probability
        raw_prob[score] = 1 / (1 + exp(-logit))
    
    Else:  # Rare score without model
        # Use distance-based heuristic
        expected_margin = -spread
        expected_total = total × 0.25
        
        margin_diff = |actual_margin - expected_margin|
        total_diff = |actual_total - expected_total|
        
        margin_factor = exp(-margin_diff / 10)
        total_factor = exp(-total_diff / 5)
        
        raw_prob[score] = empirical_prob[score] × margin_factor × total_factor

# Phase 2: Normalize
total = Σ raw_prob[score]
For each score:
    model_prob[score] = raw_prob[score] / total

# Phase 3: Market calibration
For each score:
    # Interpolate market probability
    market_prob = interpolate_market_data(spread, total, score)
    
    If market_prob exists:
        # Calculate calibration weight based on distance
        distance = min_distance_to_calibration_point(spread, total, score)
        α = 0.7 × exp(-distance / 2.0)
        
        # Blend model and market
        calibrated_prob[score] = α × market_prob + (1-α) × model_prob
    Else:
        calibrated_prob[score] = model_prob[score]

# Phase 4: Final normalization
total = Σ calibrated_prob[score]
For each score:
    final_prob[score] = calibrated_prob[score] / total
```

#### Step 5: Output Distribution

```
Score  | Empirical | Model    | Market   | Final   
-------|-----------|----------|----------|--------
7-0    |   11.0%   |  14.2%   |  13.5%   | 13.7%
7-7    |    7.6%   |   8.1%   |   7.4%   |  7.6%
0-0    |    7.1%   |   5.8%   |   6.2%   |  6.1%
14-0   |    6.6%   |   8.9%   |   8.2%   |  8.4%
...
```

---

## Worked Example

### Setup

**Game characteristics:**
- Spread: -10.5 (home favored by 10.5)
- Total: 54.5

**Historical data:**
- 7,000 games total
- Score 14-0 occurred 460 times (6.6%)
- Score 7-7 occurred 528 times (7.6%)

### Step 1: Empirical Probabilities

```
P(14-0) = 460 / 7000 = 0.066
P(7-7) = 528 / 7000 = 0.076
```

### Step 2: Calculate Priors

```
logit_prior(14-0) = log(0.066 / 0.934) = -2.65
logit_prior(7-7) = log(0.076 / 0.924) = -2.49
```

### Step 3: Train Models

**For 14-0 score:**

Regularization: `λ = 0.1 × (1 + 1/460) = 0.100`

After optimization:
```
β₀ = -2.60  (slightly higher than prior, 14-0 slightly more common than baseline)
β₁ = +0.85  (strong positive: more likely when home is favored)
β₂ = +0.12  (weak positive: slightly more likely in higher-scoring games)
β₃ = -0.08  (negative quadratic: effect diminishes at extreme spreads)
β₄ = +0.03  (small quadratic: minimal non-linear total effect)
```

**For 7-7 score:**

Regularization: `λ = 0.1 × (1 + 1/528) = 0.100`

After optimization:
```
β₀ = -2.48  (close to prior)
β₁ = -0.02  (near zero: happens regardless of favorite)
β₂ = +0.31  (moderate positive: more likely in higher-scoring games)
β₃ = -0.15  (negative quadratic: less likely at extreme spreads)
β₄ = +0.05  (small quadratic: minimal non-linear total effect)
```

### Step 4: Make Predictions

**Normalize inputs:**
```
norm_spread = -10.5 / 10.0 = -1.05
norm_total = (54.5 - 50) / 20.0 = +0.225
norm_spread² = 1.1025
norm_total² = 0.0506
```

**Calculate logits:**

For 14-0:
```
logit = -2.60 + 0.85×(-1.05) + 0.12×0.225 + (-0.08)×1.1025 + 0.03×0.0506
      = -2.60 - 0.893 + 0.027 - 0.088 + 0.002
      = -3.55
```

For 7-7:
```
logit = -2.48 + (-0.02)×(-1.05) + 0.31×0.225 + (-0.15)×1.1025 + 0.05×0.0506
      = -2.48 + 0.021 + 0.070 - 0.165 + 0.003
      = -2.55
```

**Convert to raw probabilities:**
```
raw_prob(14-0) = 1 / (1 + exp(3.55)) = 0.0275 (2.75%)
raw_prob(7-7) = 1 / (1 + exp(2.55)) = 0.0723 (7.23%)
```

**After normalizing all 300+ scores:**
```
model_prob(14-0) = 0.089 (8.9%)
model_prob(7-7) = 0.064 (6.4%)
```

### Step 5: Market Calibration

**Assume calibration files:**
- `10.5_54.5.csv` exists (exact match)
- Contains: 14-0 → 0.082, 7-7 → 0.068

**Distance = 0 (exact match), so α = 0.70**

**For 14-0:**
```
market_prob = 0.082
calibrated_prob = 0.70 × 0.082 + 0.30 × 0.089
                = 0.0574 + 0.0267
                = 0.084 (8.4%)
```

**For 7-7:**
```
market_prob = 0.068
calibrated_prob = 0.70 × 0.068 + 0.30 × 0.064
                = 0.0476 + 0.0192
                = 0.067 (6.7%)
```

### Step 6: Final Normalization

After calibrating all scores and normalizing:
```
final_prob(14-0) = 0.084 (8.4%)
final_prob(7-7) = 0.067 (6.7%)
```

### Summary of Adjustments

| Score | Empirical | Model | Market | Final | Change |
|-------|-----------|-------|--------|-------|--------|
| 14-0 | 6.6% | 8.9% | 8.2% | 8.4% | +1.8% |
| 7-7 | 7.6% | 6.4% | 6.8% | 6.7% | -0.9% |

**Why the changes?**

- **14-0:** Heavy home favorite (-10.5 spread) makes blowout start more likely. Model increases probability, market confirms this adjustment.
- **7-7:** Large spread makes balanced score less likely. Model decreases probability, market confirms this pattern.

---

## Key Takeaways

### Model Architecture

1. **Independent binary models:** Each score gets its own logistic regression
2. **Adaptive regularization:** Rare scores stay closer to baseline
3. **Market calibration:** Blend model with market data when available
4. **Normalization:** Ensure valid probability distribution

### Why This Approach Works

**Flexibility:**
- Per-score regularization adapts to data availability
- Can incorporate market data without retraining
- Handles 300+ scores efficiently

**Robustness:**
- Regularization prevents overfitting rare scores
- Market calibration corrects systematic model biases
- Distance-based fallback for scores without models

**Interpretability:**
- Each score's coefficients are interpretable
- Can examine which features drive specific scores
- Easy to debug individual score predictions

### Limitations

**Data dependency:**
- Quality limited by historical data availability
- Rare score combinations have high uncertainty
- Market calibration only as good as market efficiency

**Model assumptions:**
- Linear and quadratic effects may miss complex patterns
- Independence assumption ignores score correlations
- Calibration assumes market is well-informed

**Scope:**
- Only predicts Q1 scores (not Q2, Q3, Q4)
- Doesn't account for coaching tendencies
- Doesn't incorporate first possession information
- Doesn't consider weather, injuries, or other context

---

## Future Improvements

### Potential Enhancements

1. **Coaching tendencies:**
   - Incorporate coach-specific patterns
   - Model first possession decisions
   - Account for offensive/defensive styles

2. **Additional features:**
   - Team strength ratings
   - Recent form/momentum
   - Weather conditions
   - Key player availability

3. **Sequential modeling:**
   - Model Q2, Q3, Q4 probabilities
   - Incorporate Q1 outcome into later quarter predictions
   - Full game simulation

4. **Advanced calibration:**
   - Time-varying calibration (early week vs close to game)
   - Book-specific calibration
   - Ensemble multiple market sources

5. **Model refinement:**
   - Explore other basis functions beyond quadratic
   - Test alternative regularization schemes
   - Incorporate score correlation structure

---

## Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| `P(A)` | Probability of event A |
| `P(A\|B)` | Probability of A given B |
| `log(x)` | Natural logarithm |
| `exp(x)` | Exponential function (e^x) |
| `β` | Regression coefficient |
| `λ` | Regularization strength |
| `Σ` | Summation |
| `α` | Calibration weight |
| `ε` | Small constant (epsilon) |

---

## Glossary

**Binary Classification:** Predicting one of two outcomes (yes/no, 1/0)

**Calibration:** Adjusting model predictions to match observed frequencies

**Empirical Distribution:** Probability distribution derived from observed data

**Feature:** Input variable used for prediction (spread, total)

**Logit:** Log-odds, transformation of probability to unbounded scale

**One-vs-All:** Training separate binary classifiers for each class

**Overfitting:** Model learning noise rather than signal from training data

**Regularization:** Penalty term that prevents extreme coefficient values

**Sigmoid:** Function that maps any number to probability between 0 and 1

**Softmax:** Function that converts scores to probabilities that sum to 1

---

*This methodology represents the current state of the model. As new data becomes available and techniques improve, the approach will evolve.*