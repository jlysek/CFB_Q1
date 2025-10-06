# Statistical Methodology

This is my attempt to breakdown the statistical methods used in the College Football Quarter Scoring Prediction Model as clearly as possible.

---

## Table of Contents

- [Overview](#overview)
- [Understanding Probability Distributions](#understanding-probability-distributions)
  - [Empirical vs Model-Based Distributions](#empirical-vs-model-based-distributions)
- [Building the Empirical Distribution](#building-the-empirical-distribution)
- [Multinomial Logistic Regression](#multinomial-logistic-regression)
  - [Part 1: Understanding Logistic Functions](#part-1-understanding-logistic-functions)
  - [Part 2: Simple Logistic Regression](#part-2-simple-logistic-regression)
  - [Part 3: Multinomial Logistic Regression](#part-3-multinomial-logistic-regression)
  - [Part 4: Making Predictions with Softmax](#part-4-making-predictions-with-softmax)
  - [Part 5: Training the Model](#part-5-training-the-model)
  - [Part 6: Feature Normalization](#part-6-feature-normalization)
- [Regularization and Overfitting](#regularization-and-overfitting)
- [Putting It All Together](#putting-it-all-together)
- [Worked Example](#worked-example)

---

## Overview

**Goal:** Given pregame closing betting markets (spread and total), what is the probability of every possible first quarter score?

### Example

- **Spread:** Home team favored by 7.5 points
- **Total:** 58.5 points expected in the full game
- **Question:** What's the probability Q1 ends 7-7? 14-7? 0-0? 7-3?

### Regularized Supervised Learning

Use multinomial logistic regression to directly model `P(Score | Spread, Total)`. Apply regularization to prevent overfitting, especially for rare scores.

The model learns which scores correlate with which betting lines from historical games, while regularization keeps predictions reasonable for scores with limited data.

---

## Understanding Probability Distributions

A probability distribution assigns a probability to every possible outcome. Probabilities must sum to exactly 1 (or 100%).

### Simple Example - Rolling a Die

| Outcome | Probability |
|---------|-------------|
| 1 | 1/6 |
| 2 | 1/6 |
| 3 | 1/6 |
| 4 | 1/6 |
| 5 | 1/6 |
| 6 | 1/6 |
| **Total** | **1.0 (100%)** |

### Conditional Probability Example
P(x > 4) = P(5) + P(6) = 1/6 + 1/6 = 2/6 = 1/3
P(x > 4 | x is even) = "Probability x is greater than 4 GIVEN x is even"

Since x is even, possible outcomes shrink to: `{2, 4, 6}`
P(x > 4 | x is even) = P(6) / P(2,4,6) = (1/6) / (3/6) = 1/3

> The "|" symbol means "given" in probability notation

### Our Example - Q1 Scores

| Score | Probability |
|-------|-------------|
| 0-0 | 0.062 |
| 3-0 | 0.038 |
| 7-0 | 0.098 |
| 0-3 | 0.023 |
| 7-3 | 0.037 |
| ... | ... |
| **Total** | **1.0 (100%)** |

---

## Empirical vs Model-Based Distributions

### Empirical Distribution: Built from real data
- Count how often each outcome happened historically
- Turn counts into percentages
- **Example:** In 6,000 games, Q1 ended 7-0 in 720 games = 12% probability

### Theoretical Distribution: Based on assumptions or models
- Use mathematical formulas to assign probabilities
- **Example:** Normal distribution, exponential distribution

**We use both:** Start with empirical frequencies, then train models that adjust predictions based on game characteristics.

---

## Building the Empirical Distribution

### Step 1: Collect Historical Data

Every FBS college football game since 2014 with Q1 scores.

### Step 2: Count Score Occurrences

From the actual model output (at time of writing):

| Score | Count | Percentage |
|-------|-------|------------|
| 7-0 | 765 | 11.0% |
| 7-7 | 528 | 7.6% |
| 0-7 | 490 | 7.1% |
| 0-0 | 490 | 7.1% |
| 14-0 | 460 | 6.5% |
| 3-0 | 320 | 4.5% |
| ... | ... | ... |

### Step 3: Use as Training Foundation

This empirical distribution serves two purposes:

1. **Training target:** The model learns to predict these historical patterns
2. **Regularization anchor:** Prevents wild predictions for rare scores

---

## Multinomial Logistic Regression

This is the core of our prediction model.

**We need to predict:** Given a spread and total, what is the probability of each possible Q1 score?

This is a classification problem with hundreds of possible classes (each unique score combination).

### Part 1: Understanding Logistic Functions

#### From Probability to Logit

Regression works best with numbers ranging from -∞ to +∞. Probabilities are constrained between 0 and 1.

So we transform the probabilities using the **logit function**:
logit(p) = log(p / (1-p))

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

The inverse operation:
p = exp(logit) / (1 + exp(logit))

This is the **logistic function** - it maps any number back to a probability between 0 and 1.

---

### Part 2: Simple Logistic Regression

Before multinomial, understand regular logistic regression.

**Example Goal:** Predict if home team wins based on spread.
logit(p_win) = β₀ + β₁ × spread

**What coefficients mean:**
- `β₀` (intercept): Baseline log-odds when spread is 0 for home team
- `β₁` (slope): How spread affects win probability

**Example with learned values:**
logit(p_win) = 0.0 + 0.3 × spread

| Scenario | Calculation | Result |
|----------|-------------|--------|
| Spread = -7 (home favored) | logit = -2.1 | p_win = 0.11 (11%) |
| Spread = 0 (pick'em) | logit = 0.0 | p_win = 0.50 (50%) |
| Spread = +7 (home underdog) | logit = +2.1 | p_win = 0.89 (89%) |

---

### Part 3: Multinomial Logistic Regression

**Multinomial** = consisting of several terms. Extend this to many possible outcomes (all different Q1 scores).

#### The Setup

Instead of predicting one probability, we predict a probability for each possible score. Each score gets its own logit equation.

**For each score:**
logit(score) = β₀ + β₁ × spread + β₂ × total + β₃ × spread² + β₄ × total²

**5 coefficients per score:**

1. `β₀`: Baseline - probability when spread and total are zero
2. `β₁`: Spread effect - How a 1-point change in spread affects this score's probability
3. `β₂`: Total effect - How a 1-point change in total affects this score's probability
4. `β₃`: Non-linear spread effect - Captures non-linear spread effects
5. `β₄`: Non-linear total effect - Captures non-linear total effects

#### Why Quadratic Terms (spread² and total²)?

Relationships aren't always linear. Consider:

- Spread moving from -2.5 to -5.5: **Huge impact**
- Spread moving from -29.5 to -32.5: **Minimal impact**

Quadratic terms capture this: they allow effects to vary based on the input magnitude.

**Visual Example:**
Linear effect:     Each point of spread matters equally
Quadratic effect:  Early points matter more than later points

---

### Part 4: Making Predictions with Softmax

For 300 possible Q1 scores, calculate logits for each:
logit_score1 = β₀₁ + β₁₁ × spread + β₂₁ × total + β₃₁ × spread² + β₄₁ × total²
logit_score2 = β₀₂ + β₁₂ × spread + β₂₂ × total + β₃₂ × spread² + β₄₂ × total²
...
logit_score300 = β₀₃₀₀ + β₁₃₀₀ × spread + β₂₃₀₀ × total + β₃₃₀₀ × spread² + β₄₃₀₀ × total²

**Convert to probabilities using softmax:**
p(score_i) = exp(logit_i) / sum(exp(logit_j) for all j)

This ensures all probabilities sum to exactly 1.

---

### Part 5: Training the Model

**Input:** Historical games with Q1 scores, spreads, and totals

**Process:**

1. For each score, create a binary indicator (1 if actual score, 0 otherwise)
2. Use optimization to find β coefficients that maximize log-likelihood
3. Apply regularization to prevent overfitting (see next section)

**What the model learns:**

- 14-0 scores correlate with large spreads and moderate totals
- 3-3 scores correlate with small spreads and lower totals

---

### Part 6: Feature Normalization

Before regression, we normalize inputs:
normalized_value = (actual_value - mean) / standard_deviation

**Why normalize?**

- Spread ranges: -40 to +40
- Total ranges: 30 to 100
- Without normalization, total dominates due to larger magnitude
- Normalization puts both on equal footing

**Example:**

| Feature | Raw | Mean | Std Dev | Normalized |
|---------|-----|------|---------|------------|
| Spread | -14 | 0 | 12 | -1.17 |
| Total | 58 | 60 | 10 | -0.20 |

---

## Regularization and Overfitting

### The Overfitting Problem

**Scenario:** Only 5 games in history ended Q1 with score 24-17.

**Problem:** With limited data, the model might learn random noise:
- "24-17 games always have totals between 67.5-69.5"
- This is likely coincidence, not a real pattern

**Result:** Wild, overconfident predictions for rare scores.

### The Solution: L2 Regularization

Add a penalty for extreme coefficients:
Loss = -LogLikelihood + λ × ||β - β_prior||²

**Components:**

- `-LogLikelihood`: How well the model fits training data (minimize this)
- `λ`: Regularization strength (higher = more penalty)
- `||β - β_prior||²`: Distance from prior coefficients

### How It Works

Pull coefficients toward sensible "prior" values based on empirical frequencies.

**For common scores (lots of data):**
- Low λ (weak regularization)
- Model learns mostly from data
- Can deviate from prior

**For rare scores (little data):**
- High λ (strong regularization)
- Model stays close to prior
- Prevents overfitting

### Adaptive Regularization
λ_score = base_λ × (1 / √count_score)

Rare scores automatically get stronger regularization.

**Example:**

| Score Occurrences | Calculation | λ Value | Strength |
|-------------------|-------------|---------|----------|
| 600 | 0.1 × (1/√600) | 0.004 | Weak |
| 5 | 0.1 × (1/√5) | 0.045 | Strong |

### Distance-Based Smoothing

For extremely rare scores, borrow information from similar scores:

**If 28-17 is very rare, borrow from:**
- 28-14 (similar home score)
- 24-17 (similar away score)
- 28-20 (similar total)

Weight by similarity to create smoothed estimates.
---

## Putting It All Together

### The Complete Pipeline

#### Step 1: Calculate Empirical Distribution
```python
For each Q1 score:
    prior_prob[score] = count[score] / total_games
Step 2: Train Regularized Models
pythonFor each Q1 score:
    Learn coefficients β₀, β₁, β₂, β₃, β₄
    Apply adaptive regularization based on frequency
    Store trained model
Step 3: Make Predictions
Given: spread = -7.5, total = 58.5
pythonFor each possible score:
    # Normalize inputs
    norm_spread = (spread - mean) / std
    norm_total = (total - mean) / std
    
    # Calculate raw logit
    logit[score] = β₀ + β₁×norm_spread + β₂×norm_total 
                   + β₃×norm_spread² + β₄×norm_total²
    
    # Convert to raw probability
    raw_prob[score] = exp(logit[score])

# Normalize with softmax
For each score:
    final_prob[score] = raw_prob[score] / sum(all raw_probs)
Step 4: Output Distribution
ScoreEmpiricalAdjustedFinal %7-011.0%→9.8%7-77.6%→9.8%0-77.1%→6.9%............

Worked Example
Let's walk through a complete prediction with real numbers.
Given Information
Game Parameters:

Spread: -2.5 (home team favored by 2.5 points)
Total: 57.5


Prediction Output
================================================================================
PREDICTION: Spread -2.5, Total 57.5
================================================================================
Probability Adjustment Analysis (Top 15 Scores)
ScoreEmpiricalAdjustedChangeFinal %7-70.0760260.102450+0.02642410.25%7-00.1101510.091173-0.0189799.12%0-70.0705540.082379+0.0118258.24%0-00.0706980.067480-0.0032196.75%7-30.0410370.050209+0.0091725.02%14-70.0358530.043866+0.0080134.39%3-70.0331170.042057+0.0089404.21%10-00.0368610.041124+0.0042634.11%3-00.0462200.040266-0.0059544.03%14-00.0666670.036943-0.0297233.69%7-140.0231820.034230+0.0110483.42%0-140.0309580.032289+0.0013313.23%0-30.0311020.027493-0.0036092.75%10-70.0200140.024419+0.0044042.44%0-100.0192940.022476+0.0031812.25%

Scores with Probability ≥ 0.5%
ScoreProbabilityPercentage7-70.10245010.25%7-00.0911739.12%0-70.0823798.24%0-00.0674806.75%7-30.0502095.02%14-70.0438664.39%3-70.0420574.21%10-00.0411244.11%3-00.0402664.03%14-00.0369433.69%7-140.0342303.42%0-140.0322893.23%0-30.0274932.75%10-70.0244192.44%0-100.0224762.25%7-100.0222052.22%3-30.0196501.97%14-30.0179321.79%14-140.0160711.61%21-70.0117271.17%21-00.0108551.09%3-100.0093020.93%3-140.0092080.92%7-60.0087110.87%6-70.0082780.83%14-100.0081000.81%10-30.0079480.79%17-00.0076800.77%0-60.0075130.75%6-00.0071160.71%13-00.0063400.63%7-210.0051650.52%

Betting Markets
2-Way Moneyline (Draws Void)
TeamProbabilityPercentageOddsHome0.572457.24%-133Away0.427642.76%+133
3-Way Moneyline
TeamProbabilityPercentageOddsHome0.453345.33%+120Draw0.208020.80%+380Away0.338733.87%+195
Spread Markets
LineHome CoverHome OddsAway CoverAway Odds+2.567.13%-20432.87%+204+1.566.99%-20233.01%+202+0.566.13%-19533.87%+195-0.545.33%+12054.67%-120-1.544.32%+12555.68%-125-2.544.11%+12655.89%-126-3.536.89%+17163.11%-171
Total Markets
LineOver ProbOver OddsUnder ProbUnder Odds8.565.58%-19034.42%+1909.564.44%-18135.56%+18110.548.86%+10451.14%-10411.548.83%+10451.17%-10412.548.76%+10551.24%-105
Special Markets
MarketProbabilityPercentageDraw and Over 10.50.120912.09%Draw and Over 12.50.120812.08%

Key Observations

7-7 is most likely (10.25%) - makes sense given the small spread and moderate total
Model adjusts empirical probabilities - 7-7 jumped from 7.6% to 10.25%, while 14-0 dropped from 6.7% to 3.7%
Home team slightly favored - 57.24% probability to outscore in Q1 (after removing draws)
High draw probability - 20.80% chance of a tied first quarter
Betting lines derived from distribution - All markets calculated directly from score probabilities


Understanding Softmax
The softmax function is crucial for multinomial logistic regression. Here's why:
The Problem
After calculating logits for all possible scores, we need to convert them to probabilities that:

Are all between 0 and 1
Sum to exactly 1

The Solution
p(score_i) = exp(logit_i) / Σ exp(logit_j) for all j
What this does:

exp(logit_i) converts each logit to a positive number
Dividing by the sum normalizes so all probabilities sum to 1
Higher logits → higher probabilities

Numerical Example
ScoreLogitexp(logit)Probability7-72.07.397.39/20.09 = 0.3687-01.54.484.48/20.09 = 0.2230-71.02.722.72/20.09 = 0.1350-00.51.651.65/20.09 = 0.082Others...3.853.85/20.09 = 0.192Sum-20.091.000