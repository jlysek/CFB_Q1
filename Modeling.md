# Statistical Methodology Guide

A complete breakdown of the statistical methods used in the College Football Quarter Scoring Prediction Model. This guide assumes no advanced statistics background and builds concepts from the ground up.

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Understanding Probability Distributions](#understanding-probability-distributions)
3. [What is Bayesian Statistics?](#what-is-bayesian-statistics)
4. [Building the Prior Distribution](#building-the-prior-distribution)
5. [Multinomial Logistic Regression](#multinomial-logistic-regression)
6. [Regularization and Overfitting](#regularization-and-overfitting)
7. [Putting It All Together](#putting-it-all-together)
8. [Worked Example](#worked-example)

---

## The Big Picture

### What Are We Trying To Do?

We want to answer this question: **"Given what betting markets think about a game (the spread and total), what is the probability of every possible first quarter score?"**

For example:
- Spread: Home team favored by 7.5 points
- Total: 58.5 points expected in the full game
- Question: What's the probability Q1 ends 7-0? 14-7? 0-0? 21-3?

### Why Is This Hard?

There are hundreds of possible Q1 score combinations, and we need to assign a probability to each one that:
1. Makes sense given the pregame betting line
2. Reflects historical scoring patterns
3. Sums to exactly 100% across all possibilities

### Our Solution: Bayesian Statistics

Instead of trying to predict scores from scratch, we start with historical patterns (what normally happens) and update those patterns based on new information (the betting market's opinion of this specific game).

---

## Understanding Probability Distributions

### What is a Probability Distribution?

A probability distribution assigns a probability to every possible outcome. The probabilities must sum to exactly 1 (or 100%).

**Simple Example - Rolling a Die:**
```
Outcome:     1    2    3    4    5    6
Probability: 1/6  1/6  1/6  1/6  1/6  1/6
Total:       1.0 (or 100%)
```

**Our Example - Q1 Scores:**
```
Score:       0-0   3-0   7-0   0-3   7-3   ...  (hundreds of possibilities)
Probability: 0.08  0.05  0.12  0.04  0.03  ...
Total:       1.0 (or 100%)
```

### Empirical vs Theoretical Distributions

**Empirical Distribution:** Built from real data
- Count how often each outcome happened historically
- Turn counts into percentages
- Example: In 6,000 games, Q1 ended 7-0 in 720 games = 12% probability

**Theoretical Distribution:** Based on assumptions or models
- Use mathematical formulas to assign probabilities
- Example: Normal distribution, exponential distribution
- We use a hybrid approach: start empirical, then adjust with models

---

## What is Bayesian Statistics?

### The Core Concept

Bayesian statistics is about updating beliefs with new evidence. Start with what you know, then adjust when you learn more.

**Real Life Example:**
- **Prior belief:** "There's a 50% chance it will rain today" (based on this month's average)
- **New evidence:** You see dark clouds forming
- **Updated belief:** "There's an 80% chance it will rain today"

### The Bayesian Formula

```
P(Hypothesis | Evidence) = [P(Hypothesis) × P(Evidence | Hypothesis)] / P(Evidence)
```

**Breaking this down:**
- **P(Hypothesis | Evidence):** Probability of hypothesis GIVEN the evidence (what we want)
- **P(Hypothesis):** Probability of hypothesis BEFORE seeing evidence (our prior)
- **P(Evidence | Hypothesis):** Probability of seeing this evidence IF the hypothesis is true (likelihood)
- **P(Evidence):** Probability of seeing this evidence overall (normalizing constant)

### Our Application

```
P(Q1 Score | Spread & Total) = [P(Q1 Score) × P(Spread & Total | Q1 Score)] / P(Spread & Total)
```

**In plain English:**
- **What we want:** Probability of a specific Q1 score given the betting line
- **Prior:** Historical probability of that Q1 score (from 10+ years of data)
- **Likelihood:** If Q1 did end with that score, how likely would we see this betting line?
- **Evidence:** How common is this betting line overall? (ensures probabilities sum to 1)

---

## Building the Prior Distribution

### Step 1: Collect Historical Data

We have every FBS college football game since 2014 - over 6,000 games with Q1 scores.

### Step 2: Count Score Occurrences

```
Score   Count   Percentage
0-0     480     8.0%
7-0     720     12.0%
7-3     180     3.0%
14-7    120     2.0%
...
```

### Step 3: Create the Prior

This becomes our **prior distribution** - what we expect when we know nothing about a specific game. It represents the "average" Q1 in college football.

**Key Insight:** Some scores are naturally more common (7-0, 0-0, 14-7) while others are rare (21-17, 28-3). Our prior captures these natural patterns.

### Why This Matters

Without any information about a game, we should predict scores proportional to how often they actually happen. A 7-0 Q1 is much more common than a 28-14 Q1, and our prior reflects this.

---

## Multinomial Logistic Regression

This is the heart of our model and requires the most explanation. We'll build up the concept step by step.

### What Problem Does It Solve?

We need to figure out: **Given a Q1 score, what spread and total would we expect to see?**

This seems backwards, but it's necessary for Bayesian updating. We need P(Spread & Total | Score) to calculate P(Score | Spread & Total).

### Part 1: Understanding Logistic Functions

#### From Probability to Logit

**Problem:** Regression models work best with numbers that can range from negative infinity to positive infinity. Probabilities are trapped between 0 and 1.

**Solution:** Transform probabilities using the logit function.

```
logit(p) = log(p / (1-p))
```

**What this does:**

| Probability | Logit Value | Interpretation |
|-------------|-------------|----------------|
| 0.01 (1%)   | -4.6        | Very unlikely |
| 0.10 (10%)  | -2.2        | Unlikely |
| 0.50 (50%)  | 0.0         | Even odds |
| 0.90 (90%)  | +2.2        | Likely |
| 0.99 (99%)  | +4.6        | Very likely |

Now we can work with numbers from -∞ to +∞, which regression loves.

#### From Logit Back to Probability

The inverse operation:
```
p = exp(logit) / (1 + exp(logit))
```

This is called the **logistic function** and it maps any number back to a probability between 0 and 1.

### Part 2: Simple Logistic Regression

Before multinomial, let's understand regular logistic regression.

**Example Goal:** Predict if home team wins based on spread.

```
logit(p_win) = β₀ + β₁ × spread
```

**What the coefficients mean:**
- **β₀ (intercept):** Probability of winning when spread is 0 (pick'em game)
- **β₁ (slope):** How much probability changes per point of spread

**Example with learned values:**
```
logit(p_win) = 0.0 + 0.3 × spread

Spread = -7 (home favored):  logit = -2.1  →  p_win = 0.89 (89%)
Spread = 0 (pick'em):        logit = 0.0   →  p_win = 0.50 (50%)
Spread = +7 (home underdog): logit = +2.1  →  p_win = 0.11 (11%)
```

**Learning the coefficients:** We use historical data where we know both the spread and the outcome. An optimization algorithm finds the β values that best predict the observed outcomes.

### Part 3: Multinomial Logistic Regression

Now extend this to many possible outcomes (all the different Q1 scores).

#### The Setup

Instead of predicting one probability, we predict a probability for each possible score simultaneously. Each score gets its own logit equation.

**For each score, we model:**
```
logit(score) = β₀ + β₁ × spread + β₂ × total + β₃ × spread² + β₄ × total²
```

**Why 5 coefficients per score?**
1. **β₀:** Baseline - how common is this score regardless of betting line?
2. **β₁:** Spread effect - does a bigger spread favor this score?
3. **β₂:** Total effect - do higher totals make this score more likely?
4. **β₃:** Non-linear spread effect - relationship may not be linear
5. **β₄:** Non-linear total effect - relationship may not be linear

#### Why Quadratic Terms (spread² and total²)?

Relationships aren't always linear. Consider the move from spread of 3 to spread of 6 - this is huge (changes the game dynamic significantly). But the move from spread of 33 to 36? Much less meaningful.

Quadratic terms capture this: they allow the effect of spread to change depending on how large the spread already is.

**Visual Example:**
```
Linear effect:     Each point of spread matters equally
Quadratic effect:  Points matter more when spread is small, less when it's large
```

### Part 4: Making Predictions

Let's say we have 300 possible Q1 scores. For each one, we calculate:

```
logit_score1 = β₀₁ + β₁₁ × spread + β₂₁ × total + β₃₁ × spread² + β₄₁ × total²
logit_score2 = β₀₂ + β₁₂ × spread + β₂₂ × total + β₃₂ × spread² + β₄₂ × total²
...
logit_score300 = β₀₃₀₀ + β₁₃₀₀ × spread + β₂₃₀₀ × total + β₃₃₀₀ × spread² + β₄₃₀₀ × total²
```

Then convert to probabilities using softmax (a generalized logistic function):

```
p(score_i) = exp(logit_i) / sum(exp(logit_j) for all j)
```

This ensures all probabilities sum to exactly 1.

### Part 5: Training the Model

**Input data:** Historical games with Q1 scores, spread, and total

**Process:**
1. For each possible score, create a binary variable (1 if this was the actual Q1 score, 0 otherwise)
2. Use optimization to find β coefficients that maximize likelihood
3. The model learns which scores are associated with which betting lines

**Key Insight:** A 14-0 Q1 score correlates with larger spreads and moderate totals. A 3-3 score correlates with small spreads and lower totals. The coefficients capture these patterns mathematically.

### Part 6: Normalization

Before running regression, we normalize spread and total:
```
normalized_value = (actual_value - mean) / standard_deviation
```

**Why normalize?**
- Spread ranges from -40 to +40
- Total ranges from 30 to 100
- Without normalization, total would dominate because its numbers are bigger
- Normalization puts both on the same scale

**Example:**
```
Raw:        spread = -14,  total = 58
Mean:       spread = 0,    total = 60
Std Dev:    spread = 12,   total = 10

Normalized: spread = -1.17, total = -0.20
```

Now both variables contribute proportionally to the model.

---

## Regularization and Overfitting

### The Overfitting Problem

**Scenario:** Only 5 games in history ended Q1 with a score of 24-17.

**Problem:** With so little data, the model might learn spurious patterns:
- "Games ending 24-17 always have totals between 67.5 and 69.5"
- This is probably just random chance, not a real pattern

**Result:** The model makes wild, overconfident predictions for rare scores.

### The Solution: L2 Regularization

Add a penalty term to the loss function that discourages extreme coefficient values:

```
Loss = -LogLikelihood + λ × ||β - β_prior||²
```

**Breaking this down:**

**-LogLikelihood:** How well the model fits the training data (we want to minimize this)

**λ:** Regularization strength (higher = more penalty)

**||β - β_prior||²:** Distance between learned coefficients and prior coefficients

### How It Works

Instead of letting coefficients go anywhere, we pull them toward "prior" values that make sense based on the empirical distribution.

**For common scores (lots of data):**
- Low λ (weak regularization)
- Model learns mostly from data
- Coefficients can deviate significantly from prior

**For rare scores (little data):**
- High λ (strong regularization)
- Model stays close to prior
- Prevents wild predictions from limited data

### Adaptive Regularization

```
λ_score = base_lambda × (1 / sqrt(count_score))
```

Scores with fewer observations get stronger regularization automatically.

### Distance-Based Smoothing

For extremely rare scores, we also borrow information from similar scores:

```
If score 28-17 is very rare, borrow from:
- 28-14 (similar home score)
- 24-17 (similar away score)  
- 28-20 (similar total)
```

**Method:** Weight nearby scores by similarity, create a smoothed estimate.

This prevents the model from having "holes" in its predictions where it has no idea what to predict.

---

## Putting It All Together

### The Complete Pipeline

**Step 1: Calculate Prior**
```
For each possible Q1 score:
    prior_prob[score] = count[score] / total_games
```

**Step 2: Train Multinomial Logistic Regression**
```
For each possible Q1 score:
    Learn coefficients β₀, β₁, β₂, β₃, β₄
    Apply regularization based on score frequency
    Store learned model
```

**Step 3: Make Predictions**
```
Given: spread = -7.5, total = 58.5

For each possible score:
    # Normalize inputs
    norm_spread = (spread - mean_spread) / std_spread
    norm_total = (total - mean_total) / std_total
    
    # Calculate logit
    logit[score] = β₀ + β₁×norm_spread + β₂×norm_total 
                   + β₃×norm_spread² + β₄×norm_total²
    
    # Get likelihood from logistic regression
    likelihood[score] = exp(logit[score]) / sum(exp(all logits))
    
    # Combine with prior using Bayes
    numerator[score] = prior[score] × likelihood[score]

# Normalize to get final probabilities
For each score:
    final_prob[score] = numerator[score] / sum(all numerators)
```

**Step 4: Output Probability Distribution**
```
Score   Prior   Likelihood   Posterior
0-0     0.080   0.024       0.042
7-0     0.120   0.089       0.185
7-3     0.030   0.041       0.027
...
```

---

## Worked Example

Let's walk through a complete example with real numbers.

### Given Information
- **Spread:** Home team favored by 7.5 points (-7.5)
- **Total:** 58.5 points
- **Goal:** Find probability of Q1 ending 7-0 (home team leads)

### Step 1: Prior Probability

From historical data:
```
P(7-0) = 720 / 6000 = 0.12 (12%)
```

Out of 6,000 games, 720 ended Q1 with a 7-0 score.

### Step 2: Normalize Inputs

```
Historical averages: mean_spread = 0, std_spread = 10
                    mean_total = 60, std_total = 8

norm_spread = (-7.5 - 0) / 10 = -0.75
norm_total = (58.5 - 60) / 8 = -0.19
```

### Step 3: Calculate Likelihood

Learned coefficients for 7-0 score (these come from training):
```
β₀ = -2.1   (intercept)
β₁ = -0.4   (spread effect)
β₂ = 0.15   (total effect)
β₃ = -0.05  (spread² effect)
β₄ = 0.02   (total² effect)
```

Calculate logit:
```
logit = -2.1 + (-0.4)×(-0.75) + (0.15)×(-0.19) 
        + (-0.05)×(-0.75)² + (0.02)×(-0.19)²
      = -2.1 + 0.30 - 0.029 - 0.028 + 0.001
      = -1.86
```

Convert to probability using softmax (assuming sum of all exp(logits) = 45.2):
```
likelihood = exp(-1.86) / 45.2 = 0.156 / 45.2 = 0.0035
```

**Interpretation:** Given the betting line (home favored by 7.5), we'd see a 7-0 Q1 score about 0.35% of the time according to our likelihood model.

### Step 4: Apply Bayes' Rule

```
Numerator = Prior × Likelihood
          = 0.12 × 0.0035
          = 0.00042
```

Do this for all 300+ possible scores, then normalize:
```
Sum of all numerators = 0.0485

Posterior probability = 0.00042 / 0.0485 = 0.0087 (0.87%)
```

### Step 5: Interpret Result

**Prior:** 12% (before seeing betting line)
**Posterior:** 0.87% (after seeing betting line)

**Why did it drop?**
- A 7-0 score means a one-score lead
- But the spread is 7.5 points, meaning we expect the home team to dominate
- A 7-0 Q1 doesn't reflect enough home team dominance for this spread
- More likely Q1 scores: 14-0, 10-0, 14-3 (bigger home leads)

The model correctly reduced the probability of 7-0 because it doesn't align well with the betting market's expectation of a blowout.

---

## Key Takeaways

### For the Non-Technical Reader

1. **We start with history:** What normally happens in Q1?
2. **We update with information:** How does this specific game differ from average?
3. **We use sophisticated math:** To combine these two pieces of information optimally
4. **We get probabilities:** Not just one prediction, but how likely every possible outcome is

### For the Technical Reader

1. **Bayesian framework** naturally combines prior knowledge with observed evidence
2. **Multinomial logistic regression** models the relationship between game characteristics and scoring outcomes
3. **Regularization** prevents overfitting and stabilizes predictions for rare events
4. **Normalization and polynomial features** improve model performance and capture non-linear relationships

### Why This Approach is Powerful

- **Probabilistic:** Provides full distribution, not just point estimates
- **Data-driven:** Learns from 10+ years of actual games
- **Adaptive:** Automatically adjusts for common vs rare scores
- **Interpretable:** Each component has clear statistical meaning
- **Extensible:** Can easily add new features (weather, coaching tendencies, etc.)

---

## Further Reading

### Probability and Statistics
- **Bayesian Statistics:** "Doing Bayesian Data Analysis" by John Kruschke
- **Logistic Regression:** "An Introduction to Statistical Learning" by James et al.

### Regularization
- **Ridge Regression (L2):** ESL Chapter 3, Hastie et al.
- **Bias-Variance Tradeoff:** Understanding overfitting vs underfitting

### Sports Analytics
- **NFL Modeling:** FiveThirtyEight's Elo ratings
- **Expected Goals (xG):** Soccer analytics frameworks
- **Win Probability:** Baseball's WPA (Win Probability Added)

---

*This methodology guide is designed to make advanced statistical modeling accessible. Questions and suggestions for improvement are welcome.*