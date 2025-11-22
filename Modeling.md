# Statistical Methodology - College Football Quarter Prediction

This document explains the statistical methods used in our college football prediction model. 

## Table of Contents

- [Core Concepts](#core-concepts)
- [The Prediction Problem](#the-prediction-problem)
- [Feature Engineering](#feature-engineering)
- [Quarter-by-Quarter Modeling](#quarter-by-quarter-modeling)
- [Correlation Between Quarters](#correlation-between-quarters)
- [Monte Carlo Simulation](#monte-carlo-simulation)
- [Calibration Process](#calibration-process)
- [Worked Example](#worked-example)
- [Key Takeaways](#key-takeaways)

---

## Core Concepts

### Probability Distributions

A **probability distribution** tells you the chances of every possible outcome. 

**Example: Rolling a die**

| Outcome | Probability |
|---------|-------------|
| 1 | 16.7% |
| 2 | 16.7% |
| 3 | 16.7% |
| 4 | 16.7% |
| 5 | 16.7% |
| 6 | 16.7% |
| **Total** | **100%** |


### Conditional Probability

**Conditional probability** is "the probability of A happening, GIVEN that B already happened."

**Example: Weather**
- P(rain tomorrow) = 30%
- P(rain tomorrow | cloudy today) = 60%

If it's cloudy today, rain tomorrow becomes more likely. The "|" symbol means "given" or "knowing that."

**Example: Football**
- P(Q2 is high-scoring) = 25%
- P(Q2 is high-scoring | Q1 was high-scoring) = 35%

If Q1 had lots of points, Q2 is more likely to have lots of points too. Teams that score early often keep scoring.

### Independence vs Correlation

**Independent events:** Knowing one doesn't tell you about the other
- Coin flip #1 and coin flip #2

**Correlated events:** Knowing one tells you something about the other
- Q1 score and Q2 score in football

**Correlation coefficient (r):**
- r = 0: No relationship
- r = +0.5: Moderate positive relationship (when one goes up, other tends to go up)
- r = -0.5: Moderate negative relationship (when one goes up, other tends to go down)
- r = +1.0: Perfect positive relationship

In our data:
- Q1 total points and Q2 total points: r = +0.15 (slightly correlated)
- Q1 margin and Q2 margin: r = +0.12 (slightly correlated)

This means: high-scoring Q1 → slightly more likely to have high-scoring Q2.

### Logistic Regression

**Goal:** Predict probability of yes/no outcomes.

**Example: Will it rain tomorrow?**
```
Inputs: Temperature, humidity, cloud cover
Output: Probability of rain (0% to 100%)
```

**How it works:**

1. Calculate a "score" using the inputs:
   ```
   score = β₀ + β₁×temperature + β₂×humidity + β₃×clouds
   ```

2. Convert score to probability using the **sigmoid function**:
   ```
   probability = 1 / (1 + e^(-score))
   ```

**Sigmoid function properties:**
- Very negative score → probability near 0%
- Score of 0 → probability = 50%
- Very positive score → probability near 100%
- Always gives a valid probability between 0% and 100%

**The coefficients (β values):**
- β₁ > 0: As temperature increases, probability of rain increases
- β₁ < 0: As temperature increases, probability of rain decreases
- β₁ = 0: Temperature doesn't affect rain probability

---

## The Prediction Problem

### What We're Predicting

**Input:** Pregame betting lines
- Spread: -7.5 (home favored by 7.5 points)
- Total: 58.5 (expected combined score)

**Output:** Probability distribution for every possible score in every quarter

**Example Q1 predictions:**

| Score | Probability | Why |
|-------|-------------|-----|
| 7-0 | 10.2% | Most common score overall |
| 7-7 | 6.8% | Both teams score once |
| 14-0 | 5.4% | Favored team scores twice |
| 0-0 | 4.1% | Defensive start |
| 3-0 | 4.0% | Field goal only |

## Feature Engineering

### The Problem with Raw Inputs

Spread and total are **correlated** (not independent):

| Game Type | Typical Spread | Typical Total |
|-----------|---------------|---------------|
| Both teams good | -3 (small) | 58 (high) |
| One team dominates | -14 (large) | 52 (medium) |
| Defensive struggle | -7 (medium) | 45 (low) |

Notice: Large spreads often come with lower totals. This is **multicollinearity** - the inputs contain overlapping information.

**Problem:** When features are correlated, the model can't tell which one is causing the effect. 

### The Solution: Orthogonal Features

We transform the correlated inputs into **orthogonal** (uncorrelated) features.

**Step 1: Decompose into team expectations**

The spread and total implicitly tell us each team's expected score:

```
Spread = Home expected - Away expected
Total = Home expected + Away expected

Solving for individual teams:
  Favorite expected = (Total + |Spread|) / 2
  Underdog expected = (Total - |Spread|) / 2
```

**Example:**
```
Spread = -7.5 (home favored)
Total = 58.5

Home (favorite) expected = (58.5 + 7.5) / 2 = 33.0 points
Away (underdog) expected = (58.5 - 7.5) / 2 = 26.0 points
```

**Step 2: Normalize**

Divide by typical team score to make features scale-invariant:

```python
typical_team_score = 27.5  # Average in CFB
norm_fav = 33.0 / 27.5 = 1.200
norm_dog = 26.0 / 27.5 = 0.945
```

**Step 3: Create orthogonal features**

```python
feature_total = norm_fav + norm_dog = 1.200 + 0.945 = 2.145
feature_margin = norm_fav - norm_dog = 1.200 - 0.945 = 0.255
```

**Why this works:**

These new features are **orthogonal** - they're uncorrelated by mathematical construction. Think of them as:
- **feature_total:** How much scoring should we expect? (2.145 = high scoring)
- **feature_margin:** How lopsided should the game be? (0.255 = moderate favorite)

### Interpreting the Features

| feature_total | feature_margin | Game Type |
|---------------|----------------|-----------|
| 2.2 | 0.1 | High-scoring, close game (45-42) |
| 2.2 | 0.5 | High-scoring, blowout (38-24) |
| 1.8 | 0.1 | Low-scoring, close game (20-17) |
| 1.8 | 0.5 | Low-scoring, blowout (27-13) |

The features cleanly separate:
- **How much** scoring (total)
- **How lopsided** the scoring (margin)

---

## Quarter-by-Quarter Modeling

### The Approach: One Model Per Score

For each possible quarter score (7-0, 7-7, 14-0, etc.), we train a separate logistic regression model.

**The model:**
```
P(score happens) = sigmoid(β₀ + β₁×feature_total + β₂×feature_margin)
```

Each score learns its own β values.

### Two-Tier Architecture

Not all scores have enough data for reliable models:

**Tier 1: Anchor Scores (≥50 occurrences)**

These get individual models. In Q1:
- 7-0: 775 occurrences
- 7-7: 528 occurrences
- 0-0: 498 occurrences
- 14-0: 460 occurrences
- And about 40 more...

**Tier 2: Rare Scores (<50 occurrences)**

These are grouped into categories:

| Category | Description | Example Scores |
|----------|-------------|----------------|
| Favorite blowout | Fav wins by 14+ | 21-0, 28-7, 21-7 |
| Favorite wins | Fav wins by 7-13 | 10-3, 14-7, 17-7 |
| Close game | Fav wins by 1-6 | 7-3, 10-7, 14-10 |
| Underdog wins | Dog wins | 0-7, 3-10, 7-14 |
| Ties | Tied score | 3-3, 14-14, 10-10 |

**How rare scores work:**

1. Train model for entire category
2. Predict category probability
3. Distribute among scores proportionally

**Example:**
```
Category "ties" has P(category) = 5%
Historical proportions: 3-3 (40%), 14-14 (30%), 10-10 (30%)

Final predictions:
  P(3-3) = 5% × 40% = 2.0%
  P(14-14) = 5% × 30% = 1.5%
  P(10-10) = 5% × 30% = 1.5%
```

### Learning Logical Patterns

The models learn sensible patterns through coefficient signs:

**Score 0-0 (scoreless tie):**
- β₁ (total) = -0.8 (negative)
- β₂ (margin) = 0.0 (near zero)
- **Logic:** High-scoring games rarely stay 0-0

**Score 14-0 (favorite blowout start):**
- β₁ (total) = +0.8 (positive)
- β₂ (margin) = +1.2 (strongly positive)
- **Logic:** More likely when both scoring AND margin are high

**Score 7-7 (tie):**
- β₁ (total) = +0.3 (positive)
- β₂ (margin) = -0.5 (negative)
- **Logic:** Needs scoring but unlikely with large margin

These patterns emerge from training on 7,000+ games.

---

## Correlation Between Quarters

### Why Independence Fails

**Naive approach:**
```
P(Full game) = P(Q1) × P(Q2) × P(Q3) × P(Q4)
```

This assumes quarters are independent, but they're not:

**Evidence from data:**
- Q1-Q2 total correlation: r = +0.15
- Q1-Q2 margin correlation: r = +0.12
- Q3-Q4 total correlation: r = +0.18
- Q3-Q4 margin correlation: r = +0.14

**What this means:**
- High-scoring Q1 → Slightly more likely to have high-scoring Q2
- Blowout Q1 → Slightly more likely to have blowout Q2

### Conditional Probability Approach

Instead of independence, we use:

```
P(Full game) = P(Q1) × P(Q2|Q1) × P(Q3|H1) × P(Q4|H1,Q3)
```

Where:
- P(Q2|Q1) = Probability of Q2 score given Q1 outcome
- P(Q3|H1) = Probability of Q3 score given first half outcome
- P(Q4|H1,Q3) = Probability of Q4 score given entire game context

### Learning Conditional Distributions

**Step 1: Categorize prior outcomes**

We can't track every possible Q1 score separately (not enough data). Instead, we categorize:

**By total points:**
- Low: 0-7 points
- Medium: 8-20 points
- High: 21+ points

**By margin type:**
- Tie: 0 point difference
- Close favorite: 1-7 point lead
- Blowout favorite: 8+ point lead
- Close underdog: 1-7 point deficit
- Blowout underdog: 8+ point deficit

**Example categories:**
- "low_tie" = 0-0, 3-3, 7-7
- "med_close_fav" = 10-7, 14-7, 14-10
- "high_blow_fav" = 21-0, 28-7, 21-7

**Step 2: Count conditional frequencies**

For each category, track what happens next:

**Example: After "low_tie" Q1 (like 7-7):**
```
Q2 outcomes:
  0-0: 8%
  7-0: 12%
  7-7: 9%
  14-7: 7%
  ...
```

**Example: After "high_blow_fav" Q1 (like 21-0):**
```
Q2 outcomes:
  0-0: 6%
  7-0: 15%  (favorite keeps scoring)
  0-7: 8%   (underdog responds)
  ...
```

**Step 3: Apply conditional probabilities**

When predicting Q2, we:
1. Look at simulated Q1 outcome
2. Find its category
3. Adjust Q2 base probabilities using the conditional distribution

**Blending formula:**
```
P(Q2 score) = 70% × Base model + 30% × Conditional probability
```

The 70/30 split balances:
- Base model (from regression, uses betting lines)
- Conditional adjustment (from history, captures correlation)

### Building the Chain

**Q1:** Use base model only
```
P(Q1) from betting lines
```

**Q2:** Condition on Q1
```
P(Q2) = Adjust base model using P(Q2|Q1 category)
```

**Q3:** Condition on first half
```
H1 = Q1 + Q2
P(Q3) = Adjust base model using P(Q3|H1 category)
```

**Q4:** Condition on first half and Q3
```
P(Q4) = Adjust base model using P(Q4|H1 category, Q3 category)
```

This creates a Bayesian chain where each quarter's prediction uses all prior information.

---

## Monte Carlo Simulation

### The Concept

**Monte Carlo simulation:** Run the game thousands of times with random outcomes based on probabilities, then count the results.

### Our Football Simulation

**For each of 5,000 simulations:**

1. **Sample Q1** from its distribution
   ```
   Draw random number between 0 and 1
   Use that to pick a Q1 score based on probabilities
   ```

2. **Sample Q2** given Q1
   ```
   Look at Q1 category (e.g., "med_close_fav")
   Adjust Q2 probabilities based on P(Q2|Q1)
   Draw Q2 score from adjusted distribution
   ```

3. **Calculate first half**
   ```
   H1 = Q1 + Q2
   ```

4. **Sample Q3** given H1
   ```
   Look at H1 category
   Adjust Q3 probabilities based on P(Q3|H1)
   Draw Q3 score
   ```

5. **Sample Q4** given everything
   ```
   Look at H1 and Q3 categories
   Adjust Q4 probabilities based on P(Q4|H1,Q3)
   Draw Q4 score
   ```

6. **Calculate full game**
   ```
   Full game = Q1 + Q2 + Q3 + Q4
   ```

**After 5,000 simulations:**
```
Count how often each full game score appeared:
  31-24: 156 times → 156/5000 = 3.12%
  28-21: 142 times → 142/5000 = 2.84%
  34-27: 133 times → 133/5000 = 2.66%
  ...
```


## Calibration Process

### The Problem

After simulation, our full game distribution might not match the Pregame Market Inputs:

```
Market: -7.5 spread, 58.5 total
Our simulation:
  Favorite covers spread: 54% (should be 50%)
  Game goes over total: 48% (should be 50%)
```

**Why?** Our quarter models were trained independently, and even with correlation modeling, small biases can compound.

### The Solution: Iterative Calibration

**Goal:** Adjust quarter distributions so the simulated full game matches market exactly.

### The Algorithm

**Setup:**
```
Target: 50% cover spread, 50% over total
Tolerance: 49-51% (within 1%)
Max iterations: 10
```

**Each iteration:**

1. **Simulate full game** with current quarter distributions
   ```
   Run 2,000-5,000 simulations (adaptive)
   ```

2. **Calculate errors**
   ```
   spread_error = 50% - P(favorite covers)
   total_error = 50% - P(over total)
   ```

3. **Determine responsibility** for each quarter score
   ```
   For score 14-7 (21 total points, +7 margin):
   
   If full game is scoring too high:
     → This score is "responsible" (it's high-scoring)
     → Reduce its probability
     
   If full game is scoring too low:
     → This score helps (it's low-scoring)
     → Increase its probability
   ```

4. **Apply proportional adjustments** to ALL four quarters
   ```
   For each quarter Q in [Q1, Q2, Q3, Q4]:
     For each score in Q:
       total_factor = 1.0 + 0.12 × total_error × total_responsibility
       margin_factor = 1.0 + 0.12 × spread_error × margin_responsibility
       
       new_prob = old_prob × total_factor × margin_factor
   ```

5. **Renormalize** each quarter to sum to 100%

6. **Check convergence**
   ```
   If |spread_error| < 1% AND |total_error| < 1%:
     Done!
   ```

### Responsibility Calculation

**How much does a score contribute to the error?**

**For total points:**
```
score_total = 21 (for 14-7)
expected_quarter_total = 13.5 (23.5% of 58.5)

total_responsibility = (21 - 13.5) / 13.5 = +0.56

If total_error = +2% (scoring too high):
  Adjustment = -2% × 0.56 = -1.1%
  → Reduce probability of 14-7
```

**For spread:**
```
score_margin = +7 (favorite ahead)
expected_quarter_margin = 1.9 (25% of 7.5)

margin_responsibility = (7 - 1.9) / 1.9 = +2.7

If spread_error = -2% (favorite covering too often):
  Adjustment = -2% × 2.7 = -5.4%
  → Reduce probability of 14-7
```

**Combined effect:**
```
new_prob = old_prob × (1 - 1.1%) × (1 - 5.4%)
         = old_prob × 0.989 × 0.946
         = old_prob × 0.935
         
If 14-7 was 4%, it becomes 4% × 0.935 = 3.74%
```

### Adaptive Sample Sizes

To speed up calibration:

**Early iterations (rough adjustment):**
```
Iterations 1-3: 2,000 simulations (fast, ~1 second)
Iterations 4-6: 3,000 simulations (medium, ~1.5 seconds)
Iterations 7-10: 5,000 simulations (accurate, ~2.5 seconds)
```

**Why:** Early iterations make big changes, don't need high precision. Final iterations fine-tune, need accuracy.

### Example Calibration Run

```
Target: 48-52% for spread=-7.5, total=58.5

Iter 1: Cover=54.2%, Over=48.1% (2000 sims)
Iter 2: Cover=52.3%, Over=49.4% (2000 sims)  
Iter 3: Cover=51.1%, Over=50.2% (2000 sims)
Iter 4: Cover=50.8%, Over=49.8% (3000 sims)
Iter 5: Cover=50.3%, Over=50.1% (3000 sims)
✓ Converged at iteration 5
  Final Cover: 50.3%
  Final Over: 50.1%
```

Total time: ~8 seconds

---

## Worked Example

### Setup

**Game:** Alabama (-10.5) vs Auburn  
**Total:** 54.5  
**Historical data:** 7,042 games

### Step 1: Calculate Features

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
- feature_total = 1.982 → Moderate scoring (typical is ~2.0)
- feature_margin = 0.382 → Solid favorite 

### Step 2: Q1 Base Predictions

**Score 7-0 (most common):**
```
Empirical: 775/7042 = 11.0%
Model parameters: β₀=-2.18, β₁=+0.42, β₂=+0.31

logit = -2.18 + 0.42×1.982 + 0.31×0.382
      = -2.18 + 0.833 + 0.118  
      = -1.229

probability = 1 / (1 + e^1.229) = 22.6% (before normalization)
```

**Score 14-0 (favorite blowout):**
```
Empirical: 460/7042 = 6.53%
Model parameters: β₀=-2.60, β₁=+0.83, β₂=+1.12

logit = -2.60 + 0.83×1.982 + 1.12×0.382
      = -2.60 + 1.645 + 0.428
      = -0.527

probability = 1 / (1 + e^0.527) = 37.1% (before normalization)
```

Notice: 14-0 gets a huge boost from high margin coefficient (+1.12).

**Score 7-7 (tie):**
```
Empirical: 528/7042 = 7.50%
Model parameters: β₀=-2.48, β₁=+0.31, β₂=-0.52

logit = -2.48 + 0.31×1.982 + (-0.52)×0.382
      = -2.48 + 0.614 - 0.199
      = -2.065

probability = 1 / (1 + e^2.065) = 11.3% (before normalization)
```

Notice: 7-7 gets penalized by negative margin coefficient (-0.52).

**After normalizing all 80+ Q1 scores:**
```
7-0  → 10.2%
14-0 → 8.9%  (up from 6.5% empirical due to large spread)
7-7  → 6.4%  (down from 7.5% empirical due to large spread)
```

### Step 3: First Simulation

**Run 1 of 5,000:**

1. Sample Q1: Draw random number 0.327 → Falls in 7-0 bucket (10.2%)
   ```
   Q1 = 7-0
   Q1 category: "low_close_fav" (7 points, +7 margin)
   ```

2. Adjust Q2 probabilities based on Q1:
   ```
   Base P(7-7) = 6.8%
   P(7-7|low_close_fav) = 8.2% (from historical data)
   
   Adjusted P(7-7) = 0.7×6.8% + 0.3×8.2% = 4.76% + 2.46% = 7.22%
   ```
   
   Sample Q2: Draw 0.512 → Falls in 7-7 bucket
   ```
   Q2 = 7-7
   H1 = 7-0 + 7-7 = 14-7 (21 points, +7 margin)
   H1 category: "med_close_fav"
   ```

3. Adjust Q3 probabilities based on H1:
   ```
   Sample Q3: Gets 7-3
   Q3 = 7-3
   Q3 category: "low_close_fav"
   ```

4. Adjust Q4 probabilities based on H1 and Q3:
   ```
   Sample Q4: Gets 7-7
   Q4 = 7-7
   ```

5. Calculate full game:
   ```
   Full game = 14-7 + 7-3 + 7-7 = 28-17
   Alabama wins by 11 (covers 10.5)
   Total = 45 (under 54.5)
   ```

**Run 2 of 5,000:** Different random numbers → Different outcome  
**...continue for all 5,000 runs...**

**Results after 5,000 simulations:**
```
31-24: 3.2% (160 times)
28-21: 2.9% (145 times)
34-27: 2.7% (135 times)
...

P(Alabama covers -10.5) = 54.2%
P(Over 54.5) = 48.1%
```

### Step 4: Calibration

**Iteration 1:**
```
Error: Cover = 54.2% (target 50%), error = +4.2%
Error: Over = 48.1% (target 50%), error = -1.9%

Adjustments needed:
  - Reduce probabilities of favorite-heavy scores
  - Increase probabilities of high-scoring games
```

**Adjusting Q1 score 14-0:**
```
Current: 8.9%
total_responsibility = (14+0-13.5)/13.5 = +0.037 (slightly high-scoring)
margin_responsibility = (14-0-2.6)/2.6 = +4.38 (very favorite-heavy)

total_factor = 1.0 + 0.12×(-1.9%)×(+0.037) = 1.0 - 0.0084 = 0.992
margin_factor = 1.0 + 0.12×(+4.2%)×(+4.38) = 1.0 + 0.022 = 1.022

new_prob = 8.9% × 0.992 × 1.022 = 9.0%

Wait, that increased it! But total_factor reduced it by 0.8% and margin_factor 
would increase by 2.2%, net = +1.4%... Actually this is wrong because 
spread_error is +4.2% meaning we're covering TOO MUCH, so we need to 
REDUCE favorite-heavy scores.

Let me recalculate:
spread_error = 0.50 - 0.542 = -0.042 (negative because covering too much)

margin_factor = 1.0 + 0.12×(-0.042)×(+4.38) = 1.0 - 0.022 = 0.978

new_prob = 8.9% × 0.992 × 0.978 = 8.6%
```

**After adjusting all quarters and re-simulating:**
```
Iteration 2: Cover=52.1%, Over=49.5%
Iteration 3: Cover=50.7%, Over=50.2%
Iteration 4: Cover=50.2%, Over=50.0%
✓ Converged!
```

### Step 5: Final Output

**Q1 predictions (calibrated):**
```
7-0   → 10.1%
14-0  → 8.4%  (reduced from 8.9% due to calibration)
7-7   → 6.5%  (increased slightly)
3-0   → 4.2%
0-0   → 4.0%
```

**Full game predictions (after 5,000 simulations with correlation):**
```
31-24 → 3.1%  (Alabama by 7, total 55)
28-21 → 2.8%  (Alabama by 7, total 49)
34-27 → 2.6%  (Alabama by 7, total 61)
38-24 → 2.5%  (Alabama by 14, total 62)
28-17 → 2.4%  (Alabama by 11, total 45)
```

**Market checks:**
```
P(Alabama -10.5) = 50.1% ✓
P(Over 54.5) = 50.0% ✓
```

---

## Key Takeaways

### Why Orthogonal Features Matter

**Problem:** Spread and total are correlated (r ≈ -0.3)  
**Solution:** Transform to uncorrelated features
- feature_total: Overall scoring level
- feature_margin: Game lopsidedness

**Benefit:** Stable, interpretable coefficients

### Why Two-Tier Modeling Works

**Anchor scores (≥50 games):** Individual models with reliable estimates  
**Rare scores (<50 games):** Bucket models that borrow information

**Benefit:** Handles data sparsity without overfitting

### Why Correlation Matters

**Independence assumption:** P(game) = P(Q1) × P(Q2) × P(Q3) × P(Q4)  
**Reality:** Quarters are correlated (r ≈ 0.15)

**Our approach:** P(game) = P(Q1) × P(Q2|Q1) × P(Q3|H1) × P(Q4|H1,Q3)

**Benefit:** Captures realistic game flow

### Why Monte Carlo Simulation

**Alternative:** Multiply all combinations (7,000 Q1 scores × 7,000 Q2 scores × ...)  
**Problem:** Computationally impossible and can't model conditional dependence

**Monte Carlo:** Sample representative outcomes, count frequencies

**Benefit:** Fast, accurate, handles correlations naturally

### Why Calibration Is Essential

**Problem:** Quarter models trained independently may not combine to match Vegas  
**Solution:** Iteratively adjust quarter distributions

**Method:** Proportional responsibility-weighted updates  
**Benefit:** Final distribution matches market while maintaining realistic quarters

### Current Limitations

**Not modeled:**
- First possession (who gets ball in Q1 and Q3)
- Coaching tendencies (aggressive vs conservative)
- Weather conditions
- Injuries or lineup changes
- Pace of play differences

---

## Mathematical Notation Reference

| Symbol | Meaning | Example |
|--------|---------|---------|
| P(A) | Probability of A | P(rain) = 30% |
| P(A\|B) | Probability of A given B | P(Q2=7-0\|Q1=7-0) = 12% |
| r | Correlation coefficient | r = 0.15 |
| β | Regression coefficient | β₁ = 0.83 |
| e | Euler's number | e ≈ 2.718 |
| exp(x) | e raised to power x | exp(2) ≈ 7.389 |
| log(x) | Natural logarithm | log(e) = 1 |
| sigmoid(x) | 1 / (1 + e^(-x)) | sigmoid(0) = 0.5 |
| Σ | Summation | Σ(1,2,3) = 6 |
| \|x\| | Absolute value | \|-5\| = 5 |

---

## Glossary

**Anchor Score:** A score that appears frequently enough (≥50 times) to warrant its own individual model

**Bayesian Updating:** Adjusting probabilities based on new information (e.g., what happened in Q1)

**Calibration:** Adjusting model outputs to match known benchmarks (Vegas lines)

**Conditional Probability:** P(A|B) - probability of A happening given that B already happened

**Correlation:** Statistical relationship between two variables (r between -1 and +1)

**Empirical Distribution:** Probability distribution derived from observed historical frequencies

**Feature Engineering:** Transforming raw inputs into more useful predictive features

**Logistic Regression:** Statistical model for predicting binary outcomes using a sigmoid function

**Monte Carlo Simulation:** Running many random trials to estimate probability distributions

**Multicollinearity:** When predictor variables are correlated, causing unstable estimates

**Orthogonal:** Mathematically independent (correlation = 0)

**Regularization:** Constraining model parameters to prevent overfitting

**Responsibility:** How much a particular outcome contributes to prediction error

**Sigmoid Function:** S-shaped curve that maps any number to a probability (0-1)

---
