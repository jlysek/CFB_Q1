# College Football Quarter Scoring Prediction Model

A statistical model that predicts first quarter score probability distributions in college football games using independent binary logistic regression with adaptive regularization and market calibration.

**Initial motivation:** Landing rate edge case analysis on DraftKings 1Q tie markets (7-7, 10-10, 14-14), which appeared mispriced.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Next Steps](#next-steps)
- [Technical Details](#technical-details)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)

---

## How It Works

### The Statistical Approach

**Goal:**  
`P(Q1 Score | Pregame Spread & Total)`

**Method:** Independent Binary Logistic Regression (One-vs-All) with Market Calibration

**Three-Layer Architecture:**

1. **Model Layer:** Independent binary logistic regression for each score
   - Each score gets its own model: `P(this_score | spread, total)`
   - Features: spread, total, spread², total² (normalized)
   - Adaptive L2 regularization based on score frequency

2. **Normalization Layer:** Ensure probabilities sum to 1
   - Models trained independently don't naturally sum to 1
   - Normalize raw predictions to valid probability distribution

3. **Calibration Layer:** Blend with market data
   - Use 2D interpolation on market probability files
   - Blend model predictions with market probabilities
   - Weight blending by distance to calibration points

**Regularization:** Adaptive penalty prevents overfitting
- Rare scores (few occurrences) → strong regularization → stay close to baseline
- Common scores (many occurrences) → weak regularization → learn from data
- Formula: `λ(score) = 0.1 × (1 + 1/max(count(score), 5))`

**Data:** 7,000+ FBS games from 2014–present with:
- Quarter-by-quarter scoring
- Pregame closing spreads and totals
- Team classifications

For detailed methodology, see [Modeling.md](Modeling.md)

---

## Project Structure

### Core Files

| File | Description |
|------|-------------|
| **Q1.py** | Core prediction engine with independent binary logistic regression and market calibration |
| **scraper.py** | Data collection from CollegeFootballData.com API. Populates MySQL with games, scores, and betting lines. Incremental updates for new games |
| **prediction_server.py** | Flask API server that serves predictions via REST endpoints and proxies CFBD API calls |
| **CFBInterface.html** | Web-based interface for viewing weekly games with SGP (same game parlay) builder based on probability distributions |
| **Calibration/** | Folder containing market probability CSV files (`<spread>_<total>.csv`) used for calibration layer |
| **requirements.txt** | Python dependencies |
| **Modeling.md** | Statistical methodology documentation |
| **.env** | Configuration file (create from template below) |

### Directory Structure

```
CFB_Q1/
├── Q1.py                      # Core prediction model
├── scraper.py                 # Data collection
├── prediction_server.py       # Flask API server
├── CFBInterface.html          # Web interface
├── Calibration/               # Market calibration data
│   ├── 3_49.csv
│   ├── 7.5_45.5.csv
│   ├── 10.5_46.5.csv
│   └── ...
├── requirements.txt           # Python dependencies
├── .env                       # Configuration (create this)
├── README.md                  # This file
└── Modeling.md                # Statistical methodology
```

---

## Database Schema

### `cfb.games`

Every FBS game since 2014 with final scores and betting information.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | Integer | Unique identifier from CFBD API |
| `season` | Integer | Year of season |
| `week` | Integer | Week number (0–15) |
| `home_team` | String | Home team name |
| `away_team` | String | Away team name |
| `home_points` | Integer | Final home score |
| `away_points` | Integer | Final away score |
| `neutral_site` | Integer | 1 if neutral site, 0 otherwise |
| `conference_game` | Integer | 1 if conference game, 0 otherwise |
| `pregame_spread` | Float | Closing spread (negative = home favored) |
| `pregame_total` | Float | Closing total (over/under) |
| `home_ml_prob` | Float | Home moneyline probability (2021+) |
| `away_ml_prob` | Float | Away moneyline probability (2021+) |
| `betting_provider` | String | Source of betting line |
| `home_classification` | String | Division (fbs/fcs) |
| `away_classification` | String | Division (fbs/fcs) |

### `cfb.quarter_scoring`

Quarter-by-quarter scoring (4 rows per game).

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `game_id` | Integer | Links to games table |
| `quarter` | Integer | Quarter number (1–4) |
| `home_score` | Integer | Points scored by home team this quarter |
| `away_score` | Integer | Points scored by away team this quarter |
| `total_score` | Integer | Computed (`home_score + away_score`) |
| `created_at` | Date | Record creation timestamp |

### `cfb.coach_first_possession`

*Note: Not yet implemented in current model version*

Tracking coach tendencies for receiving/deferring the opening kickoff.

---

## Installation & Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** – [Download Python](https://www.python.org/downloads/)
- **MySQL 8.0 or higher** – [Download MySQL](https://dev.mysql.com/downloads/mysql/)
- **College Football Data API Key** – [Get Free API Key](https://collegefootballdata.com/)

---

### Step 1: Download the Project

```bash
git clone https://github.com/jlysek/CFB_Q1.git
cd CFB_Q1
```

---

### Step 2: Set Up MySQL Database

Create the database:

```sql
CREATE DATABASE cfb;
```

The tables will be created automatically when you run the scraper for the first time.

**Note Your MySQL Credentials:**
- Host (usually `127.0.0.1` or `localhost`)
- Port (usually `3306`)
- Username (usually `root`)
- Password (what you set during MySQL installation)

---

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `mysql-connector-python` – Database connectivity
- `pandas` – Data manipulation
- `numpy` – Numerical computing
- `scipy` – Optimization algorithms
- `flask` – Web server
- `flask-cors` – Cross-origin resource sharing
- `python-dotenv` – Environment variable management
- `requests` – HTTP client

---

### Step 4: Configure Environment Variables

Create a file named `.env` in the project root directory:

```bash
# College Football Data API Key
CFBD_API_KEY=your_api_key_here

# MySQL Database Configuration
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password_here
DB_NAME=cfb
```

---

### Step 5: Load Historical Data

Run the scraper to populate your database with historical game data. This will take 10–15 minutes for the initial load:

```bash
python scraper.py
```

The scraper will:
- Create the necessary database tables
- Fetch all FBS games since 2014
- Download quarter-by-quarter scoring data
- Pull pregame betting lines

You should see progress updates in the terminal. When complete, you'll see a summary showing the total number of games and quarters loaded.

---

### Step 6: Create Calibration Folder

Create a folder for market calibration data:

```bash
mkdir Calibration
```

Add market probability CSV files with naming convention: `<spread>_<total>.csv`

**CSV format:**
```csv
Home Score,Away Score,Fair
0,0,0.0821
7,0,0.1243
0,7,0.0532
7,7,0.0687
...
```

The model will work without calibration data but will be more accurate with it.

---

### Step 7: Start the Prediction Server

```bash
python prediction_server.py
```

You should see:
```
Loading historical data...
Loaded 7042 historical games

Calculating empirical distribution...
[Empirical distribution output]

Fitting model...
[Model fitting progress]

Loading market calibration data...
Loaded 15 calibration datasets

MODEL READY
 * Running on http://127.0.0.1:5000
```

---

### Step 8: Access the Web Interface

Open your browser and go to:

```
http://localhost:5000/interface
```

You should now see the CFB Quarter Predictor interface with:
- Current week's games loaded from CFBD API
- Manual prediction input fields
- SGP (Same Game Parlay) builder
- Betting markets display

---

## Usage

### Command Line Interface

Run Q1.py directly for command-line predictions:

```bash
python Q1.py
```

**Interactive prompts:**
```
Enter spread and total: -7.5 58.5
```

**Output includes:**
- Probability adjustment analysis
- All scores with ≥0.5% probability
- 2-way and 3-way moneyline odds
- Spread markets at multiple lines
- Total markets at multiple lines
- Special draw markets

### Web Interface

1. **Weekly Games View:**
   - Automatically loads current week's games
   - Click any game to generate predictions

2. **Manual Input:**
   - Enter spread and total manually
   - Click "Generate Predictions"

3. **SGP Builder:**
   - Select multiple bets from the same game
   - Calculate parlay probability and fair odds
   - Compare to sportsbook offerings

### API Endpoints

**Get predictions:**
```bash
curl "http://localhost:5000/predict?spread=-7.5&total=58.5"
```

**Get weekly games:**
```bash
curl "http://localhost:5000/games/2024/10"
```

---

## Troubleshooting

<details>
<summary><strong>Database connection error</strong></summary>

- Verify MySQL is running: `mysql --version`
- Check your `DB_PASSWORD` in the `.env` file
- Ensure the `cfb` database exists: `mysql -u root -p -e "SHOW DATABASES;"`
</details>

<details>
<summary><strong>API authentication failed</strong></summary>

- Verify your `CFBD_API_KEY` in the `.env` file
- Get a new API key from [collegefootballdata.com](https://collegefootballdata.com)
- Check API key format (should be a long alphanumeric string)
</details>

<details>
<summary><strong>"Module not found" errors</strong></summary>

- Run `pip install -r requirements.txt` again
- Ensure you're using Python 3.8 or higher: `python --version`
- Try creating a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
</details>

<details>
<summary><strong>Port already in use</strong></summary>

- Another application is using port 5000
- Change the port in `prediction_server.py`:
  ```python
  app.run(debug=True, port=5001)  # Change to 5001
  ```
- Update the URL accordingly: `http://localhost:5001/interface`
</details>

<details>
<summary><strong>Model initialization is slow</strong></summary>

- The first run takes 30–60 seconds to load and fit the model
- This is normal – subsequent predictions will be fast
- Check that you have 6000+ games loaded: 
  ```sql
  SELECT COUNT(*) FROM cfb.games;
  ```
</details>

<details>
<summary><strong>Optimization failed errors</strong></summary>

- This can happen for very rare scores
- The model has fallback mechanisms
- Check that you have sufficient historical data
- Rare scores use distance-based heuristic instead
</details>

<details>
<summary><strong>Calibration files not loading</strong></summary>

- Check that `Calibration/` folder exists in project root
- Verify CSV file naming: `<spread>_<total>.csv` (e.g., `7.5_45.5.csv`)
- Check CSV format: Must have columns `Home Score`, `Away Score`, `Fair`
- The model will work without calibration but won't be as accurate
</details>

---

## Next Steps

Future development roadmap:

### High Priority

1. **First Possession Analysis**
   - Quantify value of receiving vs deferring kickoff
   - Incorporate `cfb.coach_first_possession` data
   - Adjust Q1 and Q3 spreads based on possession

2. **Coaching Tendencies**
   - Model offensive/defensive pace and style
   - Account for conservative vs aggressive play-calling
   - Incorporate timeout usage patterns

3. **Expand to Other Quarters**
   - Build Q2, Q3, Q4 models
   - Sequential dependency: condition later quarters on earlier results
   - Full game simulation

### Medium Priority

4. **Enhanced Features**
   - Team strength ratings (SP+, FPI, etc.)
   - Recent form and momentum indicators
   - Weather conditions (wind, temperature, precipitation)
   - Key player availability (injuries, suspensions)

5. **Improved Modeling**
   - Test alternative basis functions (splines, interactions)
   - Explore ensemble methods
   - Incorporate score correlation structure
   - Bayesian hierarchical models for rare scores

6. **Better Calibration**
   - Time-varying calibration (days before game)
   - Multiple sportsbook aggregation
   - Sharper calibration algorithms
   - Automated calibration data collection

### Low Priority

7. **User Interface Enhancements**
   - Save favorite teams/bets
   - Historical performance tracking
   - Bet simulation and backtesting
   - Mobile-responsive design improvements

8. **Infrastructure**
   - Automated daily data updates
   - Cloud deployment (AWS/GCP)
   - Database optimization for faster queries
   - Caching layer for common predictions

---

## Technical Details

### Model Architecture

**Independent Binary Logistic Regression (One-vs-All):**
- Each score has its own binary classifier
- Predicts: `P(this_score | spread, total)` vs `P(not_this_score | spread, total)`
- Models trained separately with individual regularization
- Raw probabilities normalized to sum to 1

**Features:**
```
logit(score) = β₀ + β₁×(spread/10) + β₂×((total-50)/20) 
               + β₃×(spread/10)² + β₄×((total-50)/20)²
```

Where:
- `spread`: Pregame point spread (negative = home favored)
- `total`: Pregame total points line
- Normalization ensures features contribute proportionally

### Regularization Strategy

**Adaptive L2 penalty prevents overfitting:**

```python
λ(score) = 0.1 × (1 + 1 / max(count(score), 5))

Loss = -LogLikelihood + λ × Σ(β - β_prior)²
```

**Effect:**
- Rare scores (≤10 occurrences): Strong regularization, stay close to baseline
- Common scores (>100 occurrences): Weak regularization, learn from data
- Prevents overfitting while allowing data-driven adjustments

### Market Calibration

**2D Inverse Distance Weighting (IDW):**

For each score at given (spread, total):
1. Find all calibration files containing this score
2. Calculate normalized distance to each:
   ```
   distance = sqrt((Δspread/10)² + (Δtotal/20)²)
   ```
3. Weight by inverse distance squared:
   ```
   weight_i = 1 / (distance_i² + ε)
   ```
4. Interpolate market probability:
   ```
   market_prob = Σ(weight_i × prob_i) / Σ(weight_i)
   ```

**Adaptive blending:**
```python
α = 0.7 × exp(-min_distance / 2.0)
final_prob = α × market_prob + (1-α) × model_prob
```

**Distance-based trust:**
- Exact match (distance=0): 70% market, 30% model
- Distance=1: 42% market, 58% model  
- Distance=2: 26% market, 74% model
- Distance=5+: ≈0% market, 100% model

### Distance-Based Smoothing

For scores without fitted models (too rare):

```python
expected_margin = -spread
expected_q1_total = total × 0.25

margin_distance = |actual_margin - expected_margin|
total_distance = |actual_total - expected_q1_total|

margin_factor = exp(-margin_distance / 10.0)
total_factor = exp(-total_distance / 5.0)

adjusted_prob = empirical_prob × margin_factor × total_factor
```

This borrows information from similar scores while maintaining reasonable predictions.

### Optimization

**L-BFGS-B with bounded search:**
- Primary: Limited-memory Broyden-Fletcher-Goldfarb-Shanno
- Fallback: Nelder-Mead (gradient-free)
- Bounds prevent extreme coefficients
- Typically converges in 20-50 iterations per score

---

## Contributing

This is an active project. Questions, suggestions, and bug reports are greatly appreciated.

### Contact

**Email:** lysek.jarrett@gmail.com

**GitHub:** [github.com/jlysek/CFB_Q1](https://github.com/jlysek/CFB_Q1)

---
