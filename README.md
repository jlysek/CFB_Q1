# College Football Quarter Scoring Prediction Model

A statistical model that predicts first quarter score probability distributions in college football games using regularized multinomial logistic regression trained on historical data and pregame betting markets.

Initial motivation to build was a landing rate edge case in which I suspected DraftKings 1Q pricing of 7-7, 10-10, 14-14... were too thin.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [Installation & Setup](#installation--setup)
- [Next Steps](#next-steps)
- [Technical Details](#technical-details)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)

---

## How It Works

### The Statistical Approach

**Goal:**  
`P(Q1 Score | Pregame Spread & Total)`

**Method:** Regularized Multinomial Logistic Regression  
- Train separate models for each possible Q1 score  
- Use pregame spread and total as predictive features  
- Apply adaptive L2 regularization to prevent overfitting rare scores  
- Regularization strength varies inversely with score frequency  

**Data:** 6,000+ FBS games from 2014–present with:  
- Quarter-by-quarter scoring  
- Pregame closing spreads and totals  
- Team and game context  

For detailed methodology, see [Modeling.md](Modeling.md)

---

## Project Structure

### Core Files

| File | Description |
|------|--------------|
| **Q1.py** | Multinomial logistic regression implementation and score probability prediction engine |
| **scraper.py** | Data collection from CollegeFootballData.com API. Populates MySQL with games, scores, and betting lines. Incremental updates for new games every week |
| **prediction_server.py** | Flask API server that serves predictions via REST endpoints and proxies CFBD API calls |
| **CFBInterface.html** | Web-based interface for all games in upcoming week with SGP engine based on probability distribution |
| **requirements.txt** | Python dependencies |
| **Modeling.md** | Statistical methodology documentation |

### File Structure

CFB_Q1/
├── Q1.py # Core prediction model
├── scraper.py # Data collection
├── prediction_server.py # Flask API server
├── CFBInterface.html # Web interface
├── requirements.txt # Python dependencies
├── .env # Configuration (create this)
├── README.md # This file
└── Modeling.md # Statistical methodology

markdown
Copy code

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

### `cfb.quarter_scoring`

Quarter-by-quarter scoring (4 rows per game).

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | Integer | Links to games table |
| `quarter` | Integer | Quarter number (1–4) |
| `home_score` | Integer | Points scored by home team this quarter |
| `away_score` | Integer | Points scored by away team this quarter |
| `total_score` | Integer | Computed (`home_score + away_score`) |

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
Step 2: Set Up MySQL Database
Create the Database

sql
Copy code
CREATE DATABASE cfb;
The tables will be created automatically when you run the scraper for the first time.

Note Your MySQL Credentials
You'll need:

scss
Copy code
Host (usually 127.0.0.1 or localhost)
Port (usually 3306)
Username (usually root)
Password (what you set during MySQL installation)
Step 3: Install Python Dependencies
bash
Copy code
pip install -r requirements.txt
Step 4: Configure Environment Variables
Create a file named .env in the project root directory with the following content:

bash
Copy code
# College Football Data API Key
CFBD_API_KEY=your_api_key_here

# MySQL Database Configuration
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password_here
DB_NAME=cfb
Step 5: Load Historical Data
Run the scraper to populate your database with historical game data. This will take 10–15 minutes for the initial load:

bash
Copy code
python scraper.py
The scraper will:

Create the necessary database tables

Fetch all FBS games since 2014

Download quarter-by-quarter scoring data

Pull pregame betting lines

You should see progress updates in the terminal. When complete, you'll see a summary showing the total number of games and quarters loaded.

Step 6: Start the Prediction Server
bash
Copy code
python prediction_server.py
Step 7: Access the Web Interface
Open your browser and go to:

bash
Copy code
http://localhost:5000/interface
You should now see the CFB Quarter Predictor interface with:

Current week's games loaded from CFBD API

Manual prediction input fields

SGP builder and betting markets

Troubleshooting
<details> <summary><strong>Database connection error</strong></summary>
Verify MySQL is running
Check your DB_PASSWORD in the .env file
Ensure the cfb database exists

</details> <details> <summary><strong>API authentication failed</strong></summary>
Verify your CFBD_API_KEY in the .env file
Get a new API key from collegefootballdata.com

</details> <details> <summary><strong>"Module not found" errors</strong></summary>
Run pip install -r requirements.txt again
Ensure you're using Python 3.8 or higher: python --version

</details> <details> <summary><strong>Port already in use</strong></summary>
Another application is using port 5000
Change the port in prediction_server.py (line with port=5000)

</details> <details> <summary><strong>Model initialization is slow</strong></summary>
The first run takes 30–60 seconds to load and fit the model
This is normal — subsequent predictions will be fast
Check that you have 6000+ games loaded in the database

</details>
Next Steps
Future development roadmap:

Quantify first possession value – Analyze how much getting the ball first matters based on spread and total

Coach configurations – Incorporate coaching tendencies into predictions

Expand to other quarters – Extend model to Q2, Q3, and Q4 predictions

Improve the modeling – Refine regularization, feature engineering, and prediction algorithms

Technical Details
Regularization Strategy
Adaptive L2 penalty prevents overfitting:
λ(score) = base_λ × (1 / √count)
Rare scores get stronger regularization, keeping predictions reasonable.

Normalization
Features are standardized before modeling:
normalized = (value - mean) / std_dev
This ensures spread and total contribute proportionally.

Distance-Based Smoothing
Extremely rare scores borrow information from similar scores based on margin and total similarity.

Disclaimer
This project is just a baseline starting point and should not be used as the complete truth. It doesn't take into account coaching configurations and the data is only as good as the API.
Basically every line of code was written by Claude with my guidance as my CS knowledge is very weak (as you can probably already tell).

“All models are wrong, but some are useful.” – George E. P. Box

Contributing
This is an active project. Questions, suggestions, and bug reports are greatly appreciated.
Contact: lysek.jarrett@gmail.com