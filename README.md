# Retirement Monte-Carlo Simulator

A Monte-Carlo simulation tool for retirement financial planning, working in **real (inflation-adjusted) shekels (₪)**. It models your portfolio, income, spending, properties, and one-off lump sums across thousands of randomized market scenarios to estimate your probability of financial ruin and the expected range of outcomes.

---

## Setup

### 1. Create the virtual environment and install dependencies

```bash
./setup_venv.sh
```

This creates a `.venv/` directory, upgrades pip, and installs everything from `requirements.txt`. You only need to run this once (or again after adding new dependencies).

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

Your prompt will change to show `(.venv)`. All subsequent `python` / `pip` commands in that shell will use the isolated environment.

### 3. VS Code

Open the workspace and VS Code will automatically use `.venv/bin/python` (configured in `.vscode/settings.json`). The launch configuration already points to this interpreter, so debugging works out of the box.

---

## How to Run

```bash
python simlulation_main.py                          # uses scenario_data_example.xlsx
python simlulation_main.py my_scenario.xlsx         # uses a custom file
python simlulation_main.py my_scenario.xlsx --plot  # also shows the interactive chart
```

---

## Input File Structure (Excel `.xlsx`)

All input is provided in a **single flat Excel sheet**. Each section (spending, income, lumps, properties, travel) uses its own set of columns. Rows are independent per section — you can have more rows in one section than another; unused cells in other sections are simply ignored (left blank).

The file is read from row 2 onwards (row 1 is the header). Only the first row is used for the global parameters (`initial_portfolio`, `start_age`, `end_age`).

---

### 🔵 Global Parameters
These are read only from **row 1** of the sheet.

| Column | Description | Example |
|---|---|---|
| `initial_portfolio` | Starting liquid portfolio value in real ₪ | `8,000,000` |
| `start_age` | Husband's age at the start of the simulation | `50` |
| `end_age` | Age at end of simulation (planning horizon) | `95` |

---

### 🟢 Spending Bands
Define how much you spend per month in each life phase. Multiple bands can overlap — they are **summed** for each age.

| Column | Description | Example |
|---|---|---|
| `spending_age_from` | Age when this spending band starts | `50` |
| `spending_age_to` | Age when this spending band ends (inclusive) | `60` |
| `spending_amount_monthly` | Monthly spending amount in real ₪ | `25,000` |
| `spending_comment` | Description of this phase | `kids at home` |

**Example from `scenario_data_example.xlsx`:**
- Ages 50–60: ₪25,000/month (*kids at home*)
- Ages 61–80: ₪15,000/month (*empty nest*)
- Ages 81–95: ₪22,000/month (*diur mugan*)

---

### 🟢 Travel Allowance
Extra travel spending layered on top of base spending. Defined per age range — multiple bands can be added for different life phases.

| Column | Description | Example |
|---|---|---|
| `travel_age_from` | Age when this travel allowance starts | `50` |
| `travel_age_to` | Age when this travel allowance ends (inclusive) | `80` |
| `travel_amount_annual` | Annual travel amount in real ₪ | `20,000` |
| `travel_comment` | Description | `travel on top of regular expenses` |

> Travel bands are internally merged into the spending schedule, so they appear as a separate labeled bar in the chart.

---

### 🟢 Income Bands
All income sources (salaries, pensions, rent, Bituach Leumi, etc.). Multiple bands are **summed** per age.

| Column | Description | Example |
|---|---|---|
| `income_age_from` | Age when this income starts | `50` |
| `income_age_to` | Age when this income ends (inclusive) | `61` |
| `income_amount_monthly` | Monthly income amount in real ₪ | `15,000` |
| `income_comment` | Description of this income source | `salary wife` |

**Example from `scenario_data_example.xlsx`:**
- Ages 50–61: ₪15,000/month (*salary wife*)
- Ages 50–67: ₪15,000/month (*salary husband*)
- Ages 62–95: ₪8,000/month (*pension wife*)
- Ages 67–95: ₪15,600/month (*pension husband*)
- Ages 62–95: ₪2,700/month (*Bituach Leumi wife*)
- Ages 67–95: ₪2,700/month (*Bituach Leumi husband*)
- Ages 80–95: ₪7,000/month (*rent apt*)

---

### 🟡 Lump Sums
One-off cash events at a specific age. Positive = inflow (receiving money), Negative = outflow (spending/gifting money).

| Column | Description | Example |
|---|---|---|
| `lump_age` | Husband's age when this event occurs | `65` |
| `lump_amount` | Amount in real ₪ (positive=inflow, negative=outflow) | `-1,000,000` |
| `lump_comment` | Description | `gift child2 turns 28` |

**Example from `scenario_data_example.xlsx`:**
- Age 51: −₪700,000 (*build 3 rental units*)
- Age 51: +₪100,000 (*RSU 2026 with 20% downside*)
- Age 62: −₪1,000,000 (*gift child1 turns 28*)
- Age 65: −₪1,000,000 (*gift child2 turns 28*)
- Age 67: +₪1,100,000 (*tax-free Gemel withdrawal*)
- Age 80: −₪1,500,000 (*Diur Mugan deposit*)
- Age 95: +₪600,000 (*Diur Mugan refund*)

---

### 🟣 Properties
Real-estate assets held **outside** the liquid portfolio. They grow independently and contribute rental income. Their value is tracked separately and added to the total estate.

| Column | Description | Example |
|---|---|---|
| `properties_age_from` | Age when this property enters the simulation | `50` |
| `properties_initial_value` | Market value in real ₪ at that age | `5,000,000` |
| `properties_rent_monthly` | Monthly net rental income in real ₪ | `8,333` |
| `properties_comment` | Description | `3 rental units` |

> Properties have their own stochastic growth model (separate from the portfolio), using market-specific real appreciation mean and volatility defined in `configuration.py`.

**Example from `scenario_data_example.xlsx`:**
- Age 50: Apartment worth ₪5M, no rent (*apt 1, pre-rental*)
- Age 65: 3 rental units worth ₪1.5M, yielding ₪8,333/month

---

## Simulation Parameters (`configuration.py`)

These are global settings that control how the Monte-Carlo engine runs. Edit `configuration.py` to change them.

| Parameter | Description | Default |
|---|---|---|
| `MARKET` | Market preset: `"IL"`, `"US"`, or `"UK"`. Controls return and volatility assumptions. | `"IL"` |
| `ANNUAL` | `True` = annual simulation steps, `False` = monthly steps | `True` |
| `FAT_TAIL_DF` | Degrees of freedom for the Student-t return distribution. Lower = fatter tails (more crashes). Use `None` for normal distribution. `5` is moderate fat tails, `3` is extreme, `10` is near-normal. | `5` |

### Market Return Assumptions

| Market | Portfolio Mean Real Return | Portfolio Std Dev | Property Mean Real Growth | Property Std Dev |
|---|---|---|---|---|
| IL (Israel) | 5.0% | 16.0% | 1.8% | 8.0% |
| US | 3.0% | 12.0% | 1.0% | 6.0% |
| UK | 5.3% | 20.0% | 2.4% | 7.0% |

### Monte-Carlo Engine Settings (in `simulation_params.py`)

| Parameter | Description | Default |
|---|---|---|
| `n_paths` | Number of simulated life paths | `10,000` |
| `random_seed` | Random seed for reproducibility | `42` |

---

## Understanding the Output

### Console Output

```
Ruin probability: 4.231%

Tracks that ended in ruin:
  Path 42 ruined in year 78. Portfolio sequence: [...]
  Path 317 ruined in year 85. Portfolio sequence: [...]

Distribution of ruined paths per year (sorted):
  Year 74: 1 paths ruined
  Year 78: 3 paths ruined
  ...
```

- **Ruin probability**: The percentage of the 10,000 simulated paths where the liquid portfolio hit ₪0.
- **Ruined tracks**: Each path that ended in ruin is listed with the age at which it went to zero, and the full year-by-year portfolio value sequence leading up to ruin.
- **Ruin distribution by year**: How many paths ruined in each specific year — helps you understand *when* the most vulnerable period is.

---

### Interactive Chart

The `--plot` flag opens a two-panel interactive Plotly chart in your browser.

#### Panel 1 — Cash-Flow Breakdown (bar chart)
Shows the year-by-year breakdown of all cash flows by category:
- 🟢 **Green bars** = income sources (salary, pension, Bituach Leumi, etc.)
- 🔵 **Blue bars** = rental income from properties
- 🔴 **Red bars** = spending (base + travel, each as a separate bar)
- ⬜ **Grey bars** = lump sums (positive or negative)
- **Black line** = net cash flow (positive = portfolio is growing from income, negative = drawing down)

#### Panel 2 — Portfolio & Estate Over Time (line chart)
Shows the distribution of portfolio and estate outcomes across all 10,000 paths:
- **Solid blue line** = median portfolio value
- **Purple line** = median property value
- **Dotted black line** = total median estate (portfolio + property)
- **Blue percentile lines** = 5th, 25th, 75th, 95th percentile portfolio paths
- **Dark shaded band** = 25th–75th percentile range (middle 50% of outcomes)
- **Light blue band** = 5th–95th percentile range (middle 90% of outcomes)

#### Ruin Probability Icon (top-right corner)
An **ℹ️ icon** with a colored label shows the assessment of your ruin probability:

| Ruin Range | Color | Title |
|---|---|---|
| 0–1% | 🟢 Green | Too defensive / ultra-safe |
| 1–3% | 🟢 Green | Super safe |
| 3–5% | 🟢 Green | Safe |
| 5–10% | 🟠 Orange | Reasonable only with flexibility |
| 10–15% | 🔴 Red | Risky but maybe manageable |
| 15–25% | 🔴 Red | Risky |
| >25% | 🔴 Red | Speculative / underfunded |

Hovering over the icon shows a detailed explanation of what the ruin probability means for your plan.

---

## Key Concepts

- **All values are in real (inflation-adjusted) shekels** — you don't need to account for inflation in your inputs. A pension of ₪15,000/month means ₪15,000 in today's purchasing power every month, forever.
- **Ages refer to the husband's age** throughout the simulation.
- **Ruin** is defined as the liquid portfolio reaching ₪0. Properties are tracked separately and are not liquidated.
- **Overlapping bands are additive** — if two income bands overlap the same age, both amounts are received. Same for spending bands.
- The simulation uses a **Student-t distribution** for returns (when `FAT_TAIL_DF` is set) to model fat-tail crash scenarios more realistically than a normal distribution.

