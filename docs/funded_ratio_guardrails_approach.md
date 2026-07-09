
# funded-ratio guardrails

Each year in each path, calculate:

```text
funded_ratio = current_portfolio / PV(remaining net spending needs)
```

Where:

```text

PV stands for Present Value

remaining net spending needs =
    essential spending
  + child gifts / one-off expenses
  + parent support
  + discretionary spending
  - salary
  - all_other_income
```

Then apply guardrails.

Example:

```text
if funded_ratio < lower_guardrail:
    cut discretionary spending

if funded_ratio > upper_guardrail:
    allow spending increase

otherwise:
    keep spending unchanged in real terms
```

This is extremely cheap computationally. It adds almost no cost to your existing 10,000-path simulation.

---

# Split spending into fixed and flexible parts

Do not adjust the whole spending path equally.

Use something like:

```text
fixed_spending[t] =
    essential household spending
  + fixed family commitments
  + required housing/health costs
  + committed child gifts
  - reliable income

flex_spending[t] =
    travel
  + lifestyle extras
  + optional help
  + discretionary upgrades
```

Then each Monte Carlo path gets its own spending multiplier:

```python
spending_multiplier[path]
```

At the beginning:

```python
spending_multiplier[:] = 1.0
```

Then in each year, each path may cut or raise its own multiplier.

Actual portfolio withdrawal becomes:

```text
withdrawal[t] =
    fixed_spending[t]
  + spending_multiplier[path] * flex_spending[t]
```

This gives you path-specific spending without nested Monte Carlo.

---

# The key formula

Precompute the present value of the remaining fixed and flexible spending paths:

```python
PV_fixed[t] = PV of fixed_spending from year t onward
PV_flex[t]  = PV of flex_spending from year t onward
```

Then for each simulated path:

```python
PV_need = PV_fixed[t] + spending_multiplier[path] * PV_flex[t]

funded_ratio = portfolio[path] / PV_need
```

If the funded ratio is too low, reduce the multiplier.

If it is very high, increase the multiplier.

---

# How to calculate the spending adjustment

Suppose your current funded ratio is below the lower guardrail.

You want to move toward a target funded ratio:

```text
target_funded_ratio = 1.00 or 1.05 or 1.10
```

Solve:

```text
current_portfolio / new_PV_need = target_funded_ratio
```

So:

```text
new_PV_need = current_portfolio / target_funded_ratio
```

Since:

```text
new_PV_need = PV_fixed + new_multiplier * PV_flex
```

You get:

```text
new_multiplier =
    (current_portfolio / target_funded_ratio - PV_fixed)
    / PV_flex
```

That is the whole adjustment formula.

No Monte Carlo inside Monte Carlo.

---

# Vectorized pseudocode

```python
import numpy as np

def precompute_pv(cashflows, real_discount_rate):
    T = len(cashflows)
    pv = np.zeros(T)

    for t in range(T):
        years = np.arange(T - t)
        discount = (1 + real_discount_rate) ** years
        pv[t] = np.sum(cashflows[t:] / discount)

    return pv


def simulate_with_guardrails(
    returns,                  # shape: [n_paths, T], real returns
    initial_portfolio,
    fixed_spending,            # real NIS, net of reliable income if you prefer
    flex_spending,             # real NIS discretionary path
    real_discount_rate=0.01,
    fr_lower=0.85,
    fr_target=1.05,
    fr_upper=1.30,
    adjustment_fraction=0.25,
    max_cut_per_year=0.10,
    max_raise_per_year=0.10,
    min_multiplier=0.40,
    max_multiplier=1.50,
):
    n_paths, T = returns.shape

    portfolio = np.full(n_paths, initial_portfolio, dtype=float)
    multiplier = np.ones(n_paths, dtype=float)

    portfolio_history = np.zeros((n_paths, T + 1))
    spending_history = np.zeros((n_paths, T))
    multiplier_history = np.zeros((n_paths, T))

    portfolio_history[:, 0] = portfolio

    pv_fixed = precompute_pv(fixed_spending, real_discount_rate)
    pv_flex = precompute_pv(flex_spending, real_discount_rate)

    for t in range(T):
        # Current PV of remaining planned spending
        pv_need = pv_fixed[t] + multiplier * pv_flex[t]

        # Avoid division by zero
        pv_need = np.maximum(pv_need, 1.0)

        funded_ratio = portfolio / pv_need

        cut_mask = funded_ratio < fr_lower
        raise_mask = funded_ratio > fr_upper
        active_mask = cut_mask | raise_mask

        # Spending multiplier that would move us toward target funded ratio
        target_multiplier = (
            (portfolio / fr_target - pv_fixed[t])
            / max(pv_flex[t], 1.0)
        )

        target_multiplier = np.clip(
            target_multiplier,
            min_multiplier,
            max_multiplier
        )

        # Partial adjustment, not full jump
        proposed_multiplier = (
            multiplier
            + adjustment_fraction * (target_multiplier - multiplier)
        )

        # Cap annual cuts and raises
        proposed_multiplier = np.where(
            cut_mask,
            np.maximum(proposed_multiplier, multiplier * (1 - max_cut_per_year)),
            proposed_multiplier
        )

        proposed_multiplier = np.where(
            raise_mask,
            np.minimum(proposed_multiplier, multiplier * (1 + max_raise_per_year)),
            proposed_multiplier
        )

        # Apply only if a guardrail was hit
        multiplier = np.where(active_mask, proposed_multiplier, multiplier)

        # Compute this year's spending
        withdrawal = fixed_spending[t] + multiplier * flex_spending[t]

        spending_history[:, t] = withdrawal
        multiplier_history[:, t] = multiplier

        # Start-of-year withdrawal convention
        portfolio = portfolio - withdrawal

        # Portfolio cannot go below zero
        portfolio = np.maximum(portfolio, 0)

        # Apply real return
        portfolio = portfolio * (1 + returns[:, t])

        portfolio_history[:, t + 1] = portfolio

    return portfolio_history, spending_history, multiplier_history
```

---

# How to set the thresholds

I would not start with probability-of-success thresholds inside the simulation. I would translate them into funded-ratio thresholds.

A reasonable first version:

```text
fr_lower  = 0.85
fr_target = 1.05
fr_upper  = 1.30
```

Meaning:

```text
If current assets are less than 85% of the PV of remaining planned spending:
    cut discretionary spending.

If assets are around 105% of remaining needs:
    plan is roughly on track.

If assets exceed 130% of remaining needs:
    allow a raise.
```

But the exact numbers depend heavily on the real discount rate used.

If you use a low/conservative real discount rate, say:

```text
real_discount_rate = 0% to 1.5%
```

then funded-ratio thresholds around 0.85 / 1.05 / 1.30 are plausible.

If you use a high expected-return discount rate, say:

```text
real_discount_rate = 3% to 4%
```

then the PV of future needs will look artificially cheap, so you would need higher funded-ratio thresholds.

For your model, I would probably use:

```text
real_discount_rate = 1%
fr_lower = 0.85
fr_target = 1.05
fr_upper = 1.25 to 1.35
```

Then test whether the resulting behavior is acceptable.

---

# Better: calibrate the funded-ratio thresholds from your Monte Carlo

You can still keep things efficient.

Run separate calibration jobs **once**, not inside each path.

For each retirement year `t`, estimate the portfolio value needed to support the remaining spending path with different success probabilities:

```text
W_80[t] = wealth needed for 80% success from year t onward
W_95[t] = wealth needed for 95% success from year t onward
W_99[t] = wealth needed for 99% success from year t onward
```

Then convert those to funded-ratio thresholds:

```text
fr_lower[t]  = W_80[t] / PV_need_at_baseline[t]
fr_target[t] = W_95[t] / PV_need_at_baseline[t]
fr_upper[t]  = W_99[t] / PV_need_at_baseline[t]
```

Then inside the main simulation you just use table lookups:

```python
fr_lower_t = fr_lower[t]
fr_target_t = fr_target[t]
fr_upper_t = fr_upper[t]
```

This gives you risk-based guardrails without nested simulation.

The computational cost becomes something like:

```text
40 years × 3 thresholds × binary search
```

rather than:

```text
10,000 paths × 40 years × inner Monte Carlo
```

That is a massive difference.

---

# Even simpler alternative: safe-spending-rate table

Another good option is to precompute a table like this:

```text
age / remaining horizon → safe withdrawal rate
```

Example:

```text
remaining years | 80% success | 95% success | 99% success
40              | 4.3%        | 3.4%        | 2.9%
35              | 4.6%        | 3.7%        | 3.1%
30              | 5.0%        | 4.1%        | 3.5%
25              | 5.6%        | 4.7%        | 4.0%
```

Then in each path/year:

```python
current_withdrawal_rate = current_withdrawal / current_portfolio
```

Guardrail logic:

```text
if current_withdrawal_rate > WR_80:
    spending is too risky → cut

if current_withdrawal_rate < WR_99:
    spending is very safe → raise

else:
    no change
```

Adjustment:

```text
target withdrawal = portfolio * WR_95
```

This is very fast and intuitive.

But for your specific situation, I prefer funded ratio because you have uneven future cash flows: salary, consulting, pension assets, child gifts, changing household expenses, possible rent, and later-life spending changes.

---

# Important modeling point

You should not ask each path:

```text
“What is my true probability of success from here?”
```

Because that requires nested simulation.

Instead, define a rule such as:

```text
“When my funded ratio falls below X, I cut spending.”
```

Then your outer Monte Carlo tells you:

```text
How often did I hit the lower guardrail?
How large were the cuts?
How often did I recover?
What was the final wealth distribution?
What was the actual ruin probability?
What was the 10th percentile spending path?
```

That is the right structure.

The Monte Carlo evaluates the policy; it does not need to recalculate the policy with another Monte Carlo inside each path.

---

# What I would implement first

For your simulator, I would implement this version:

```text
1. Split spending into fixed and flexible buckets.
2. Use one spending multiplier per path.
3. Calculate funded ratio each year.
4. Cut/raise only the flexible bucket.
5. Use annual adjustment caps.
6. Track spending-path percentiles, not only final wealth.
```

Initial parameters:

```text
real_discount_rate = 1%

fr_lower  = 0.85
fr_target = 1.05
fr_upper  = 1.30

adjustment_fraction = 25%

max_cut_per_year = 10%
max_raise_per_year = 5% to 10%

min_multiplier = 0.40 to 0.60
max_multiplier = 1.25 to 1.50
```

Then evaluate:

```text
probability of ruin
probability of hitting lower guardrail
median number of cuts
95th percentile number of cuts
worst 5th percentile spending level
median ending wealth
5th percentile ending wealth
```

That will tell you whether the guardrail strategy is financially safe **and** psychologically tolerable.



Since the plan includes planned expenses (e.g. gifts for the girls, simulated disaster events, car upgrades, weddings) - which are singular inceases of spending (one time events). Obviously - in these years there will be a temporary spike in draw on the portfolio... 

This is exactly why we prefer the **funded-ratio / PV-liability approach** over a simple “current withdrawal rate” guardrail.

A year with a 1M NIS gift, wedding, car replacement, or disaster should **not automatically look like a bad retirement-spending year**. It is a balance-sheet event, not necessarily a lifestyle overspending event.

The solution is to split spending into categories.

---

## 1. Separate spending into 4 buckets

Instead of one spending array, use something like:

```text
essential_recurring[t]
discretionary_recurring[t]
planned_lumpy_fixed[t]
planned_lumpy_flexible[t]
```

Examples:

```text
essential_recurring:
    food, housing, health, utilities, basic transport

discretionary_recurring:
    travel, restaurants, optional lifestyle

planned_lumpy_fixed:
    committed gifts to daughters
    minimum wedding support
    required car replacement
    known housing repairs

planned_lumpy_flexible:
    bigger wedding help
    nicer car upgrade
    extra gifts
    optional family trips
```

Then random disaster events are separate:

```text
random_shock[path, t]
```

---

## 2. Planned one-time events should be included in the PV calculation

The adjustment formula **does take them into account** if you include them in the future liability stream.

Instead of:

```text
PV_need = PV_fixed + multiplier * PV_flex
```

use:

```text
PV_need =
    PV_essential_recurring
  + PV_planned_lumpy_fixed
  + multiplier * PV_discretionary_recurring
  + lumpy_multiplier * PV_planned_lumpy_flexible
```

Then the guardrail already knows, for example:

```text
In 7 years there is a 1,000,000 NIS gift.
In 10 years there is a car replacement.
In 12 years there may be a wedding contribution.
```

So the simulator does **not** panic in that year. The spending spike was already funded as a known future liability.

---

## 3. The revised adjustment formula

Previously:

```text
new_multiplier =
    (portfolio / target_funded_ratio - PV_fixed)
    / PV_flex
```

Now:

```text
new_multiplier =
    (
      portfolio / target_funded_ratio
      - PV_essential_recurring
      - PV_planned_lumpy_fixed
      - lumpy_multiplier * PV_planned_lumpy_flexible
    )
    / PV_discretionary_recurring
```

Or more simply:

```text
new_discretionary_multiplier =
    available_for_lifestyle / PV_discretionary_recurring
```

Where:

```text
available_for_lifestyle =
    portfolio / target_funded_ratio
  - committed_future_liabilities
```

And:

```text
committed_future_liabilities =
    PV_essential_recurring
  + PV_planned_lumpy_fixed
```

This means the regular lifestyle is adjusted **after reserving for known commitments**.

---

## 4. Do not adjust spending because of the event-year spike itself

Suppose in year 8 you give one daughter 1M NIS.

A bad rule would say:

```text
withdrawal_rate = this_year_withdrawal / portfolio
```

That may look huge and trigger a cut.

A better rule says:

```text
Was this expense already planned?
If yes, it was already included in PV liabilities.
Do not treat it as lifestyle overspending.
```

So the guardrail should be based on:

```text
funded_ratio = portfolio / PV_remaining_needs
```

not:

```text
current_year_withdrawal / portfolio
```

The one-off expense affects the plan because portfolio drops, but the remaining liability also drops because the obligation has now been paid. Those two effects mostly offset if the event was already planned correctly.

---

## 5. Equivalent implementation trick: reserve upcoming lumpy expenses

For very large planned events, you can model a separate “reserved liability bucket.”

Example:

```text
portfolio_total = 9M NIS

reserve_for_known_lumpy_expenses =
    daughter gift due in 3 years
  + car replacement due in 2 years
  + wedding reserve
```

Then guardrails operate on:

```text
risk_portfolio = portfolio_total - reserve_for_known_lumpy_expenses
```

And spending decisions are based on whether the **remaining lifestyle portfolio** is sufficiently funded.

This is psychologically cleaner too:

```text
The 1M gift is not part of my lifestyle budget anymore.
It is already earmarked.
```

In code, the PV approach and reserve-bucket approach are basically two versions of the same idea.

---

## 6. How to treat random disaster events

Random shocks are different.

Examples:

```text
large medical cost
major home repair
supporting family member
legal/tax surprise
car accident
unexpected income loss
```

These are not known future liabilities. There are two good ways to model them.

### Option A: event hits portfolio, then guardrails adjust future spending

This is the simplest and probably best.

Inside each path:

```text
if disaster occurs in year t:
    portfolio -= disaster_amount
```

Then calculate the funded ratio **after the shock**.

If the shock materially reduces the funded ratio, cut future discretionary spending.

But do not try to “pay for the disaster” by retroactively cutting the same year’s normal spending unless that is realistic.

So the rule is:

```text
planned one-offs:
    included in PV from the start

random shocks:
    hit portfolio when they happen
    then may trigger future spending cuts
```

---

### Option B: include a probabilistic disaster reserve

You can also add an expected shock reserve:

```text
PV_disaster_reserve[t] =
    expected PV of future random shocks
```

For example, if there is a 5% annual chance of a 300k NIS event:

```text
expected annual shock cost = 0.05 * 300k = 15k NIS
```

Then add that to the fixed liability stream.

But be careful: expected value understates tail risk. For retirement modeling, I would usually simulate the shocks directly and maybe add a modest reserve.

---

## 7. Recommended treatment by expense type

| Expense                            | Treatment                                                     |
| ---------------------------------- | ------------------------------------------------------------- |
| Gifts to daughters                 | Planned lumpy fixed, or split into minimum + optional         |
| Weddings                           | Planned lumpy flexible unless you consider it committed       |
| Car replacement                    | Planned lumpy semi-flexible; can downsize or delay            |
| Home renovation                    | Planned lumpy flexible unless required                        |
| Disaster events                    | Random shocks; reduce portfolio and trigger future guardrails |
| Parent support                     | Fixed recurring if committed; stochastic if uncertain         |
| Travel                             | Discretionary recurring                                       |
| Assisted living / later-life costs | Fixed recurring scenario or stochastic health-cost scenario   |

For your own model, I would split each daughter-related item into:

```text
committed_floor
optional_extra
```

Example:

```text
daughter_gift_fixed = 700k
daughter_gift_optional = 300k
```

Then if the plan is stressed, you do not necessarily remove the gift entirely. You first reduce the optional layer.

---

## 8. Guardrail order of cuts

I would not use one multiplier for everything. Use a hierarchy:

```text
1. Reduce discretionary recurring spending
2. Reduce optional lumpy spending
3. Delay flexible lumpy spending, such as car upgrades
4. Reduce committed gifts only if funded ratio is deeply stressed
5. Essential spending is protected
```

Example guardrail behavior:

```text
funded_ratio > 1.30:
    allow lifestyle raise
    allow optional lumpy expenses

0.90 to 1.30:
    continue as planned

0.80 to 0.90:
    reduce travel/lifestyle multiplier

0.70 to 0.80:
    reduce lifestyle and optional lumpy expenses

< 0.70:
    consider reducing or delaying even semi-committed lumpy expenses
```

That makes the model more realistic.

---

## 9. A better formula with multiple flexible layers

Let:

```text
PV_committed[t] =
    PV essential recurring
  + PV fixed lumpy
  - PV reliable income

PV_lifestyle[t] =
    PV discretionary recurring

PV_optional_lumpy[t] =
    PV optional gifts / car upgrades / wedding extras
```

Each path has:

```text
lifestyle_multiplier[path]
optional_lumpy_multiplier[path]
```

Then:

```text
PV_need =
    PV_committed[t]
  + lifestyle_multiplier * PV_lifestyle[t]
  + optional_lumpy_multiplier * PV_optional_lumpy[t]
```

When the lower guardrail is hit, first solve for the lifestyle multiplier:

```text
target_lifestyle_multiplier =
    (
      portfolio / target_funded_ratio
      - PV_committed[t]
      - optional_lumpy_multiplier * PV_optional_lumpy[t]
    )
    / PV_lifestyle[t]
```

If that is not enough, reduce optional lumpy spending too.

---

## 10. Practical pseudocode

```python
# yearly arrays, real NIS
essential = ...
discretionary = ...
fixed_lumpy = ...
optional_lumpy = ...
reliable_income = ...

committed_cashflow = essential + fixed_lumpy - reliable_income
lifestyle_cashflow = discretionary
optional_lumpy_cashflow = optional_lumpy

PV_committed = precompute_pv(committed_cashflow, real_discount_rate)
PV_lifestyle = precompute_pv(lifestyle_cashflow, real_discount_rate)
PV_optional_lumpy = precompute_pv(optional_lumpy_cashflow, real_discount_rate)

lifestyle_mult = np.ones(n_paths)
optional_lumpy_mult = np.ones(n_paths)

for t in range(T):

    # Random disaster, if any
    portfolio -= random_shock[:, t]
    portfolio = np.maximum(portfolio, 0)

    pv_need = (
        PV_committed[t]
        + lifestyle_mult * PV_lifestyle[t]
        + optional_lumpy_mult * PV_optional_lumpy[t]
    )

    funded_ratio = portfolio / np.maximum(pv_need, 1)

    cut_mask = funded_ratio < fr_lower
    raise_mask = funded_ratio > fr_upper

    # First adjust lifestyle
    target_lifestyle_mult = (
        portfolio / fr_target
        - PV_committed[t]
        - optional_lumpy_mult * PV_optional_lumpy[t]
    ) / max(PV_lifestyle[t], 1)

    target_lifestyle_mult = np.clip(
        target_lifestyle_mult,
        min_lifestyle_mult,
        max_lifestyle_mult
    )

    lifestyle_mult = np.where(
        cut_mask | raise_mask,
        lifestyle_mult + adjustment_fraction * (target_lifestyle_mult - lifestyle_mult),
        lifestyle_mult
    )

    # Cap annual lifestyle changes
    lifestyle_mult = np.where(
        cut_mask,
        np.maximum(lifestyle_mult, lifestyle_mult * (1 - max_cut_per_year)),
        lifestyle_mult
    )

    lifestyle_mult = np.where(
        raise_mask,
        np.minimum(lifestyle_mult, lifestyle_mult * (1 + max_raise_per_year)),
        lifestyle_mult
    )

    # If still under severe stress, reduce optional lumpy spending
    severe_mask = funded_ratio < fr_severe

    optional_lumpy_mult = np.where(
        severe_mask,
        optional_lumpy_mult * 0.90,
        optional_lumpy_mult
    )

    # Actual spending this year
    withdrawal = (
        essential[t]
        + lifestyle_mult * discretionary[t]
        + fixed_lumpy[t]
        + optional_lumpy_mult * optional_lumpy[t]
        - reliable_income[t]
    )

    portfolio -= withdrawal
    portfolio = np.maximum(portfolio, 0)

    portfolio *= (1 + returns[:, t])
```

One caution: the annual cut/raise capping should use the previous multiplier as a stored value. The pseudocode above shows the concept, but in actual code I would store `prev_lifestyle_mult` before modifying it.

---

## 11. My recommendation for your simulator

Use this structure:

```text
Essential recurring spending:
    protected

Discretionary recurring spending:
    primary guardrail adjustment variable

Planned fixed lumpy expenses:
    included in PV, not adjusted

Planned optional lumpy expenses:
    included in PV, but can be reduced/delayed under stress

Random disaster events:
    simulated directly, hit portfolio, then affect future guardrails
```

For your daughters, I would model gifts/weddings like this:

```text
fixed_daughter_support:
    amount you feel morally/financially committed to

optional_daughter_support:
    amount you would like to give if the plan is healthy
```

For cars:

```text
baseline_car_replacement:
    fixed lumpy

upgrade_car_budget:
    optional lumpy
```

Then the guardrail does not overreact to known spikes, but it still responds correctly if those spikes plus poor market returns leave the remaining plan underfunded.
