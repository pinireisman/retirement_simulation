from dataclasses import dataclass
from typing import Callable, List, Optional
from configuration import housing_growth_mean_by_market, MARKET, ANNUAL, housing_growth_sd_by_market, portfolio_real_return_mean_by_market, \
    portfolio_real_return_sd_by_market


###############################################################################
# Core data structures
###############################################################################

@dataclass
class Band:
    start: int
    end: int
    annual: float
    label: str


@dataclass
class Lump:
    age: int
    amount: float  # positive=inflow, negative=outflow
    label: str = ""  # description of the lump


@dataclass
class Property:
    """Real‑estate asset kept outside the liquid portfolio."""
    start_age: int  # first year property exists
    initial_value: float  # market value at start_age (real ₪)
    rent_annual: float  # net rent added to cash‑flow each year ≥ start_age
    growth_mean: float = housing_growth_mean_by_market[MARKET]  # long‑run real appreciation µ
    growth_sd: float = housing_growth_sd_by_market[MARKET]  # annual real volatility σ
    label: str = ""


###############################################################################
# Utility: build age→amount functions where overlapping bands are summed.
###############################################################################

def aggregate_schedule(bands: List[Band]) -> Callable[[int], float]:
    """Return a schedule function that adds all (start, end, amount) bands."""

    def _fn(age: int) -> float:
        return sum(b.annual for b in bands if b.start <= age <= b.end)

    return _fn


@dataclass
class SimulationParams:

    def __init__(self, scenario_data: dict):

        # Horizon --------------------------------------------------------------
        self.start_age: int = scenario_data['start_age']
        self.end_age: int = scenario_data['end_age']
        self.annual = ANNUAL  # True if annual, False if monthly

        # Portfolio ------------------------------------------------------------
        self.initial_portfolio: float = scenario_data['initial_portfolio']
        self.real_return_mean: float = portfolio_real_return_mean_by_market[MARKET]
        self.real_return_sd: float = portfolio_real_return_sd_by_market[MARKET]

        # Monte‑Carlo ----------------------------------------------------------
        self.n_paths: int = 10_000
        self.random_seed: Optional[int] = 42  # some random seed for reproducibility

        # Core spending (real) -------------------------------------------------
        self.spending_bands: List[Band] = []
        for i, row in scenario_data['spending'].iterrows():
            self.spending_bands.append(Band(row.spending_age_from, row.spending_age_to,
                                            row.spending_amount_monthly * 12,
                                            row.spending_comment))

        # Extra travel allowance --------------------------------

        self.travel_annual: float = scenario_data['travel'].travel_amount_annual.iloc[
            0]  # this is beyond 40K that is in the spending_bands
        self.travel_annual_start: int = scenario_data['travel'].travel_age_from.iloc[0]
        self.travel_annual_end: int = scenario_data['travel'].travel_age_to.iloc[0]
        if self.travel_annual:
            self.spending_bands.append(Band(self.travel_annual_start,
                                            self.travel_annual_end,
                                            self.travel_annual,
                                            "extra travel allowance"))

        # Income bands (real); overlaps add up -------------------------------
        self.income_bands: List[Band] = []
        # age is husband age.
        for i, row in scenario_data['income'].iterrows():
            self.income_bands.append(Band(row.income_age_from, row.income_age_to,
                                          row.income_amount_monthly * 12,
                                          row.income_comment))

        # One‑off lumps --------------------------------------------------------
        self.lumps: List[Lump] = []
        for i, row in scenario_data['lumps'].iterrows():
            self.lumps.append(Lump(age=row.lump_age, amount=row.lump_amount, label=row.lump_comment))

        # Property list --------------------------------------------------------
        self.properties: List[Property] = []
        for i, row in scenario_data['properties'].iterrows():
            self.properties.append(
                Property(
                    start_age=row.properties_age_from,
                    initial_value=row.properties_initial_value,
                    rent_annual=row.properties_rent_monthly * 12,
                    label=row.properties_comment
                )
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def spending_fn(self) -> Callable[[int], float]:
        return aggregate_schedule(self.spending_bands)

    def income_fn(self) -> Callable[[int], float]:
        return aggregate_schedule(self.income_bands)
