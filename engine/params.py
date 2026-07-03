from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional
import re

import pandas as pd
from openpyxl import Workbook

from engine.markets import MARKETS


@dataclass
class Band:
    start: int
    end: int
    annual: float            # engine-side amounts are ANNUAL real currency
    label: str = ""
    category: str = "strict"  # strict | lifestyle | gifts (income bands ignore it)


@dataclass
class Lump:
    age: int
    amount: float            # annual-mode amount; positive inflow, negative outflow
    label: str = ""
    category: str = "strict"


@dataclass
class Property:
    start_age: int
    initial_value: float
    rent_annual: float
    growth_mean: float
    growth_sd: float
    label: str = ""


def aggregate_schedule(bands: List[Band]) -> Callable[[int], float]:
    """Return a schedule function that adds all (start, end, amount) bands."""

    def _fn(age: int) -> float:
        return sum(b.annual for b in bands if b.start <= age <= b.end)

    return _fn


@dataclass
class SimulationParams:
    start_age: int
    end_age: int
    initial_portfolio: float
    real_return_mean: float
    real_return_sd: float
    fat_tails_df: Optional[int]      # None => normal distribution
    annual: bool = True
    n_paths: int = 10_000
    random_seed: Optional[int] = 42
    spending_bands: List[Band] = field(default_factory=list)
    income_bands: List[Band] = field(default_factory=list)
    lumps: List[Lump] = field(default_factory=list)
    properties: List[Property] = field(default_factory=list)
    scenario_name: str = ""          # replaces the input_file hack in figures

    def spending_fn(self) -> Callable[[int], float]:
        return aggregate_schedule(self.spending_bands)

    def income_fn(self) -> Callable[[int], float]:
        return aggregate_schedule(self.income_bands)

    @classmethod
    def from_scenario(cls, scenario: dict,
                       playground_events: list[dict] | None = None) -> "SimulationParams":
        """Build from the scenario JSON (PRD §4.1). Converts monthly->annual
        (x12), resolves market mu/sigma from engine.markets, and appends
        playground events as Lumps (category 'strict', never scaled by
        guardrails)."""
        portfolio = scenario["portfolio"]
        market = MARKETS[portfolio["market"]]

        spending_bands = [
            Band(b["age_from"], b["age_to"], b["amount_monthly"] * 12,
                 b.get("label", ""), b.get("category", "strict"))
            for b in scenario.get("spending_bands", [])
        ]
        income_bands = [
            Band(b["age_from"], b["age_to"], b["amount_monthly"] * 12, b.get("label", ""))
            for b in scenario.get("income_bands", [])
        ]
        lumps = [
            Lump(l["age"], l["amount"], l.get("label", ""), l.get("category", "strict"))
            for l in scenario.get("lumps", [])
        ]
        for ev in playground_events or []:
            lumps.append(Lump(ev["age"], ev["amount"], ev.get("label", "Playground"), "strict"))
        properties = [
            Property(p["start_age"], p["initial_value"], p["rent_monthly"] * 12,
                     market["housing_mu"], market["housing_sigma"], p.get("label", ""))
            for p in scenario.get("properties", [])
        ]

        return cls(
            start_age=portfolio["start_age"],
            end_age=portfolio["end_age"],
            initial_portfolio=portfolio["initial_portfolio"],
            real_return_mean=market["mu"],
            real_return_sd=market["sigma"],
            fat_tails_df=portfolio["fat_tails_df"] if portfolio.get("fat_tails_enabled") else None,
            annual=portfolio.get("mode", "annual") == "annual",
            n_paths=portfolio.get("n_paths", 10_000),
            random_seed=portfolio.get("random_seed", 42),
            spending_bands=spending_bands,
            income_bands=income_bands,
            lumps=lumps,
            properties=properties,
            scenario_name=scenario.get("name", ""),
        )

    @classmethod
    def from_legacy_scenario_data(cls, scenario_data: dict, market: str = "IL",
                                   fat_tails_df: Optional[int] = 5,
                                   annual: bool = True) -> "SimulationParams":
        """Build from the dict shape produced by the CLI's read_scenario_data
        (raw xlsx column groups, pre-category-tag). market/fat_tails_df/annual
        default to the values that used to live in the deleted configuration.py."""
        m = MARKETS[market]

        spending_bands = [
            Band(row.spending_age_from, row.spending_age_to,
                 row.spending_amount_monthly * 12, row.spending_comment)
            for _, row in scenario_data["spending"].iterrows()
        ]
        for _, row in scenario_data["travel"].iterrows():
            if row.travel_amount_annual:
                spending_bands.append(Band(row.travel_age_from, row.travel_age_to,
                                           row.travel_amount_annual, row.travel_comment, "lifestyle"))
        income_bands = [
            Band(row.income_age_from, row.income_age_to,
                 row.income_amount_monthly * 12, row.income_comment)
            for _, row in scenario_data["income"].iterrows()
        ]
        lumps = [
            Lump(row.lump_age, row.lump_amount, row.lump_comment,
                 "gifts" if row.lump_amount < 0 else "strict")
            for _, row in scenario_data["lumps"].iterrows()
        ]
        properties = [
            Property(row.properties_age_from, row.properties_initial_value,
                     row.properties_rent_monthly * 12, m["housing_mu"], m["housing_sigma"],
                     row.properties_comment)
            for _, row in scenario_data["properties"].iterrows()
        ]

        return cls(
            start_age=scenario_data["start_age"],
            end_age=scenario_data["end_age"],
            initial_portfolio=scenario_data["initial_portfolio"],
            real_return_mean=m["mu"],
            real_return_sd=m["sigma"],
            fat_tails_df=fat_tails_df,
            annual=annual,
            spending_bands=spending_bands,
            income_bands=income_bands,
            lumps=lumps,
            properties=properties,
        )


# Category regex and helper for parsing comments from xlsx files
CATEGORY_RE = re.compile(r"^(.*?)\s*\[(strict|lifestyle|gifts)\]\s*$")

def split_category(comment: str, default: str) -> tuple[str, str]:
    """Parse a comment field into label and category."""
    m = CATEGORY_RE.match(str(comment or ""))
    if m:
        return m.group(1).strip(), m.group(2)
    return str(comment or "").strip(), default


def validate_scenario(scenario: dict) -> list[str]:
    """Validate a scenario according to PRD flow 2.3 and field rules table in §4.1.
    
    Returns a list of human-readable error strings (empty list = valid).
    """
    errors = []
    
    # Check portfolio ages
    portfolio = scenario["portfolio"]
    start_age = portfolio["start_age"]
    end_age = portfolio["end_age"]
    
    if not (18 <= start_age < end_age <= 110):
        errors.append(f"Portfolio ages must satisfy: 18 <= start_age < end_age <= 110. "
                      f"Got start_age={start_age}, end_age={end_age}")
    
    # Check spending bands
    spending_bands = scenario.get("spending_bands", [])
    if not spending_bands:
        errors.append("Spending bands must be non-empty")
    
    # Check all bands in spending_bands and income_bands
    for i, band in enumerate(spending_bands):
        if band["age_from"] > band["age_to"]:
            errors.append(f"Spending band {i+1} has age_from > age_to: "
                          f"{band['age_from']} > {band['age_to']}. "
                          f"Field name: age_from")
    
    # Check income bands as well
    income_bands = scenario.get("income_bands", [])
    for i, band in enumerate(income_bands):
        if band["age_from"] > band["age_to"]:
            errors.append(f"Income band {i+1} has age_from > age_to: "
                          f"{band['age_from']} > {band['age_to']}. "
                          f"Field name: age_from")
    
    return errors


def scenario_to_xlsx(scenario: dict, path) -> None:
    """Convert a scenario dict to xlsx format with the exact column order specified in PRD."""
    
    # Extract portfolio data
    portfolio = scenario["portfolio"]
    
    # Build column groups with proper padding to make all lists same length
    spending_rows = []
    income_rows = []
    lump_rows = []
    properties_rows = []
    
    # Process spending bands
    for i, band in enumerate(scenario.get("spending_bands", [])):
        label = band.get("label", "")
        category = band.get("category", "strict")
        if label:
            comment = f"{label} [{category}]"
        else:
            comment = f"[{category}]"
            
        spending_rows.append({
            "spending_age_from": band["age_from"],
            "spending_age_to": band["age_to"],
            "spending_amount_monthly": band["amount_monthly"],
            "spending_comment": comment
        })
    
    # Process income bands (no category in schema, so no tag)
    for i, band in enumerate(scenario.get("income_bands", [])):
        label = band.get("label", "")
        comment = label  # Income bands never tagged
        
        income_rows.append({
            "income_age_from": band["age_from"],
            "income_age_to": band["age_to"],
            "income_amount_monthly": band["amount_monthly"],
            "income_comment": comment
        })
    
    # Process lumps
    for i, lump in enumerate(scenario.get("lumps", [])):
        label = lump.get("label", "")
        category = lump.get("category", "strict")
        if label:
            comment = f"{label} [{category}]"
        else:
            comment = f"[{category}]"
            
        lump_rows.append({
            "lump_age": lump["age"],
            "lump_amount": lump["amount"],
            "lump_comment": comment
        })
    
    # Process properties (no category in schema, so no tag)
    for i, prop in enumerate(scenario.get("properties", [])):
        label = prop.get("label", "")
        comment = label  # Properties never tagged
        
        properties_rows.append({
            "properties_age_from": prop["start_age"],
            "properties_initial_value": prop["initial_value"],
            "properties_rent_monthly": prop["rent_monthly"],
            "properties_comment": comment
        })
    
    # Create dataframes for each group, pad to same length with NaN values
    max_spending = len(spending_rows)
    max_income = len(income_rows)
    max_lump = len(lump_rows)
    max_properties = len(properties_rows)
    
    # Create lists with proper padding
    spending_data = spending_rows + [{"spending_age_from": None, "spending_age_to": None,
                                       "spending_amount_monthly": None, "spending_comment": None}] * (max(max_spending, max_income, max_lump, max_properties) - max_spending)
    
    income_data = income_rows + [{"income_age_from": None, "income_age_to": None,
                                  "income_amount_monthly": None, "income_comment": None}] * (max(max_spending, max_income, max_lump, max_properties) - max_income)
    
    lump_data = lump_rows + [{"lump_age": None, "lump_amount": None,
                              "lump_comment": None}] * (max(max_spending, max_income, max_lump, max_properties) - max_lump)
    
    properties_data = properties_rows + [{"properties_age_from": None, "properties_initial_value": None,
                                           "properties_rent_monthly": None, "properties_comment": None}] * (max(max_spending, max_income, max_lump, max_properties) - max_properties)
    
    # Create the full dataframe with all columns in exact order.
    # At least 1 row so the portfolio scalars (row 0) always survive a write,
    # even when every band/lump/property group is empty.
    n_rows = max(max_spending, max_income, max_lump, max_properties, 1)
    df_data = []
    for i in range(n_rows):
        row = {
            # scalars belong on row 0 only, matching the real xlsx format
            "initial_portfolio": portfolio["initial_portfolio"] if i == 0 else None,
            "start_age": portfolio["start_age"] if i == 0 else None,
            "end_age": portfolio["end_age"] if i == 0 else None,
            # Add the new portfolio scalar columns right after end_age
            "market": portfolio["market"] if i == 0 else None,
            "mode": portfolio["mode"] if i == 0 else None,
            "fat_tails_enabled": portfolio["fat_tails_enabled"] if i == 0 else None,
            "fat_tails_df": portfolio["fat_tails_df"] if i == 0 else None,
            "n_paths": portfolio["n_paths"] if i == 0 else None,
            "random_seed": portfolio["random_seed"] if i == 0 else None,
            "spending_age_from": spending_data[i]["spending_age_from"] if i < len(spending_data) else None,
            "spending_age_to": spending_data[i]["spending_age_to"] if i < len(spending_data) else None,
            "spending_amount_monthly": spending_data[i]["spending_amount_monthly"] if i < len(spending_data) else None,
            "spending_comment": spending_data[i]["spending_comment"] if i < len(spending_data) else None,
            "income_age_from": income_data[i]["income_age_from"] if i < len(income_data) else None,
            "income_age_to": income_data[i]["income_age_to"] if i < len(income_data) else None,
            "income_amount_monthly": income_data[i]["income_amount_monthly"] if i < len(income_data) else None,
            "income_comment": income_data[i]["income_comment"] if i < len(income_data) else None,
            "lump_age": lump_data[i]["lump_age"] if i < len(lump_data) else None,
            "lump_amount": lump_data[i]["lump_amount"] if i < len(lump_data) else None,
            "lump_comment": lump_data[i]["lump_comment"] if i < len(lump_data) else None,
            "properties_age_from": properties_data[i]["properties_age_from"] if i < len(properties_data) else None,
            "properties_initial_value": properties_data[i]["properties_initial_value"] if i < len(properties_data) else None,
            "properties_rent_monthly": properties_data[i]["properties_rent_monthly"] if i < len(properties_data) else None,
            "properties_comment": properties_data[i]["properties_comment"] if i < len(properties_data) else None,
            "travel_age_from": None,  # Always empty for travel group
            "travel_age_to": None,
            "travel_amount_annual": None,
            "travel_comment": None
        }
        df_data.append(row)
    
    # Create the dataframe and write to Excel
    df = pd.DataFrame(df_data)
    
    # Use openpyxl to write the file with proper formatting
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)


def scenario_from_xlsx(path) -> dict:
    """Convert xlsx data back to scenario dict format."""
    
    # Read the Excel file
    df = pd.read_excel(path, engine="openpyxl")
    
    # Extract portfolio data from first row
    portfolio = {
        "initial_portfolio": float(df.iloc[0]["initial_portfolio"]),
        "start_age": int(df.iloc[0]["start_age"]),
        "end_age": int(df.iloc[0]["end_age"]),
        "market": df.iloc[0]["market"] if "market" in df.columns and pd.notna(df.iloc[0]["market"]) else "IL",
        "fat_tails_enabled": bool(df.iloc[0]["fat_tails_enabled"]) if "fat_tails_enabled" in df.columns and pd.notna(df.iloc[0]["fat_tails_enabled"]) else True,
        "fat_tails_df": int(df.iloc[0]["fat_tails_df"]) if "fat_tails_df" in df.columns and pd.notna(df.iloc[0]["fat_tails_df"]) else 5,
        "mode": df.iloc[0]["mode"] if "mode" in df.columns and pd.notna(df.iloc[0]["mode"]) else "annual",
        "n_paths": int(df.iloc[0]["n_paths"]) if "n_paths" in df.columns and pd.notna(df.iloc[0]["n_paths"]) else 10000,
        "random_seed": int(df.iloc[0]["random_seed"]) if "random_seed" in df.columns and pd.notna(df.iloc[0]["random_seed"]) else 42
    }
    
    # Build the scenario structure
    scenario = {
        "$schema": "scenario.v1",
        "name": "",  # Default as per spec - naming from filename is caller's job
        "portfolio": portfolio,
        "spending_bands": [],
        "income_bands": [],
        "lumps": [],
        "properties": []
    }
    
    # Process spending bands (including travel rows that were converted from travel group)
    spending_rows = df[df["spending_age_from"].notna()]
    
    for _, row in spending_rows.iterrows():
        # Parse comment to extract label and category
        comment = row.get("spending_comment")
        if pd.isna(comment):
            label, category = "", "strict"  # Default for spending
        else:
            label, category = split_category(comment, "strict")
            
        # Create spending band
        scenario["spending_bands"].append({
            "id": f"sb-{len(scenario['spending_bands']) + 1}",
            "age_from": int(row["spending_age_from"]),
            "age_to": int(row["spending_age_to"]),
            "amount_monthly": float(row["spending_amount_monthly"]),
            "label": label,
            "category": category
        })
    
    # Process travel rows: migrate into spending_bands (annual -> monthly), continuing the sb- counter
    travel_rows = df[df["travel_age_from"].notna()]

    for _, row in travel_rows.iterrows():
        comment = row.get("travel_comment")
        if pd.isna(comment):
            label, category = "", "lifestyle"
        else:
            label, category = split_category(comment, "lifestyle")

        scenario["spending_bands"].append({
            "id": f"sb-{len(scenario['spending_bands']) + 1}",
            "age_from": int(row["travel_age_from"]),
            "age_to": int(row["travel_age_to"]),
            "amount_monthly": float(row["travel_amount_annual"]) / 12,
            "label": label,
            "category": category
        })

    # Process income bands (no category in schema, so no tag parsing)
    income_rows = df[df["income_age_from"].notna()]
    
    for _, row in income_rows.iterrows():
        comment = row.get("income_comment")
        if pd.isna(comment):
            label = ""
        else:
            label = str(comment)
            
        # Create income band
        scenario["income_bands"].append({
            "id": f"ib-{len(scenario['income_bands']) + 1}",
            "age_from": int(row["income_age_from"]),
            "age_to": int(row["income_age_to"]),
            "amount_monthly": float(row["income_amount_monthly"]),
            "label": label
        })
    
    # Process lumps (with category parsing)
    lump_rows = df[df["lump_age"].notna()]
    
    for _, row in lump_rows.iterrows():
        comment = row.get("lump_comment")
        default_category = "gifts" if row["lump_amount"] < 0 else "strict"
        if pd.isna(comment):
            label, category = "", default_category
        else:
            label, category = split_category(comment, default_category)
            
        # Create lump
        scenario["lumps"].append({
            "id": f"lp-{len(scenario['lumps']) + 1}",
            "age": int(row["lump_age"]),
            "amount": float(row["lump_amount"]),
            "label": label,
            "category": category
        })
    
    # Process properties (no category in schema, so no tag parsing)
    properties_rows = df[df["properties_age_from"].notna()]
    
    for _, row in properties_rows.iterrows():
        comment = row.get("properties_comment")
        if pd.isna(comment):
            label = ""
        else:
            label = str(comment)
            
        # Create property
        scenario["properties"].append({
            "id": f"pr-{len(scenario['properties']) + 1}",
            "start_age": int(row["properties_age_from"]),
            "initial_value": float(row["properties_initial_value"]),
            "rent_monthly": float(row["properties_rent_monthly"]),  # already monthly, no conversion
            "label": label
        })
    
    return scenario