"""Thin CLI wrapper around engine/: preserves the old command-line flags."""
from __future__ import annotations
import argparse

import pandas as pd

from engine.params import SimulationParams
from engine.simulation import run_simulation, run_historic_scenario
from engine.figures import plot_cash_flow, plot_with_historic


def _read_numbers_as_df(path):
    from numbers_parser import Document
    doc = Document(path)
    table = doc.sheets[0].tables[0]
    rows = table.rows()
    headers = [cell.value for cell in rows[0]]
    data = [[cell.value for cell in row] for row in rows[1:]]
    return pd.DataFrame(data, columns=headers)


def read_scenario_data(path):
    print(f"Reading scenario data from {path}...")
    if str(path).lower().endswith('.numbers'):
        pdf = _read_numbers_as_df(path)
    else:
        pdf = pd.read_excel(path)
    scenario_data = {}
    scenario_data['initial_portfolio'] = pdf['initial_portfolio'].iloc[0]
    scenario_data['start_age'] = pdf['start_age'].iloc[0]
    scenario_data['end_age'] = pdf['end_age'].iloc[0]
    scenario_data['spending'] = pdf[['spending_age_from',
                                     'spending_age_to',
                                     'spending_amount_monthly',
                                     'spending_comment']].dropna()
    scenario_data['income'] = pdf[['income_age_from',
                                   'income_age_to',
                                   'income_amount_monthly',
                                   'income_comment']].dropna()
    scenario_data['lumps'] = pdf[['lump_age',
                                  'lump_amount',
                                  'lump_comment']].dropna()

    scenario_data['properties'] = pdf[['properties_age_from',
                                       'properties_initial_value',
                                       'properties_rent_monthly',
                                       'properties_comment']].dropna()
    scenario_data['travel'] = pdf[['travel_age_from',
                                   'travel_age_to',
                                   'travel_amount_annual',
                                   'travel_comment']].dropna()
    print(f"done reading. starting simulation")
    return scenario_data


def _format_historic_name(key: str) -> str:
    parts = key.split("_")
    year = parts[0]
    name = " ".join(word.capitalize() for word in parts[1:])
    return f"{name} ({year})"


def _cli():
    parser = argparse.ArgumentParser(description="Retirement Monte-Carlo simulator")
    parser.add_argument("input", type=str, nargs="?", default="./scenario_data_example.xlsx",
                        help="Path to the scenario data .xlsx file (default: ./scenario_data_example.xlsx)")

    parser.add_argument("--plot", action="store_true", help="Show interactive cash-flow plot")
    parser.add_argument("--historic", action="store_true",
                        help="Run historical return sequences and display portfolio trajectories below the main chart")
    args = parser.parse_args()

    scenario_data = read_scenario_data(args.input)
    params = SimulationParams.from_legacy_scenario_data(scenario_data)
    params.scenario_name = str(args.input)

    result = run_simulation(params)
    print("Ruin probability:", f"{result['summary']['ruin_probability']:.3%}")
    ruin_tracks_log = result.get("ruin_tracks_log", [])
    if ruin_tracks_log:
        print("\nTracks that ended in ruin:")
        for track in ruin_tracks_log:
            if 'ruin_year' in track:
                print(f"Path {track['path_index']} ruined in year {track['ruin_year']}. Portfolio sequence: {track['portfolio_sequence']}")
            elif 'ruin_month' in track:
                print(f"Path {track['path_index']} ruined in month {track['ruin_month']}. Portfolio sequence: {track['portfolio_sequence']}")
        ruin_time_key = 'ruin_year' if 'ruin_year' in ruin_tracks_log[0] else 'ruin_month'
        ruin_time_counts = {}
        for track in ruin_tracks_log:
            time = track[ruin_time_key]
            ruin_time_counts[time] = ruin_time_counts.get(time, 0) + 1
        print("\nDistribution of ruined paths per {} (sorted):".format('year' if ruin_time_key == 'ruin_year' else 'month'))
        for time in sorted(ruin_time_counts):
            print(f"{ruin_time_key.replace('ruin_', '').capitalize()} {time}: {ruin_time_counts[time]} paths ruined")
    else:
        print("\nNo tracks ended in ruin.")

    if args.historic:
        from engine.historic_returns import historical_stress_real_factors_70_30
        historic_results = []
        print("\nHistoric scenario results:")
        for key, seq in historical_stress_real_factors_70_30.items():
            r = run_historic_scenario(params, seq["real_factors"], seq.get("property_factors"))
            r["name"] = _format_historic_name(key)
            r["start_year"] = int(seq["years"][0])
            r["end_year"] = int(seq["years"][-1])
            historic_results.append(r)
            status = (f"RUIN at age {r['ruin_age']}" if r["ruined"]
                      else f"survived ₪{r['terminal_portfolio']:,.0f}")
            print(f"  [{status}]  {r['name']}")
        fig = plot_with_historic(result, historic_results)
        fig.show()
    elif args.plot:
        fig = plot_cash_flow(result)
        fig.show()


if __name__ == "__main__":
    _cli()
