import pandas as pd
import pytest
from numbers_parser import Document

from engine.params import scenario_from_file, scenario_from_xlsx


def _xlsx_to_numbers(xlsx_path, numbers_path):
    """Build a .numbers fixture from an xlsx file's data via numbers_parser.

    ponytail: NaN cells must be *skipped* (left unwritten), not written as
    None (numbers_parser raises ValueError) or "" (breaks the
    df[...].notna() filters in engine.params, which then crash on int('')).
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    doc = Document()
    table = doc.sheets[0].tables[0]
    while table.num_rows < len(df) + 1:
        table.add_row()
    while table.num_cols < len(df.columns):
        table.add_column()
    for c, col in enumerate(df.columns):
        table.write(0, c, col)
    for r in range(len(df)):
        for c in range(len(df.columns)):
            v = df.iloc[r, c]
            if pd.isna(v):
                continue
            if hasattr(v, "item"):
                v = v.item()
            table.write(r + 1, c, v)
    doc.save(numbers_path)


def test_numbers_roundtrip_matches_xlsx(tmp_path):
    """A .numbers file with the same data as the example xlsx must produce
    the same scenario dict via the scenario_from_file dispatcher."""
    numbers_path = tmp_path / "scenario.numbers"
    _xlsx_to_numbers("scenario_data_example.xlsx", numbers_path)

    got = scenario_from_file(str(numbers_path))
    want = scenario_from_xlsx("scenario_data_example.xlsx")
    assert got == want


def test_dispatcher_xlsx_branch_unchanged():
    """scenario_from_file must still route .xlsx paths through scenario_from_xlsx."""
    assert scenario_from_file("scenario_data_example.xlsx") == scenario_from_xlsx(
        "scenario_data_example.xlsx"
    )
