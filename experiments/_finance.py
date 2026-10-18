"""Finance data helpers for M8.

Downloads the Fama-French 48 Industry Portfolios daily returns from Ken
French's data library (public, no authentication, redistributable with
attribution). Returns an (n_days, 48) daily-return matrix and a per-industry
sector grouping used for colouring in the hero figure. Downloads are cached
to a compressed .npz under ``papers/a/runs/m8/`` for offline reproducibility.
"""

from __future__ import annotations

import io
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

FF48_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/48_Industry_Portfolios_daily_CSV.zip"
)

FF48_SECTOR_MAP: dict[str, str] = {
    "Agric": "Materials",
    "Food": "Consumer Staples",
    "Soda": "Consumer Staples",
    "Beer": "Consumer Staples",
    "Smoke": "Consumer Staples",
    "Toys": "Consumer Discretionary",
    "Fun": "Communication Services",
    "Books": "Communication Services",
    "Hshld": "Consumer Staples",
    "Clths": "Consumer Discretionary",
    "Hlth": "Health Care",
    "MedEq": "Health Care",
    "Drugs": "Health Care",
    "Chems": "Materials",
    "Rubbr": "Materials",
    "Txtls": "Consumer Discretionary",
    "BldMt": "Industrials",
    "Cnstr": "Industrials",
    "Steel": "Materials",
    "FabPr": "Industrials",
    "Mach": "Industrials",
    "ElcEq": "Industrials",
    "Autos": "Consumer Discretionary",
    "Aero": "Industrials",
    "Ships": "Industrials",
    "Guns": "Industrials",
    "Gold": "Materials",
    "Mines": "Materials",
    "Coal": "Energy",
    "Oil": "Energy",
    "Util": "Utilities",
    "Telcm": "Communication Services",
    "PerSv": "Consumer Discretionary",
    "BusSv": "Industrials",
    "Comps": "Information Technology",
    "Chips": "Information Technology",
    "LabEq": "Health Care",
    "Paper": "Materials",
    "Boxes": "Materials",
    "Trans": "Industrials",
    "Whlsl": "Consumer Staples",
    "Rtail": "Consumer Discretionary",
    "Meals": "Consumer Discretionary",
    "Banks": "Financials",
    "Insur": "Financials",
    "RlEst": "Real Estate",
    "Fin": "Financials",
    "Other": "Other",
}

SECTOR_PALETTE: dict[str, str] = {
    "Information Technology": "#1f77b4",
    "Health Care": "#2ca02c",
    "Financials": "#d62728",
    "Consumer Discretionary": "#ff7f0e",
    "Communication Services": "#9467bd",
    "Industrials": "#8c564b",
    "Consumer Staples": "#e377c2",
    "Energy": "#bcbd22",
    "Utilities": "#17becf",
    "Real Estate": "#7f7f7f",
    "Materials": "#aec7e8",
    "Other": "#c7c7c7",
}


@dataclass(frozen=True)
class FinanceData:
    returns: np.ndarray  # (n_days, 48) daily percent returns
    industries: list[str]
    sector_of: list[str]
    date_strings: list[str]
    start_date: str
    end_date: str


def _parse_ff48_csv(csv_text: str, min_year: int) -> FinanceData:
    """Parse the Ken French 48-industry daily CSV."""
    lines = csv_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 49 and cols[1] in FF48_SECTOR_MAP:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("could not locate FF48 value-weighted header")

    industries = [c.strip() for c in lines[header_idx].split(",")[1:49]]
    sectors = [FF48_SECTOR_MAP[ind] for ind in industries]

    data_rows: list[list[float]] = []
    date_strings: list[str] = []
    date_re = re.compile(r"^\s*\d{8}\s*$")
    for line in lines[header_idx + 1 :]:
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 49 or not cols[0]:
            break
        if not date_re.match(cols[0]):
            break
        year = int(cols[0][:4])
        if year < min_year:
            continue
        try:
            row = [float(x) for x in cols[1:49]]
        except ValueError:
            break
        data_rows.append(row)
        date_strings.append(cols[0])

    returns = np.asarray(data_rows, dtype=np.float64)
    returns = np.where(returns <= -99.0, 0.0, returns)
    return FinanceData(
        returns=returns,
        industries=industries,
        sector_of=sectors,
        date_strings=date_strings,
        start_date=date_strings[0] if date_strings else "",
        end_date=date_strings[-1] if date_strings else "",
    )


def download_ff48(
    min_year: int = 2000,
    cache_path: str | None = None,
    source_zip_path: str | None = None,
) -> FinanceData:
    """Download or load from cache the Fama-French 48 daily portfolio returns.

    Resolution order: parsed .npz cache; local source zip (if provided and
    present); HTTP download of ``FF48_URL``. The local-source path exists so
    the driver works on networks that intercept TLS; drop the zip from Ken
    French's library at ``source_zip_path`` and the next run goes offline.
    """
    if cache_path and os.path.exists(cache_path):
        payload = np.load(cache_path, allow_pickle=True)
        return FinanceData(
            returns=payload["returns"],
            industries=list(payload["industries"]),
            sector_of=list(payload["sectors"]),
            date_strings=list(payload["date_strings"]),
            start_date=str(payload["start_date"]),
            end_date=str(payload["end_date"]),
        )

    if source_zip_path and os.path.exists(source_zip_path):
        with open(source_zip_path, "rb") as f:
            content = f.read()
    else:
        with urllib.request.urlopen(FF48_URL, timeout=60) as resp:
            content = resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        inner_name = zf.namelist()[0]
        csv_text = zf.read(inner_name).decode("latin-1")

    data = _parse_ff48_csv(csv_text, min_year=min_year)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            returns=data.returns,
            industries=np.array(data.industries),
            sectors=np.array(data.sector_of),
            date_strings=np.array(data.date_strings),
            start_date=data.start_date,
            end_date=data.end_date,
        )
    return data
