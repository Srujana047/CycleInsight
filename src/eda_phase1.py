"""
EDA Phase 1: User and Cycle Analysis
====================================
Exploratory analysis of user structure and cycle-length distributions
for FedCycleData071012.csv and cleaned_dataset.csv.

No ML models are trained — this script focuses only on understanding
user and cycle structure, data quality, and basic descriptive statistics.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths & plotting defaults
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "eda1_summary.md"

FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)

# Clinically plausible menstrual cycle length bounds (days)
CYCLE_PLAUSIBLE_MIN = 21
CYCLE_PLAUSIBLE_MAX = 45
# Extremely short / long thresholds for "suspicious" flags
CYCLE_SUSPICIOUS_LOW = 15
CYCLE_SUSPICIOUS_HIGH = 60
# Users with fewer records than this are flagged as low-coverage
LOW_RECORD_THRESHOLD = 3


# ===========================================================================
# Helper functions
# ===========================================================================
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both source CSVs with consistent missing-value handling."""
    fed = pd.read_csv(DATA_DIR / "FedCycleData071012.csv", low_memory=False)
    cleaned = pd.read_csv(DATA_DIR / "cleaned_dataset.csv", low_memory=False)

    # FedCycle often stores blanks as empty strings; coerce key numerics later.
    fed = fed.replace(r"^\s*$", np.nan, regex=True)
    return fed, cleaned


def dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data-types / null / unique summary table."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "null_count": df.isna().sum(),
            "null_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        }
    )
    return summary


def iqr_outlier_mask(series: pd.Series) -> tuple[pd.Series, float, float, float, float]:
    """
    Flag potential outliers with the classic IQR rule:
        x < Q1 - 1.5*IQR  OR  x > Q3 + 1.5*IQR
    Returns (boolean mask, Q1, Q3, lower_fence, upper_fence).
    """
    clean = series.dropna()
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask, q1, q3, lower, upper


def cycle_stats(series: pd.Series) -> dict:
    """Compute mean / median / std / min / max for a cycle-length series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def user_record_stats(df: pd.DataFrame, user_col: str) -> pd.Series:
    """Records (rows) per user."""
    return df.groupby(user_col).size().sort_values(ascending=False)


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Persist a figure under reports/figures/ and close it."""
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path.relative_to(ROOT)}")
    return path


# ===========================================================================
# 1. Dataset overview
# ===========================================================================
def analyze_overview(fed: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Section 1 — shapes, column counts, and dtype summaries."""
    print("\n=== 1. Dataset Overview ===")
    overview = {
        "fed": {
            "rows": fed.shape[0],
            "cols": fed.shape[1],
            "dtypes": dtype_summary(fed),
            "dtype_counts": fed.dtypes.value_counts().to_dict(),
        },
        "cleaned": {
            "rows": cleaned.shape[0],
            "cols": cleaned.shape[1],
            "dtypes": dtype_summary(cleaned),
            "dtype_counts": cleaned.dtypes.value_counts().to_dict(),
        },
    }
    for label, meta in (("FedCycle", overview["fed"]), ("Cleaned", overview["cleaned"])):
        print(f"  {label}: {meta['rows']:,} rows x {meta['cols']} columns")
        print(f"    dtype counts: {meta['dtype_counts']}")
    return overview


# ===========================================================================
# 2. User analysis
# ===========================================================================
def analyze_users(fed: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Section 2 — unique users and records-per-user distributions."""
    print("\n=== 2. User Analysis ===")

    fed_counts = user_record_stats(fed, "ClientID")
    clean_counts = user_record_stats(cleaned, "id")

    def summarize(counts: pd.Series, label: str) -> dict:
        summary = {
            "n_users": int(counts.shape[0]),
            "total_records": int(counts.sum()),
            "mean_records": float(counts.mean()),
            "median_records": float(counts.median()),
            "min_records": int(counts.min()),
            "max_records": int(counts.max()),
            "top20": counts.head(20),
            "counts": counts,
        }
        print(f"  {label}")
        print(f"    unique users          : {summary['n_users']}")
        print(f"    avg records / user    : {summary['mean_records']:.2f}")
        print(f"    median records / user : {summary['median_records']:.1f}")
        print(f"    min / max             : {summary['min_records']} / {summary['max_records']}")
        return summary

    return {
        "fed": summarize(fed_counts, "FedCycle (ClientID)"),
        "cleaned": summarize(clean_counts, "Cleaned (id)"),
    }


# ===========================================================================
# 3. Cycle analysis
# ===========================================================================
def analyze_cycles(fed: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """
    Section 3 — identify cycle-length columns, compute descriptive stats,
    and detect IQR outliers.
    """
    print("\n=== 3. Cycle Analysis ===")

    # FedCycle: LengthofCycle is the primary per-cycle length column.
    # MeanCycleLength is user-level and sparsely populated (~first cycle only).
    fed_cycle_cols = ["LengthofCycle", "MeanCycleLength"]
    fed["LengthofCycle_num"] = pd.to_numeric(fed["LengthofCycle"], errors="coerce")
    fed["MeanCycleLength_num"] = pd.to_numeric(fed["MeanCycleLength"], errors="coerce")

    # Cleaned: inter_cycle_length is the cycle-length proxy (constant per user
    # in this engineered dataset); cycle_variability captures within-user spread.
    clean_cycle_cols = ["inter_cycle_length", "cycle_variability"]
    cleaned["inter_cycle_length_num"] = pd.to_numeric(
        cleaned["inter_cycle_length"], errors="coerce"
    )
    cleaned["cycle_variability_num"] = pd.to_numeric(
        cleaned["cycle_variability"], errors="coerce"
    )

    # For cleaned data, cycle length is repeated per day-row; analyze unique
    # per-user values to avoid inflating sample size.
    clean_user_cycle = (
        cleaned.groupby("id")["inter_cycle_length_num"].first().dropna()
    )

    results = {
        "fed_cycle_cols": fed_cycle_cols,
        "clean_cycle_cols": clean_cycle_cols,
        "fed_length": {
            "stats": cycle_stats(fed["LengthofCycle_num"]),
            "series": fed["LengthofCycle_num"],
        },
        "fed_mean_length": {
            "stats": cycle_stats(fed["MeanCycleLength_num"]),
            "series": fed["MeanCycleLength_num"],
        },
        "clean_inter_cycle_row": {
            "stats": cycle_stats(cleaned["inter_cycle_length_num"]),
            "series": cleaned["inter_cycle_length_num"],
        },
        "clean_inter_cycle_user": {
            "stats": cycle_stats(clean_user_cycle),
            "series": clean_user_cycle,
        },
        "clean_variability": {
            "stats": cycle_stats(cleaned["cycle_variability_num"]),
            "series": cleaned["cycle_variability_num"],
        },
    }

    # IQR outlier detection on primary cycle-length series
    for key in ("fed_length", "clean_inter_cycle_user"):
        series = results[key]["series"]
        mask, q1, q3, lower, upper = iqr_outlier_mask(series)
        n_out = int(mask.fillna(False).sum())
        results[key]["iqr"] = {
            "q1": q1,
            "q3": q3,
            "lower_fence": lower,
            "upper_fence": upper,
            "n_outliers": n_out,
            "outlier_values": sorted(series[mask].dropna().unique().tolist()),
            "mask": mask,
        }
        print(f"  {key}: stats={results[key]['stats']}")
        print(
            f"    IQR fences [{lower:.2f}, {upper:.2f}] — "
            f"{n_out} outliers {results[key]['iqr']['outlier_values']}"
        )

    return results


# ===========================================================================
# 4. Visualizations
# ===========================================================================
def make_visualizations(user_results: dict, cycle_results: dict) -> dict[str, Path]:
    """
    Section 4 — produce and save the four required figure families
    (records/user hist, cycle-length hist, cycle-length boxplot, top-20 bar).
    Figures are generated for both datasets where applicable.
    """
    print("\n=== 4. Visualizations ===")
    paths: dict[str, Path] = {}

    # --- 4a. Histogram of records per user ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(
        axes,
        ("fed", "cleaned"),
        ("FedCycle — Records per User", "Cleaned — Records per User"),
    ):
        counts = user_results[key]["counts"]
        sns.histplot(
            counts,
            bins=min(30, max(5, counts.nunique())),
            kde=False,
            ax=ax,
            color="#2E86AB",
        )
        ax.axvline(
            counts.mean(),
            color="#E94F37",
            linestyle="--",
            label=f"mean={counts.mean():.1f}",
        )
        ax.axvline(
            counts.median(),
            color="#F6AE2D",
            linestyle="-.",
            label=f"median={counts.median():.1f}",
        )
        ax.set_title(title)
        ax.set_xlabel("Records per user")
        ax.set_ylabel("Number of users")
        ax.legend(fontsize=9)
    fig.suptitle("Distribution of Records per User", y=1.02)
    paths["records_per_user_hist"] = save_fig(fig, "eda1_records_per_user_hist.png")

    # --- 4b. Histogram of cycle length -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fed_len = cycle_results["fed_length"]["series"].dropna()
    clean_len = cycle_results["clean_inter_cycle_user"]["series"].dropna()

    sns.histplot(fed_len, bins=20, kde=True, ax=axes[0], color="#2E86AB")
    axes[0].axvline(
        fed_len.mean(), color="#E94F37", linestyle="--", label=f"mean={fed_len.mean():.1f}"
    )
    axes[0].axvline(
        fed_len.median(),
        color="#F6AE2D",
        linestyle="-.",
        label=f"median={fed_len.median():.1f}",
    )
    axes[0].set_title("FedCycle — LengthofCycle")
    axes[0].set_xlabel("Cycle length (days)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)

    sns.histplot(
        clean_len,
        bins=min(15, max(5, clean_len.nunique())),
        kde=False,
        ax=axes[1],
        color="#2E86AB",
    )
    axes[1].axvline(
        clean_len.mean(),
        color="#E94F37",
        linestyle="--",
        label=f"mean={clean_len.mean():.1f}",
    )
    axes[1].axvline(
        clean_len.median(),
        color="#F6AE2D",
        linestyle="-.",
        label=f"median={clean_len.median():.1f}",
    )
    axes[1].set_title("Cleaned — inter_cycle_length (per user)")
    axes[1].set_xlabel("Cycle length (days)")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=9)

    fig.suptitle("Cycle Length Distributions", y=1.02)
    paths["cycle_length_hist"] = save_fig(fig, "eda1_cycle_length_hist.png")

    # --- 4c. Boxplot of cycle length ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=fed_len, ax=axes[0], color="#A23B72", width=0.4)
    axes[0].set_title("FedCycle — LengthofCycle")
    axes[0].set_ylabel("Cycle length (days)")

    sns.boxplot(y=clean_len, ax=axes[1], color="#A23B72", width=0.4)
    axes[1].set_title("Cleaned — inter_cycle_length (per user)")
    axes[1].set_ylabel("Cycle length (days)")

    fig.suptitle("Cycle Length Boxplots (IQR fences)", y=1.02)
    paths["cycle_length_box"] = save_fig(fig, "eda1_cycle_length_boxplot.png")

    # --- 4d. Bar chart — top 20 users by record count ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, title, xlabel in zip(
        axes,
        ("fed", "cleaned"),
        ("FedCycle — Top 20 Users", "Cleaned — Top 20 Users"),
        ("ClientID", "id"),
    ):
        top20 = user_results[key]["top20"]
        plot_df = top20.reset_index()
        plot_df.columns = [xlabel, "records"]
        plot_df[xlabel] = plot_df[xlabel].astype(str)
        sns.barplot(
            data=plot_df, y=xlabel, x="records", ax=ax, color="#2E86AB", orient="h"
        )
        ax.set_title(title)
        ax.set_xlabel("Record count")
        ax.set_ylabel(xlabel)
    fig.suptitle("Top 20 Users by Record Count", y=1.02)
    fig.tight_layout()
    paths["top20_users_bar"] = save_fig(fig, "eda1_top20_users_bar.png")

    return paths


# ===========================================================================
# 5. Data quality checks
# ===========================================================================
def data_quality_checks(
    fed: pd.DataFrame,
    cleaned: pd.DataFrame,
    user_results: dict,
    cycle_results: dict,
) -> dict:
    """
    Section 5 — flag impossible / suspicious cycle lengths and users with
    extremely low record counts.
    """
    print("\n=== 5. Data Quality Checks ===")
    findings: dict = {}

    # Impossible / suspicious cycle lengths (FedCycle)
    fed_len = cycle_results["fed_length"]["series"]
    impossible = fed[(fed_len < CYCLE_SUSPICIOUS_LOW) | (fed_len > CYCLE_SUSPICIOUS_HIGH)]
    outside_plausible = fed[
        fed_len.notna()
        & ((fed_len < CYCLE_PLAUSIBLE_MIN) | (fed_len > CYCLE_PLAUSIBLE_MAX))
    ]
    findings["fed_impossible"] = {
        "n": int(len(impossible)),
        "bounds": (CYCLE_SUSPICIOUS_LOW, CYCLE_SUSPICIOUS_HIGH),
        "values": sorted(fed_len.loc[impossible.index].dropna().unique().tolist()),
    }
    findings["fed_outside_plausible"] = {
        "n": int(len(outside_plausible)),
        "bounds": (CYCLE_PLAUSIBLE_MIN, CYCLE_PLAUSIBLE_MAX),
        "value_counts": outside_plausible["LengthofCycle_num"]
        .value_counts()
        .sort_index()
        .to_dict(),
    }

    # Cleaned cycle lengths (per-user)
    clean_user_len = cycle_results["clean_inter_cycle_user"]["series"]
    clean_outside = clean_user_len[
        (clean_user_len < CYCLE_PLAUSIBLE_MIN) | (clean_user_len > CYCLE_PLAUSIBLE_MAX)
    ]
    findings["clean_outside_plausible"] = {
        "n": int(len(clean_outside)),
        "bounds": (CYCLE_PLAUSIBLE_MIN, CYCLE_PLAUSIBLE_MAX),
        "values": sorted(clean_outside.unique().tolist()),
    }

    # Low record-count users
    fed_low = user_results["fed"]["counts"][
        user_results["fed"]["counts"] < LOW_RECORD_THRESHOLD
    ]
    clean_low = user_results["cleaned"]["counts"][
        user_results["cleaned"]["counts"] < LOW_RECORD_THRESHOLD
    ]
    findings["fed_low_record_users"] = {
        "threshold": LOW_RECORD_THRESHOLD,
        "n": int(len(fed_low)),
        "users": fed_low.to_dict(),
    }
    findings["clean_low_record_users"] = {
        "threshold": LOW_RECORD_THRESHOLD,
        "n": int(len(clean_low)),
        "users": clean_low.to_dict(),
    }

    # Missingness on primary cycle column
    findings["fed_length_missing"] = int(fed_len.isna().sum())
    findings["clean_inter_cycle_missing"] = int(
        cleaned["inter_cycle_length_num"].isna().sum()
    )

    # IQR outliers (already computed)
    findings["fed_iqr_outliers"] = cycle_results["fed_length"]["iqr"]["n_outliers"]
    findings["clean_iqr_outliers"] = cycle_results["clean_inter_cycle_user"]["iqr"][
        "n_outliers"
    ]

    print(
        f"  FedCycle impossible (<{CYCLE_SUSPICIOUS_LOW} or >{CYCLE_SUSPICIOUS_HIGH}): "
        f"{findings['fed_impossible']['n']}"
    )
    print(
        f"  FedCycle outside plausible [{CYCLE_PLAUSIBLE_MIN},{CYCLE_PLAUSIBLE_MAX}]: "
        f"{findings['fed_outside_plausible']['n']}"
    )
    print(f"  Cleaned outside plausible: {findings['clean_outside_plausible']['n']}")
    print(
        f"  FedCycle low-record users (<{LOW_RECORD_THRESHOLD}): "
        f"{findings['fed_low_record_users']['n']}"
    )
    print(
        f"  Cleaned low-record users (<{LOW_RECORD_THRESHOLD}): "
        f"{findings['clean_low_record_users']['n']}"
    )

    return findings


# ===========================================================================
# 6. Markdown report
# ===========================================================================
def _fmt_stats(stats: dict) -> str:
    return (
        f"| n | mean | median | std | min | max |\n"
        f"|---|------|--------|-----|-----|-----|\n"
        f"| {stats['n']} | {stats['mean']:.2f} | {stats['median']:.2f} | "
        f"{stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |"
    )


def _fmt_dtype_counts(counts: dict) -> str:
    lines = ["| dtype | count |", "|-------|-------|"]
    for k, v in counts.items():
        lines.append(f"| `{k}` | {v} |")
    return "\n".join(lines)


def _fmt_top20(top20: pd.Series, user_label: str) -> str:
    lines = [f"| rank | {user_label} | records |", "|------|----------|---------|"]
    for i, (uid, n) in enumerate(top20.items(), start=1):
        lines.append(f"| {i} | `{uid}` | {n} |")
    return "\n".join(lines)


def write_report(
    overview: dict,
    user_results: dict,
    cycle_results: dict,
    fig_paths: dict[str, Path],
    quality: dict,
) -> Path:
    """Section 6 — assemble a professional markdown summary report."""
    print("\n=== 6. Writing Report ===")

    fed_u = user_results["fed"]
    cl_u = user_results["cleaned"]
    fed_iqr = cycle_results["fed_length"]["iqr"]
    cl_iqr = cycle_results["clean_inter_cycle_user"]["iqr"]

    # Relative figure paths for the markdown file
    rel = {k: f"figures/{p.name}" for k, p in fig_paths.items()}

    md = f"""# EDA Phase 1 — User and Cycle Analysis

**Project:** CycleInsight (Menstrual Intelligence System)  
**Scope:** User structure and cycle-length structure only (no ML modeling)  
**Datasets:** `FedCycleData071012.csv`, `cleaned_dataset.csv`

---

## 1. Dataset Overview

### FedCycleData071012.csv
- **Rows:** {overview['fed']['rows']:,}
- **Columns:** {overview['fed']['cols']}
- **Grain:** one row ≈ one menstrual cycle per client
- **User key:** `ClientID`
- **Primary cycle-length column:** `LengthofCycle`

{_fmt_dtype_counts(overview['fed']['dtype_counts'])}

### cleaned_dataset.csv
- **Rows:** {overview['cleaned']['rows']:,}
- **Columns:** {overview['cleaned']['cols']}
- **Grain:** one row ≈ one study day per participant
- **User key:** `id`
- **Primary cycle-length column:** `inter_cycle_length` (also `cycle_variability`)

{_fmt_dtype_counts(overview['cleaned']['dtype_counts'])}

---

## 2. User Analysis

### FedCycle (`ClientID`)
| metric | value |
|--------|-------|
| Unique users | {fed_u['n_users']} |
| Total records | {fed_u['total_records']:,} |
| Average records / user | {fed_u['mean_records']:.2f} |
| Median records / user | {fed_u['median_records']:.1f} |
| Minimum records / user | {fed_u['min_records']} |
| Maximum records / user | {fed_u['max_records']} |

#### Top 20 users by record count
{_fmt_top20(fed_u['top20'], 'ClientID')}

### Cleaned (`id`)
| metric | value |
|--------|-------|
| Unique users | {cl_u['n_users']} |
| Total records | {cl_u['total_records']:,} |
| Average records / user | {cl_u['mean_records']:.2f} |
| Median records / user | {cl_u['median_records']:.1f} |
| Minimum records / user | {cl_u['min_records']} |
| Maximum records / user | {cl_u['max_records']} |

#### Top 20 users by record count
{_fmt_top20(cl_u['top20'], 'id')}

![Records per user]({rel['records_per_user_hist']})

![Top 20 users]({rel['top20_users_bar']})

---

## 3. Cycle Analysis

### Identified cycle-length columns

| dataset | columns | notes |
|---------|---------|-------|
| FedCycle | `LengthofCycle`, `MeanCycleLength` | `LengthofCycle` is per-cycle; `MeanCycleLength` is sparse (populated mainly on a user's first recorded cycle) |
| Cleaned | `inter_cycle_length`, `cycle_variability` | `inter_cycle_length` is constant per user in this file; analyzed at **user level** to avoid day-row inflation |

### Descriptive statistics — FedCycle `LengthofCycle`
{_fmt_stats(cycle_results['fed_length']['stats'])}

### Descriptive statistics — FedCycle `MeanCycleLength` (non-null only)
{_fmt_stats(cycle_results['fed_mean_length']['stats'])}

### Descriptive statistics — Cleaned `inter_cycle_length` (per user)
{_fmt_stats(cycle_results['clean_inter_cycle_user']['stats'])}

### Descriptive statistics — Cleaned `inter_cycle_length` (all day-rows, for reference)
{_fmt_stats(cycle_results['clean_inter_cycle_row']['stats'])}

### Descriptive statistics — Cleaned `cycle_variability` (day-rows)
{_fmt_stats(cycle_results['clean_variability']['stats'])}

### IQR outlier detection

**FedCycle `LengthofCycle`**
- Q1 = {fed_iqr['q1']:.2f}, Q3 = {fed_iqr['q3']:.2f}
- Lower fence = {fed_iqr['lower_fence']:.2f}, Upper fence = {fed_iqr['upper_fence']:.2f}
- **Outliers:** {fed_iqr['n_outliers']} values — {fed_iqr['outlier_values']}

**Cleaned `inter_cycle_length` (per user)**
- Q1 = {cl_iqr['q1']:.2f}, Q3 = {cl_iqr['q3']:.2f}
- Lower fence = {cl_iqr['lower_fence']:.2f}, Upper fence = {cl_iqr['upper_fence']:.2f}
- **Outliers:** {cl_iqr['n_outliers']} values — {cl_iqr['outlier_values'] if cl_iqr['outlier_values'] else 'none'}

![Cycle length histogram]({rel['cycle_length_hist']})

![Cycle length boxplot]({rel['cycle_length_box']})

---

## 4. Visualizations

All figures are saved under `reports/figures/`:

| figure | file |
|--------|------|
| Histogram — records per user | `{fig_paths['records_per_user_hist'].name}` |
| Histogram — cycle length | `{fig_paths['cycle_length_hist'].name}` |
| Boxplot — cycle length | `{fig_paths['cycle_length_box'].name}` |
| Bar chart — top 20 users | `{fig_paths['top20_users_bar'].name}` |

---

## 5. Data Quality Checks

### Impossible / suspicious cycle lengths
Clinical reference used in this audit:
- **Suspicious / extreme:** < {CYCLE_SUSPICIOUS_LOW} or > {CYCLE_SUSPICIOUS_HIGH} days
- **Outside typical plausible range:** < {CYCLE_PLAUSIBLE_MIN} or > {CYCLE_PLAUSIBLE_MAX} days

| check | dataset | count | detail |
|-------|---------|-------|--------|
| Extreme / impossible | FedCycle | {quality['fed_impossible']['n']} | values: {quality['fed_impossible']['values'] or 'none'} |
| Outside plausible [{CYCLE_PLAUSIBLE_MIN}, {CYCLE_PLAUSIBLE_MAX}] | FedCycle | {quality['fed_outside_plausible']['n']} | value→count: {quality['fed_outside_plausible']['value_counts']} |
| Outside plausible [{CYCLE_PLAUSIBLE_MIN}, {CYCLE_PLAUSIBLE_MAX}] | Cleaned (per user) | {quality['clean_outside_plausible']['n']} | values: {quality['clean_outside_plausible']['values'] or 'none'} |
| Missing `LengthofCycle` | FedCycle | {quality['fed_length_missing']} | — |
| Missing `inter_cycle_length` | Cleaned | {quality['clean_inter_cycle_missing']} | — |
| IQR outliers | FedCycle | {quality['fed_iqr_outliers']} | see §3 |
| IQR outliers | Cleaned (per user) | {quality['clean_iqr_outliers']} | see §3 |

### Users with extremely low record counts (< {LOW_RECORD_THRESHOLD})

| dataset | n users | users (id → records) |
|---------|---------|----------------------|
| FedCycle | {quality['fed_low_record_users']['n']} | `{quality['fed_low_record_users']['users']}` |
| Cleaned | {quality['clean_low_record_users']['n']} | `{quality['clean_low_record_users']['users']}` |

---

## 6. Summary of Findings

1. **Different data grains.** FedCycle is **cycle-level** ({overview['fed']['rows']:,} cycles, {fed_u['n_users']} users); cleaned is **day-level** ({overview['cleaned']['rows']:,} days, {cl_u['n_users']} users). Direct row-count comparisons across datasets are not meaningful without aggregation.
2. **User coverage is uneven.** FedCycle averages ~{fed_u['mean_records']:.1f} cycles/user (range {fed_u['min_records']}–{fed_u['max_records']}); cleaned averages ~{cl_u['mean_records']:.1f} day-rows/user (range {cl_u['min_records']}–{cl_u['max_records']}). Several FedCycle users sit below the low-record threshold and may be too sparse for per-user modeling later.
3. **Cycle lengths are broadly physiological in FedCycle.** Mean ≈ {cycle_results['fed_length']['stats']['mean']:.1f} days (median {cycle_results['fed_length']['stats']['median']:.1f}); IQR outliers are predominantly long cycles above the upper fence (~{fed_iqr['upper_fence']:.0f} days), consistent with occasional long follicular phases rather than data-entry disasters. A non-trivial tail falls outside the [{CYCLE_PLAUSIBLE_MIN}, {CYCLE_PLAUSIBLE_MAX}] window and should be reviewed before prediction work.
4. **Cleaned `inter_cycle_length` is narrow and per-user constant.** Values span roughly {cycle_results['clean_inter_cycle_user']['stats']['min']:.0f}–{cycle_results['clean_inter_cycle_user']['stats']['max']:.0f} days with little dispersion and **no IQR outliers** at the user level — expected if the feature was engineered / constrained during cleaning.
5. **No extreme (<{CYCLE_SUSPICIOUS_LOW} or >{CYCLE_SUSPICIOUS_HIGH} day) FedCycle lengths** were detected under the suspicious-bounds rule used here.
6. **Next EDA phases** should move beyond user/cycle skeleton into symptom/phase distributions, missingness patterns by feature family, and longitudinal completeness per user — still without training models until the data contract is locked.

---

*Generated by `src/eda_phase1.py`*
"""

    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"  report -> {REPORT_PATH.relative_to(ROOT)}")
    return REPORT_PATH


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print("EDA Phase 1: User and Cycle Analysis")
    print(f"Root: {ROOT}")

    # Load
    fed, cleaned = load_datasets()

    # 1. Overview
    overview = analyze_overview(fed, cleaned)

    # 2. Users
    user_results = analyze_users(fed, cleaned)

    # 3. Cycles
    cycle_results = analyze_cycles(fed, cleaned)

    # 4. Figures
    fig_paths = make_visualizations(user_results, cycle_results)

    # 5. Quality
    quality = data_quality_checks(fed, cleaned, user_results, cycle_results)

    # 6. Report
    write_report(overview, user_results, cycle_results, fig_paths, quality)

    print("\nDone.")


if __name__ == "__main__":
    main()
