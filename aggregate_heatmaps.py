import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

data_path = Path(__file__).parent / "TPDOpenData_PoliceActivity_2025.csv"
out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
hour_order = list(range(24))

sequential_cmap = "Blues" 

def load_data() -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["EventDay"] = pd.Categorical(df["EventDay"], categories=day_order, ordered=True)
    df["EventMonth"] = pd.Categorical(df["EventMonth"], categories=month_order, ordered=True)
    return df


def build_marginal_aggregates(df: pd.DataFrame) -> None:
    "Counts by hour, by day-of-week, and by month, each saved to CSV."
    by_hour = df.groupby("EventHour").size().reindex(hour_order, fill_value=0)
    by_hour.rename("count").to_csv(out_dir / "counts_by_hour.csv")

    by_day = df.groupby("EventDay", observed=False).size().reindex(day_order, fill_value=0)
    by_day.rename("count").to_csv(out_dir / "counts_by_dayofweek.csv")

    by_month = df.groupby("EventMonth", observed=False).size().reindex(month_order, fill_value=0)
    by_month.rename("count").to_csv(out_dir / "counts_by_month.csv")


def hour_by_day_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    "Pivot a frame into a [day-of-week x hour] count matrix, fully reindexed."
    matrix = (
        frame.groupby(["EventDay", "EventHour"], observed=False)
        .size()
        .unstack("EventHour", fill_value=0)
        .reindex(index=day_order, columns=hour_order, fill_value=0)
    )
    return matrix


def plot_total_heatmap(df: pd.DataFrame) -> None:
    matrix = hour_by_day_matrix(df)
    matrix.to_csv(out_dir / "matrix_hour_by_day_total.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        matrix,
        cmap= sequential_cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Number of events"},
        ax=ax,
    )
    ax.set_title("Police Activity Volume by Hour and Day of Week (2025)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_total_volume.png", dpi=150)
    plt.close(fig)


def plot_faceted_by_category(df: pd.DataFrame) -> None:
    categories = df["EventCategory"].value_counts().index.tolist()
    n = len(categories)
    n_cols = 5
    n_rows = -(-n // n_cols)  # ceil

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
    axes = axes.flatten()

    for ax, category in zip(axes, categories):
        sub = df[df["EventCategory"] == category]
        matrix = hour_by_day_matrix(sub)
        # Each facet uses its own color scale so low-volume categories
        # (e.g. Family And Juvenile Matters) still show their internal
        # time-of-day/week pattern, not just get washed out by scale.
        sns.heatmap(
            matrix,
            cmap= sequential_cmap,
            cbar=True,
            linewidths=0.3,
            linecolor="white",
            ax=ax,
            xticklabels=4,
        )
        ax.set_title(f"{category} (n={len(sub):,})", fontsize=10)
        ax.set_xlabel("Hour")
        ax.set_ylabel("")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Police Activity by Hour and Day of Week, Faceted by EventCategory\n"
        "(each panel scaled to its own min/max to reveal its time-of-day shape)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "heatmap_by_category.png", dpi=150)
    plt.close(fig)

def main() -> None:
    df = load_data()
    build_marginal_aggregates(df)
    plot_total_heatmap(df)
    plot_faceted_by_category(df)
    print(f"Aggregates and charts written to {out_dir}")

main()
