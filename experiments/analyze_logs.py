# parse timestamps
# compute statistics: 
# - latency matrics: p50, p95, mean, std
# - common questions asked
# - new users per day plot

import os
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime, date
from collections import defaultdict
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

load_dotenv()
LOG_PATH = Path(os.getenv("LOG_PATH", "experiments/logs/chat_logs.jsonl"))

def iter_logs():
    with LOG_PATH.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def percentile(vals: list[float], p=0.5):
    # TODO: double-check this
    if not vals:
        return None
    vals = sorted(vals)

    if p==0:
        return vals[0]
    if p==1:
        return vals[-1]
    
    x = (len(vals)-1)*p
    floor = int(x)
    if floor == x:
        return vals[floor]
    ceil = floor + 1
    return vals[floor] + (vals[ceil] - vals[floor])*(x-floor)  # linear interpolation


def latency_metrics() -> dict[str, float]:
    # todo: what's wrong with the std calculation? its too big
    tokens_per_sec = sorted([1000/ float(line['token_latency_ms']) for line in iter_logs()])
    mean = sum(tokens_per_sec) / (n:=len(tokens_per_sec))
    std = sqrt((sum([x**2 for x in tokens_per_sec])/n) - mean**2)

    return {
        'p25': percentile(tokens_per_sec, p=0.25),
        'p50': percentile(tokens_per_sec, p=.5), 
        'p75': percentile(tokens_per_sec, p=.75),
        'mean': mean,
        'std': std
        }


def daily_users() -> dict[date, int]:
    """Count unique users per day."""
    users_per_day: dict[date, set[str]] = defaultdict(set)
    for line in iter_logs():
        ts = datetime.fromisoformat(line['timestamp'])
        day = date(ts.year, ts.month, ts.day)
        user_id = line.get('user_id')
        if user_id:
            users_per_day[day].add(user_id)
    return {day: len(users) for day, users in users_per_day.items()}


def plot_daily_users(output_path: Path | None = None) -> None:
    """Create a Tufte-style plot of users per day."""
    users_per_day = daily_users()
    
    if not users_per_day:
        print("No user data to plot")
        return
    
    # Sort by date
    dates = sorted(users_per_day.keys())
    counts = [users_per_day[d] for d in dates]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Thin, subtle grid lines
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='#999999')
    ax.set_axisbelow(True)
    
    # Plot the data
    ax.plot(dates, counts, color='#1f77b4', linewidth=1.5, marker='o', 
            markersize=4, markerfacecolor='#1f77b4', markeredgecolor='white', 
            markeredgewidth=0.5)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Labels with clear typography
    ax.set_xlabel('Date', fontsize=11, color='#333333')
    ax.set_ylabel('Unique Users', fontsize=11, color='#333333')
    ax.set_title('Unique Users Per Day', fontsize=13, fontweight='normal', 
                 color='#333333', pad=10)
    
    # Ensure y-axis starts at 0 (Tufte: show full context)
    ax.set_ylim(bottom=0)
    
    # Tight layout to maximize data area
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Generate plot
    plot_path = LOG_PATH.parent / "users_per_day.png"
    plot_daily_users(output_path=plot_path)

