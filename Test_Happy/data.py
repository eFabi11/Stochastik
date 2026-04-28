import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

# Daten laden
df = pd.read_csv(DATA_DIR / "world_happiness.csv") 

# Kurzer Check
print(df.shape)
print(df.columns.tolist())
print(df['year'].unique())
print(df.head(3))


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('World Happiness Report – Erste Übersicht', fontsize=14, fontweight='bold')

# ── Plot 1: Top 10 Länder 2024 ────────────────────────────────────────────────
ax = axes[0]
top10 = df[df['year'] == 2024].nsmallest(10, 'rank_in_year')
ax.barh(top10['country'][::-1], top10['happiness_score'][::-1], color='steelblue')
ax.set_title('Top 10 glücklichste Länder (2024)', fontweight='bold')
ax.set_xlabel('Happiness Score')
ax.set_xlim(6, 8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Plot 2: Zeitreihe ausgewählter Länder ─────────────────────────────────────
ax = axes[1]
laender = ['Finland', 'Germany', 'United States', 'Afghanistan', 'Ukraine']
farben  = ['steelblue', 'orange', 'green', 'red', 'purple']
for land, farbe in zip(laender, farben):
    d = df[df['country'] == land].sort_values('year')
    ax.plot(d['year'], d['happiness_score'], marker='o', markersize=4,
            label=land, color=farbe, linewidth=2)
ax.set_title('Happiness Score über die Zeit', fontweight='bold')
ax.set_xlabel('Jahr')
ax.set_ylabel('Happiness Score')
ax.legend(fontsize=8)
ax.grid(linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Plot 3: Verteilung 2024 als Histogramm ────────────────────────────────────
ax = axes[2]
scores_2024 = df[df['year'] == 2024]['happiness_score'].dropna()
ax.hist(scores_2024, bins=20, color='steelblue', edgecolor='white')
ax.axvline(scores_2024.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mittelwert: {scores_2024.mean():.2f}')
ax.set_title('Verteilung Happiness Score (2024)', fontweight='bold')
ax.set_xlabel('Happiness Score')
ax.set_ylabel('Anzahl Länder')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(BASE_DIR / 'happiness_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gespeichert ✓")