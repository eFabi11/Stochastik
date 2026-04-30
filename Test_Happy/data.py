import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import linregress

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


def plot_correlation(ax, df, y_column, y_label):
    # Daten vorbereiten
    subset = df[df['year'] == 2024][
        ['happiness_score', y_column]
    ].dropna()

    x = subset['happiness_score']
    y = subset[y_column]

    # Pearson-Korrelation
    correlation = x.corr(y)
    r_squared = correlation**2

    # Lineare Regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Regressionslinie
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # Plot
    ax.scatter(x, y)

    ax.plot(
        x_line,
        y_line,
        color='red',
        linewidth=2,
        label='Regression'
    )

    ax.set_title(f'Korrelation Happines Score und {y_label} (2024)')
    ax.set_xlabel('Happiness Score')
    ax.set_ylabel(y_label)

    ax.legend()

    # Statistik anzeigen
    ax.text(
        0.05,
        0.95,
        f'r = {r_value:.2f}\nR² = {r_squared:.2f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', alpha=0.3)
    )

fig, axes = plt.subplots(2, 3, figsize=(18, 6))

fig.suptitle(
    'World Happiness Report – Korrelationen (2024)',
    fontsize=14,
    fontweight='bold'
)

plot_correlation(
    axes[0][0],
    df,
    'explained_social_support',
    'Social Support'
)

plot_correlation(
    axes[0][1],
    df,
    'explained_healthy_life_expectancy',
    'Healthy Life Expectancy'
)

plot_correlation(
    axes[0][2],
    df,
    'explained_freedom',
    'Freedom'
)

plot_correlation(
    axes[1][0],
    df,
    'explained_log_gdp_per_capita',
    'Log Gdp per capita'
)

plot_correlation(
    axes[1][1],
    df,
    'explained_generosity',
    'Generosity'
)

plot_correlation(
    axes[1][2],
    df,
    'explained_corruption',
    'Corruption'
)

plt.tight_layout()
plt.savefig(BASE_DIR / 'happiness_correlations.png', dpi=150, bbox_inches='tight')
plt.show()


### Multivariat: 

## Frage: Wie sieht ein multivariater Plot aus? Mehr als zwei Variablen?
# bisher nur "bivariat"