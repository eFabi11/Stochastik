import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

MONATE = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni',
          'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']

KERN_SPALTEN = ['standort', 'monat', 'gemessene_fahrzeuge', 'verstoesse',
                'gueltige_verstoesse', 'verwarnungen', 'bussgelder',
                'max_geschwindigkeit', 'einnahmen', 'jahr']

MAX_SPEED = {
    "Laube": 30,
    "Steinstr": 30,
    "Gartenstr": 30,
    "Mainaustr": 50,
    "Loh": 30,
    "Moschee": 30,
    "Casino": 50,
    "Europabruecke": 60
}


dfs = []
for year in [2018, 2019, 2020, 2021, 2022, 2023]:
    # Daten laden
    df = pd.read_csv(DATA_DIR / f"{year}.csv", sep=';', decimal=',', thousands='.', encoding='latin-1') 
    df['jahr'] = year
    df['monat'] = df['monat'].str.strip().str.replace('Maerz', 'März', regex=False)
    df = df[~df['monat'].str.lower().eq('gesamt')]
    df = df[df['standort'].str.lower() != 'gesamt']
    for col in ['max_geschwindigkeit', 'einnahmen']:
        if col not in df.columns:
            df[col] = float('nan')
    df = df[[c for c in KERN_SPALTEN if c in df.columns]]
    df = df[df['gemessene_fahrzeuge'].notna() & (df['gemessene_fahrzeuge'] > 0)]
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all['verstoessquote'] = df_all['gueltige_verstoesse'] / df_all['gemessene_fahrzeuge'] * 100

df_all['speed_limit'] = df_all['standort'].map(MAX_SPEED)
df_speed = df_all.dropna(subset=['speed_limit', 'verstoessquote']).copy()
df_all['ueberschreitung'] = df_all['max_geschwindigkeit'] - df_all['speed_limit']
df_all['avg_fine'] = df_all['einnahmen'] / df_all['gueltige_verstoesse']

# ── Jahresaggregation ─────────────────────────────────────────────────────────
trend = df_all.groupby('jahr').agg(
    verstoesse_gesamt=('gueltige_verstoesse', 'sum'),
    fahrzeuge_gesamt=('gemessene_fahrzeuge', 'sum'),
    verstoessquote=('verstoessquote', 'mean'),
    einnahmen_gesamt=('einnahmen', 'sum')
).reset_index()

jahre = trend['jahr'].values

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Trendanalyse Geschwindigkeitsüberwachung Konstanz 2018–2023',
             fontsize=13, fontweight='bold', y=1.02)

# ── Subplot 1: Absolute Verstöße ──────────────────────────────────────────────
ax = axes[0]
ax.bar(jahre, trend['verstoesse_gesamt'], color='steelblue', edgecolor='white')

# Trendlinie
z = np.polyfit(jahre, trend['verstoesse_gesamt'], 1)
p = np.poly1d(z)
ax.plot(jahre, p(jahre), color='red', linestyle='--', linewidth=2, label=f'Trend')

ax.set_title('Gültige Verstöße pro Jahr', fontweight='bold')
ax.set_xlabel('Jahr')
ax.set_ylabel('Anzahl Verstöße')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(ax.patches, trend['verstoesse_gesamt']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{int(val):,}', ha='center', va='bottom', fontsize=8)

# ── Subplot 2: Verstoßquote ───────────────────────────────────────────────────
ax = axes[1]
ax.plot(jahre, trend['verstoessquote'], color='steelblue',
        marker='o', linewidth=2.5, markersize=8)

z2 = np.polyfit(jahre, trend['verstoessquote'], 1)
p2 = np.poly1d(z2)
ax.plot(jahre, p2(jahre), color='red', linestyle='--', linewidth=2, label='Trend')

ax.fill_between(jahre, trend['verstoessquote'], alpha=0.1, color='steelblue')
ax.set_title('Ø Verstoßquote pro Jahr', fontweight='bold')
ax.set_xlabel('Jahr')
ax.set_ylabel('Verstoßquote (%)')
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for x, y in zip(jahre, trend['verstoessquote']):
    ax.text(x, y + 0.03, f'{y:.2f}%', ha='center', va='bottom', fontsize=8)

# ── Subplot 3: Einnahmen ──────────────────────────────────────────────────────
ax = axes[2]
einnahmen = trend['einnahmen_gesamt'].fillna(0)
ax.bar(jahre, einnahmen, color='steelblue', edgecolor='white')

z3 = np.polyfit(jahre[einnahmen > 0], einnahmen[einnahmen > 0], 1)
p3 = np.poly1d(z3)
ax.plot(jahre, p3(jahre), color='red', linestyle='--', linewidth=2, label='Trend')

ax.set_title('Einnahmen pro Jahr (€)', fontweight='bold')
ax.set_xlabel('Jahr')
ax.set_ylabel('Einnahmen (€)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(ax.patches, einnahmen):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{int(val):,}€', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('plot2_trendanalyse.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gespeichert als plot2_trendanalyse.png ✓")

# plot 3: Verteilung der Verstoßquote - Wie ist die Verstoßquote über alle Standorte und Monate verteilt?
data = df_all['verstoessquote'].dropna()

plt.figure(figsize=(7,5))
plt.hist(data, bins=20, color='steelblue', edgecolor='white', alpha=0.7)

plt.title('Verteilung der Verstoßquote', fontweight='bold')
plt.xlabel('Verstoßquote (%)')
plt.ylabel('Häufigkeit')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plot3_verstoessquote_verteilung.png', dpi=150, bbox_inches='tight')
plt.show()

# plot 4: Verstoßquote vs durchschnittliches Bußgeld - Gibt es einen Zusammenhang zwischen der Verstoßquote und dem durchschnittlichen Bußgeld pro Verstoß?
plt.figure(figsize=(7,5))

df_clean = df_all[['verstoessquote', 'avg_fine']].dropna()
x = df_clean['verstoessquote']
y = df_clean['avg_fine']

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

plt.scatter(x, y, alpha=0.4)

# Regression
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, p(x_line), linewidth=2)

plt.title('Verstoßquote vs durchschnittliches Bußgeld', fontweight='bold')
plt.xlabel('Verstoßquote (%)')
plt.ylabel('Ø Bußgeld pro Verstoß (€)')
plt.grid(linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot4_verstoessquote_avg_fine.png', dpi=150, bbox_inches='tight')
plt.show()

# plot 5: Einnahmen vs Verstöße - Wie hängen die Einnahmen mit der Anzahl der gültigen Verstöße zusammen? Gibt es einen linearen Zusammenhang?

df_model = df_all[['gueltige_verstoesse', 'einnahmen']].dropna()

x = df_model['gueltige_verstoesse']
y = df_model['einnahmen']

# Regression
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

y_pred = p(x)

# R²
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)

print(f"R²: {r2:.3f}")

# Plot
plt.figure(figsize=(7,5))
plt.scatter(x, y, alpha=0.4)

x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, p(x_line), linewidth=2)

plt.title(f'Einnahmen vs Verstöße (R² = {r2:.2f})', fontweight='bold')
plt.xlabel('Gültige Verstöße')
plt.ylabel('Einnahmen (€)')
plt.grid(linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot5_einnahmen_vs_verstoesse.png', dpi=150, bbox_inches='tight')
plt.show()