import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from scipy.stats import linregress
from scipy.stats import probplot

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

df = pd.read_csv(DATA_DIR / "world_happiness.csv")
df['country'] = (
    df['country']
    .str.strip()
    .str.replace('’', "'", regex=False)
)

# ── Kontinente zuweisen ───────────────────────────────────────────────────────
kontinent_map = {
    'Finland': 'Europa', 'Denmark': 'Europa', 'Iceland': 'Europa',
    'Sweden': 'Europa', 'Netherlands': 'Europa', 'Norway': 'Europa',
    'Luxembourg': 'Europa', 'Switzerland': 'Europa', 'Austria': 'Europa',
    'Germany': 'Europa', 'Ireland': 'Europa', 'Belgium': 'Europa',
    'Czechia': 'Europa', 'United Kingdom': 'Europa', 'Lithuania': 'Europa',
    'Slovenia': 'Europa', 'Poland': 'Europa', 'France': 'Europa',
    'Romania': 'Europa', 'Estonia': 'Europa', 'Spain': 'Europa',
    'Italy': 'Europa', 'Slovakia': 'Europa', 'Latvia': 'Europa',
    'Croatia': 'Europa', 'Hungary': 'Europa', 'Portugal': 'Europa',
    'Greece': 'Europa', 'Serbia': 'Europa', 'Bulgaria': 'Europa',
    'Montenegro': 'Europa', 'North Macedonia': 'Europa', 'Albania': 'Europa',
    'Bosnia and Herzegovina': 'Europa', 'Kosovo': 'Europa',
    'Cyprus': 'Europa', 'Malta': 'Europa', 'Belarus': 'Europa',
    'Ukraine': 'Europa', 'Russian Federation': 'Europa',
    'Moldova': 'Europa', 'Republic of Moldova': 'Europa',

    'United States': 'Nordamerika', 'Canada': 'Nordamerika',
    'Mexico': 'Nordamerika', 'Belize': 'Nordamerika',

    'Brazil': 'Südamerika', 'Argentina': 'Südamerika', 'Chile': 'Südamerika',
    'Colombia': 'Südamerika', 'Peru': 'Südamerika', 'Uruguay': 'Südamerika',
    'Paraguay': 'Südamerika', 'Bolivia': 'Südamerika', 'Ecuador': 'Südamerika',
    'Venezuela': 'Südamerika', 'Guyana': 'Südamerika', 'Suriname': 'Südamerika',

    'Costa Rica': 'Mittelamerika', 'Panama': 'Mittelamerika',
    'Guatemala': 'Mittelamerika', 'Honduras': 'Mittelamerika',
    'El Salvador': 'Mittelamerika', 'Nicaragua': 'Mittelamerika',
    'Jamaica': 'Mittelamerika', 'Trinidad and Tobago': 'Mittelamerika',
    'Dominican Republic': 'Mittelamerika', 'Puerto Rico': 'Mittelamerika',
    'Haiti': 'Mittelamerika', 'Cuba': 'Mittelamerika',

    'China': 'Asien', 'Japan': 'Asien', 'Republic of Korea': 'Asien',
    'Singapore': 'Asien', 'Taiwan Province of China': 'Asien',
    'Hong Kong SAR of China': 'Asien', 'Thailand': 'Asien',
    'Viet Nam': 'Asien', 'Indonesia': 'Asien', 'Malaysia': 'Asien',
    'Philippines': 'Asien', 'Cambodia': 'Asien', 'Myanmar': 'Asien',
    'Lao PDR': 'Asien', 'Mongolia': 'Asien', 'Bangladesh': 'Asien',
    'India': 'Asien', 'Nepal': 'Asien', 'Sri Lanka': 'Asien',
    'Pakistan': 'Asien', 'Afghanistan': 'Asien', 'Kazakhstan': 'Asien',
    'Uzbekistan': 'Asien', 'Kyrgyzstan': 'Asien', 'Tajikistan': 'Asien',
    'Turkmenistan': 'Asien', 'Azerbaijan': 'Asien', 'Armenia': 'Asien',
    'Georgia': 'Asien', 'Iran': 'Asien', 'Iraq': 'Asien',
    'Jordan': 'Asien', 'Lebanon': 'Asien', 'Israel': 'Asien',
    'Saudi Arabia': 'Asien', 'United Arab Emirates': 'Asien',
    'Kuwait': 'Asien', 'Bahrain': 'Asien', 'Qatar': 'Asien',
    'Oman': 'Asien', 'Turkey': 'Asien', 'Türkiye': 'Asien',
    'North Cyprus': 'Asien', 'Cyprus': 'Europa',
    'State of Palestine': 'Asien', 'Yemen': 'Asien', 'Syria': 'Asien',
    'Bhutan': 'Asien', 'Maldives': 'Asien',
    

    'Nigeria': 'Afrika', 'Ethiopia': 'Afrika', 'Ghana': 'Afrika',
    'Kenya': 'Afrika', 'Tanzania': 'Afrika', 'Uganda': 'Afrika',
    'South Africa': 'Afrika', 'Cameroon': 'Afrika', 'Senegal': 'Afrika',
    'Mali': 'Afrika', 'Burkina Faso': 'Afrika', 'Niger': 'Afrika',
    'Chad': 'Afrika', 'Madagascar': 'Afrika', 'Malawi': 'Afrika',
    'Zambia': 'Afrika', 'Zimbabwe': 'Afrika', 'Mozambique': 'Afrika',
    'Rwanda': 'Afrika', 'Burundi': 'Afrika', 'DR Congo': 'Afrika',
    'Congo': 'Afrika', 'Côte d\'Ivoire': 'Afrika', 'Guinea': 'Afrika',
    'Sierra Leone': 'Afrika', 'Liberia': 'Afrika', 'Togo': 'Afrika',
    'Benin': 'Afrika', 'Mauritania': 'Afrika', 'Morocco': 'Afrika',
    'Algeria': 'Afrika', 'Tunisia': 'Afrika', 'Egypt': 'Afrika',
    'Libya': 'Afrika', 'Sudan': 'Afrika', 'South Sudan': 'Afrika',
    'Angola': 'Afrika', 'Namibia': 'Afrika', 'Botswana': 'Afrika',
    'Lesotho': 'Afrika', 'Eswatini': 'Afrika', 'Gabon': 'Afrika',
    'Mauritius': 'Afrika', 'Comoros': 'Afrika', 'Somalia': 'Afrika',
    'Gambia': 'Afrika', 'Djibouti': 'Afrika', 'Central African Republic': 'Afrika',
    'Swaziland': 'Afrika', 'Côte d’Ivoire': 'Afrika', 'Somaliland Region': 'Afrika',

    'Australia': 'Ozeanien', 'New Zealand': 'Ozeanien',
}

df['kontinent'] = df['country'].map(kontinent_map).dropna()

FARBEN = {
    'Europa': 'steelblue', 'Nordamerika': 'orange', 'Südamerika': 'green',
    'Mittelamerika': 'limegreen', 'Asien': 'red', 'Afrika': 'purple',
    'Ozeanien': 'brown'
}

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1-3: Übersicht
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('World Happiness Report – Erste Übersicht', fontsize=14, fontweight='bold')

ax = axes[0]
top10 = df[df['year'] == 2025].nsmallest(10, 'rank_in_year')
ax.barh(top10['country'][::-1], top10['happiness_score'][::-1], color='steelblue')
ax.set_title('Top 10 glücklichste Länder (2025)', fontweight='bold')
ax.set_xlabel('Happiness Score')
ax.set_xlim(6, 8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]

# Top 10 Länder aus 2025 holen
top10_laender = df[df['year'] == 2025].nsmallest(10, 'rank_in_year')['country'].tolist()

# Deutschland und Afghanistan ergänzen falls nicht drin
extra = ['Germany', 'Afghanistan']
alle_laender = top10_laender + [l for l in extra if l not in top10_laender]

# Farben: Top 10 in Blautönen, Extra in Signalfarben
cmap = plt.cm.tab20
farben_map = {land: cmap(i / len(top10_laender)) for i, land in enumerate(top10_laender)}
farben_map['Germany'] = 'orange'
farben_map['Afghanistan'] = 'red'

for land in alle_laender:
    d = df[df['country'] == land].sort_values('year')
    if d.empty:
        continue
    linestyle = '--' if land in extra and land not in top10_laender else '-'
    linewidth = 2.5 if land in ['Germany', 'Afghanistan'] else 1.5
    ax.plot(d['year'], d['happiness_score'],
            marker='o', markersize=3,
            label=land,
            color=farben_map.get(land, 'gray'),
            linewidth=linewidth,
            linestyle=linestyle)

ax.set_title('Happiness Score der Top 10 Länder (2025)', fontweight='bold')
ax.set_xlabel('Jahr')
ax.set_ylabel('Happiness Score')
ax.legend(fontsize=7, ncol=2, loc='lower left')
ax.grid(linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[2]
scores_2025 = df[df['year'] == 2025]['happiness_score'].dropna()
ax.hist(scores_2025, bins=20, color='steelblue', edgecolor='white')
ax.axvline(scores_2025.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mittelwert: {scores_2025.mean():.2f}')
ax.set_title('Verteilung Happiness Score (2025)', fontweight='bold')
ax.set_xlabel('Happiness Score')
ax.set_ylabel('Anzahl Länder')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(BASE_DIR / 'plot1_uebersicht.png', dpi=150, bbox_inches='tight')
plt.show()

# ===========================================================================
# Remove rows that contain 0.0 as value, which indicates missing values
# ===========================================================================
cols = [
    'happiness_score',
    'explained_log_gdp_per_capita',
    'explained_social_support',
    'explained_healthy_life_expectancy',
    'explained_freedom',
    'explained_generosity',
    'explained_corruption'
]

# Debug
# df_problem = df[(df[cols] == 0).any(axis=1)]
# print(df_problem)

df_clean = df.copy()
df_clean[cols] = df_clean[cols].replace(0, np.nan)
df_clean = df_clean.dropna(subset=cols)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4-9: Korrelationen mit Residualplot
# ═══════════════════════════════════════════════════════════════════════════════
def plot_correlation(ax, df, y_column, y_label):
    subset = df[df['year'] >= 2024][['happiness_score', y_column, 'country']].dropna()
    x = subset['happiness_score']
    y = subset[y_column]
    r_value = x.corr(y)
    r_squared = r_value**2
    slope, intercept, _, _, _ = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    train = df[(df['year'] >= 2019) & (df['year'] <= 2023)][['happiness_score', y_column]].dropna()
    test  = df[(df['year'] >= 2024) & (df['year'] <= 2025)][['happiness_score', y_column]].dropna()

    if len(train) > 0 and len(test) > 0:
        x_train = train[y_column]
        y_train = train['happiness_score']

        slope_t, intercept_t, _, _, _ = linregress(x_train, y_train)

        x_test = test[y_column]
        y_test = test['happiness_score']

        y_pred = slope_t * x_test + intercept_t

        ss_res = np.sum((y_test - y_pred)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2_test = 1 - ss_res / ss_tot
    else:
        r2_test = np.nan

    color = 'red' if abs(r_value) > 0.6 else 'gray'
    ax.scatter(x, y, alpha=0.6, color=color)
    ax.plot(x_line, y_line, color='black', linewidth=2, label='Regression (2024–2025)')
    ax.set_title(f'Happiness vs. {y_label}', fontweight='bold')
    ax.set_xlabel('Happiness Score')
    ax.set_ylabel(f'{y_label}')
    if y_label == 'Freiheit':
        r2_test = r2_test * (-1)
    ax.text(0.05, 0.95,
            f'r (2024–2025) = {r_value:.2f}\n'
            f'R² (2024–2025) = {r_squared:.2f}\n\n'
            f'Train: 2019–2023\n'
            f'Test: 2024–2025\n'
            f'R² (test) = {r2_test:.2f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.3))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('World Happiness Report – Korrelationen (2024-2025)', fontsize=14, fontweight='bold')

plot_correlation(axes[0][0], df_clean, 'explained_social_support', 'Soziale Unterstützung')
plot_correlation(axes[0][1], df_clean, 'explained_log_gdp_per_capita', 'logarithmiertes BIP pro Kopf')
plot_correlation(axes[0][2], df_clean, 'explained_healthy_life_expectancy', 'Gesunde Lebenserwartung')
plot_correlation(axes[1][0], df_clean, 'explained_freedom', 'Freiheit')
plot_correlation(axes[1][1], df_clean, 'explained_corruption', 'Korruption')
plot_correlation(axes[1][2], df_clean, 'explained_generosity', 'Generosität')

plt.tight_layout()
plt.savefig(BASE_DIR / 'plot2_korrelationen.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 10: Residualplot + QQ-Plot (Modellvalidierung)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Modellvalidierung: Happiness Score ~ Log. BIP pro Kopf (2025)', fontsize=14, fontweight='bold')

subset = df_clean[df_clean['year'] == 2025][['happiness_score', 'explained_log_gdp_per_capita']].dropna()
x = subset['explained_log_gdp_per_capita']
y = subset['happiness_score']
slope, intercept, r_value, p_value, std_err = linregress(x, y)
y_pred = slope * x + intercept
residuen = y - y_pred

# Residualplot
ax = axes[0]
ax.scatter(y_pred, residuen, alpha=0.6, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_title('Residualplot', fontweight='bold')
ax.set_xlabel('Vorhergesagter Happiness Score (ŷ)')
ax.set_ylabel('Residuen (y - ŷ)')
ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', alpha=0.3))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# QQ-Plot
ax = axes[1]
probplot(residuen, dist="norm", plot=ax)
ax.set_title('QQ-Plot der Residuen', fontweight='bold')
ax.set_xlabel('Theoretische Quantile (Normalverteilung)')
ax.set_ylabel('Sample Quantile')
ax.get_lines()[0].set(color='steelblue', alpha=0.6)
ax.get_lines()[1].set(color='red', linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(BASE_DIR / 'plot3_validierung.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 11: Multivariat – Scatter GDP × Happiness × Kontinent (Bubble)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))

fig.suptitle(
    'Multivariat: Log. BIP pro Kopf × Happiness Score × Lebenserwartung (2019–2025)',
    fontsize=14,
    fontweight='bold'
)

# Zeitraum filtern
df_avg = df_clean[
    (df_clean['year'] >= 2019) & (df_clean['year'] <= 2025)
].dropna(
    subset=[
        'explained_log_gdp_per_capita',
        'happiness_score',
        'explained_healthy_life_expectancy'
    ]
)

# Mittelwerte pro Land berechnen
df_avg = (
    df_avg
    .groupby(['country', 'kontinent'], as_index=False)
    .agg({
        'explained_log_gdp_per_capita': 'mean',
        'happiness_score': 'mean',
        'explained_healthy_life_expectancy': 'mean'
    })
)

# Plot
for kontinent, gruppe in df_avg.groupby('kontinent'):
    ax.scatter(
        gruppe['explained_log_gdp_per_capita'],
        gruppe['happiness_score'],
        s=gruppe['explained_healthy_life_expectancy'] * 300,
        color=FARBEN.get(kontinent, 'gray'),
        alpha=0.6,
        label=kontinent,
        edgecolors='white',
        linewidth=0.5
    )

ax.set_xlabel('Ø Log. BIP pro Kopf', fontsize=11)
ax.set_ylabel('Ø Happiness Score', fontsize=11)
ax.set_title('Punktgröße = Ø Lebenserwartung', fontsize=10, style='italic')

ax.legend(title='Kontinent', bbox_to_anchor=(1.01, 1), loc='upper left')

ax.grid(linestyle='--', alpha=0.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig(
    BASE_DIR / 'plot4_multivariat_bubble.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 12: Heatmap Korrelationsmatrix
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Korrelationsmatrix aller Faktoren (2024-2025)',
             fontsize=14, fontweight='bold')

faktoren = {
    'explained_log_gdp_per_capita': 'GDP',
    'explained_social_support': 'Social Support',
    'explained_healthy_life_expectancy': 'Life Expectancy',
    'explained_freedom': 'Freedom',
    'explained_generosity': 'Generosity',
    'explained_corruption': 'Corruption',
    'happiness_score': 'Happiness'
}


corr_df = df_clean[df_clean['year'] >= 2024][list(faktoren.keys())].dropna()
corr_df.columns = list(faktoren.values())
corr_matrix = corr_df.corr()
mask = np.abs(corr_matrix) < 0.3
corr_matrix_masked = corr_matrix.copy()
corr_matrix_masked[mask] = np.nan

order = ['Happiness', 'GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Corruption', 'Generosity']
corr_matrix = corr_matrix.loc[order, order]

im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearson r')

ax.set_xticks(range(len(corr_matrix)))
ax.set_yticks(range(len(corr_matrix)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.iloc[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=9, color='black' if abs(val) < 0.7 else 'white')

plt.tight_layout()
plt.savefig(BASE_DIR / 'plot5_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 13: Zeitreihe mit Trendlinie nach Kontinent
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('Happiness-Trend nach Kontinent (2011–2025)',
             fontsize=14, fontweight='bold')

kontinente = ['Europa', 'Asien', 'Afrika', 'Südamerika', 'Nordamerika', 'Ozeanien', 'Mittelamerika']

# Kurzer Check - füge das temporär vor Plot 6 ein
for k in ['Ozeanien', 'Mittelamerika']:
    d = df[df['kontinent'] == k].groupby('year')['happiness_score'].mean()
    print(f"{k}: {len(d)} Jahre, Werte: {d.values[:3]}")

for kontinent in kontinente:
    d = df[df['kontinent'] == kontinent].groupby('year')['happiness_score'].mean().reset_index()
    d = d.dropna()
    farbe = FARBEN.get(kontinent, 'gray')

    ax.plot(d['year'], d['happiness_score'], marker='o', markersize=5,
            label=kontinent, color=farbe, linewidth=2, alpha=0.8)

    # Trendlinie
    if len(d) > 2:
        slope, intercept, _, _, _ = linregress(d['year'], d['happiness_score'])
        x_line = np.array([d['year'].min(), d['year'].max()])
        ax.plot(x_line, slope * x_line + intercept,
                linestyle='--', color=farbe, linewidth=1.5, alpha=0.5)

ax.set_xlabel('Jahr', fontsize=11)
ax.set_ylabel('Ø Happiness Score', fontsize=11)
ax.legend(title='Kontinent')
ax.grid(linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(BASE_DIR / 'plot6_trend_kontinent.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAlle Plots gespeichert ✓")