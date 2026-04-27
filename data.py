import pandas as pd
import os

ordner = "./Data"

MONATE = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni',
          'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']

KERN_SPALTEN = ['standort', 'monat', 'gemessene_fahrzeuge', 'verstoesse',
                'gueltige_verstoesse', 'verwarnungen', 'bussgelder',
                'max_geschwindigkeit', 'einnahmen', 'jahr']

dfs = []
for year in [2018, 2019, 2020, 2021, 2022, 2023]:
    pfad = os.path.join(ordner, f"{year}.csv")
    df = pd.read_csv(pfad, sep=';', decimal=',', thousands='.', encoding='latin-1')
    df['jahr'] = year

    # Monat normalisieren
    df['monat'] = df['monat'].str.strip().str.replace('Maerz', 'März', regex=False)

    # Jahressummenzeilen rausfiltern (monat == 'gesamt' oder 'Gesamt')
    df = df[~df['monat'].str.lower().eq('gesamt')]

    # Standort 'gesamt' rausfiltern
    df = df[df['standort'].str.lower() != 'gesamt']

    # Fehlende Spalten mit NaN auffüllen
    for col in ['max_geschwindigkeit', 'einnahmen']:
        if col not in df.columns:
            df[col] = float('nan')

    # Nur Kernspalten behalten
    df = df[[c for c in KERN_SPALTEN if c in df.columns]]

    # NaN-Zeilen raus (Stationen die in dem Monat nicht aktiv waren)
    df = df[df['gemessene_fahrzeuge'].notna() & (df['gemessene_fahrzeuge'] > 0)]

    dfs.append(df)
    print(f"{year}: {len(df)} Zeilen ✓")

df_all = pd.concat(dfs, ignore_index=True)

# Monat als geordnete Kategorie
df_all['monat'] = pd.Categorical(df_all['monat'], categories=MONATE, ordered=True)

# Verstoßquote
df_all['verstoessquote'] = df_all['gueltige_verstoesse'] / df_all['gemessene_fahrzeuge'] * 100

print(f"\nGesamt: {len(df_all)} Zeilen")
print(f"Standorte: {sorted(df_all['standort'].unique())}")
print(f"Jahre: {sorted(df_all['jahr'].unique())}")
print(f"Fehlende Einnahmen: {df_all['einnahmen'].isna().sum()} Zeilen")
