import pandas as pd


FILE_PATH = "data/hbta_configuration_metadata.csv"


df = pd.read_csv(FILE_PATH)


print("=" * 70)
print("HBTA CONFIGURATION SUMMARY")
print("=" * 70)


print("\nTotal configurations:")
print(len(df))


print("\nDamage-state counts:")
print(df["damage_state"].value_counts().sort_index())


print("\nMVS mode counts:")
print(df["mvs_mode"].value_counts())


print("\nPosition counts:")
print(df["mvs_position"].value_counts())


print("\nDirection counts:")
print(df["mvs_direction"].value_counts())


print("\nTrain-passing counts:")
print(df["train_passing"].value_counts())


print("\nWind-range counts:")
print(df["wind_range"].value_counts())


print("\nDamage state × MVS mode:")
print(
    pd.crosstab(
        df["damage_state"],
        df["mvs_mode"]
    )
)


print("\nDamage state × position:")
print(
    pd.crosstab(
        df["damage_state"],
        df["mvs_position"]
    )
)


print("\nDamage state × direction:")
print(
    pd.crosstab(
        df["damage_state"],
        df["mvs_direction"]
    )
)


print("\n" + "=" * 70)