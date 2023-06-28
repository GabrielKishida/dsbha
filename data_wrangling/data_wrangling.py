import pandas as pd
df_raw = pd.read_csv('tobacco.csv')

print("Task 1: Find the ratio of the number of missing records over the number of all records:\n")
df_missing = df_raw[df_raw['Response'].isnull()]
ratio = len(df_missing)/len(df_raw)
print("Ratio is:", round(ratio,3))

print("Task 2: Drop the missing records in \"Response\" column and print the ratio again:\n")
df_clean = df_raw.dropna(subset="Response")
df_missing = df_clean[df_clean['Response'].isnull()]
ratio = len(df_missing)/len(df_clean)
print("Ratio is:", round(ratio,3))

print("Task 3: Print the unique values of the \"Race\" column and replace them with numeric ID:\n")
races = df_raw["Race"].unique()
races_id = [i for i in range(0, len(races))]
df_races_sorted = df_raw
print("Before ID attribution:")
print(df_races_sorted["Race"],"\n")
df_races_sorted["Race"] = df_races_sorted["Race"].replace(races,races_id)
print("After ID attribution:")
print(df_races_sorted["Race"])