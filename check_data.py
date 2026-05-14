import pandas as pd

df = pd.read_csv("training_data.csv")

print(f"Total examples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\n--- SAMPLE 1 ---")
print(df.iloc[0]['prompt'])
print(df.iloc[0]['completion'])
print("\n--- SAMPLE 2 ---")
print(df.iloc[50]['prompt'])
print(df.iloc[50]['completion'])
print("\n--- SAMPLE 3 ---")
print(df.iloc[200]['prompt'])
print(df.iloc[200]['completion'])