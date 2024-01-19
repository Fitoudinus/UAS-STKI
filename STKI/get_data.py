
import pandas as pd

nama_file = 'pahlawan.csv'
df = pd.read_csv(nama_file)

master_data = []

for index, row in df.iterrows():
    master_data.append(
        f'{row["name"]}. {row["description"]}'
    )

print(master_data[0])