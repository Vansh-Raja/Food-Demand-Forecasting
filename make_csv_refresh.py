import pandas as pd

df = pd.read_csv('train1 - Copy.csv')

# write first 10000 rows to csv
df[:10000].to_csv('train1.csv', index=False)