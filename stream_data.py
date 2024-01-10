import pandas as pd
import requests
import json


csv_file_path = '../DataSet/foodDemand_train/train1 - Copy.csv'

last_row_file = 'last_row.txt'

# Specify the Power BI API URL
url = 'https://api.powerbi.com/beta/09bd1956-edda-4e9a-9543-7c7aa2cf4e81/datasets/56e8e89f-c3bb-4559-aae0-eb5aa672d270/rows?experience=power-bi&key=gMWLnB7SFRV9BAiFopeCKXh84M%2FTN7VTOxK05eavB0QbgXw3uJZhwAhPvcA%2FFS73i3IBrWppI1KAmUhXLhYR0g%3D%3D'

num_batches = int(input("Enter the number of batches to send: "))

batch_size = 10000

# Read the last processed row from the text file1
try:
    with open(last_row_file, 'r') as f:
        last_processed_row = int(f.read())
except:
    last_processed_row = 0


for i in range(num_batches):
    chunk = next(pd.read_csv(csv_file_path, skiprows=range(1, last_processed_row + 1), chunksize=batch_size), None)

    # Check if there is data in the chunk
    if chunk is not None and not chunk.empty:
        chunk_data = chunk.to_dict(orient='records')

        r = requests.post(url, json=chunk_data)

        print(r.text)
        print(r.status_code)

        last_processed_row += len(chunk)
        with open(last_row_file, 'w') as f:
            f.write(str(last_processed_row))
    else:
        print("No more data to send.")
        break
