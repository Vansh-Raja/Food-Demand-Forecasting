from pymongo import MongoClient
from prettytable import PrettyTable

def display_top_entries(collection, limit=20):
    try:
        results = collection.find({}).limit(limit)

        count = collection.count_documents({})

        if count > 0:
            columns = results[0].keys()
            table = PrettyTable(columns)
            table.align = 'l'
            for row in results:
                table.add_row(row.values())
            print(table)
        else:
            print("No results found.")

    except Exception as e:
        print(f"Error executing query: {e}")

# MongoDB connection parameters
host = "localhost"
port = 27017
database = "stud"
collection_name = "fooddel"

client = None

try:
    # Connect to MongoDB
    client = MongoClient(host, port)
    db = client[database]
    collection = db[collection_name]

    # Display the top 20 entries
    display_top_entries(collection, limit=20)

except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

finally:
    if client:
        client.close()
