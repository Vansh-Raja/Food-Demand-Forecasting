import mysql.connector
from prettytable import PrettyTable

def execute_query_and_display_results(cursor, query):
    try:
        cursor.execute(query)
        results = cursor.fetchall()

        if results:
            columns = [column[0] for column in cursor.description]
            table = PrettyTable(columns)
            table.align = 'l'
            for row in results:
                table.add_row(row)
            print(table)
        else:
            print("No results found.")

    except Exception as e:
        print(f"Error executing query: {e}")

host = "localhost"
user = "root"
password = "root"
database = "stud"

connection = None

try:
    connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
    cursor = connection.cursor()

    while True:
        user_query = input("Enter SQL query (type 'exit' to quit): ")

        if user_query.lower() == 'exit':
            break

        execute_query_and_display_results(cursor, user_query)

except mysql.connector.Error as e:
    print(f"Error connecting to MySQL: {e}")

finally:
    if connection:
        connection.close()
