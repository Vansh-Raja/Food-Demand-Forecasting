from sqlalchemy import create_engine as ce
from sqlalchemy import text
import pandas as pd
import json

def sqlite_to_json(conn, table_name):
    q = f"SELECT * FROM {table_name};"
    result = conn.execute(text(q))
    rows = result.fetchall()
    column_names = result.keys()
    data = [dict(zip(column_names, row)) for row in rows]
    json_string = json.dumps(data, indent=2)
    return json_string

def csv_to_df(name, table_name): 
    data = pd.read_csv(name)
    data.to_sql(table_name, engine, index=False, if_exists='replace')

def drop_table(conn, table_name):
    query = f"DROP TABLE IF EXISTS {table_name};"
    conn.execute(text(query))
    print(f"Table {table_name} dropped.")

def enginesql():
    db_name= input("What is your db name? (must be in the same directory if present)")
    return ce('sqlite:///'+ db_name +'.db')

def printall():
    q = "SELECT * FROM table1"
    result = conn.execute(text(q))
    for i in result.fetchall():
        print(i)

def update(conn, table_name, column, value, condition):
    q = f"UPDATE {table_name} SET {column} = '{value}' WHERE {condition};"
    conn.execute(text(q))
    conn.commit()
    print("Values updated")

def delete_row(conn, table, condition):
    q = f"DELETE FROM {table} WHERE {condition};"
    conn.execute(text(q))
    conn.commit()
    print("Row deleted")

def menu():
    try:
        print("------------------------------------------------------------------------------------------------------------------------")
        print("\n                                          WELCOME TO MYSQLLITE AUTOMATION PROGRAM                                     \n")
        print("------------------------------------------------------------------------------------------------------------------------\n")
        print()
        n = int(input("1. Create Table from csv\n2. Printall\n3. Update values\n4. Delete Row \n5. Close \n6. Drop Table  \n7. Convert SQL to json \n\nEnter your comand: "))
        if n==1:
            name = input("Enter csv file name: ")
            table_name = input("Table name: ")
            csv_to_df(name, table_name)
            print("completed import")
        if n==2:
            printall()
        if n==3:
            conn = engine.connect()
            column = input("Enter column to edit: ")
            val = input("Enter updated value: ")
            table_name = input("Table name: ")
            condition = input("conditon: ")
            update(conn, table_name, column, val, condition)
            conn.close() 
        if n==4:
            table = input("Enter table name: ")
            conn = engine.connect()
            condition = input("Enter condition: ")
            delete_row(conn, table, condition)
            conn.close()
        if n==5:
            print("Closing menu...")
        if n==6:
            conn = engine.connect()
            table_name= input("Enter table name: ")
            drop_table(conn, table_name)
        if n==7:
            conn = engine.connect()
            table_name = "table1"
            json_data = sqlite_to_json(conn, table_name)
            print(json_data)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    


if __name__ == "__main__":
    engine = enginesql()
    conn = engine.connect()

    menu()
    conn.close()
