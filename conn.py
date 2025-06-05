import mysql.connector
from mysql.connector import Error

def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host='aiml-surveillance.czoey2saeds6.ap-south-1.rds.amazonaws.com',        # Change to your DB host
            user='admin',    # Change to your DB username
            password='vtpl1234',# Change to your DB password
            database='CV_surveillance' # Change to your DB name
        )

        if connection.is_connected():
            print("✅ Connected to MySQL database")
            db_info = connection.get_server_info()
            print("Server version:", db_info)

            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print("You're connected to:", record)

    except Error as e:
        print("❌ Error while connecting to MySQL", e)

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("🔌 MySQL connection closed")

if __name__ == "__main__":
    connect_to_database()
