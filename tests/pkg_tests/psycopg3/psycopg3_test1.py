import psycopg
print(psycopg.__version__)

dsn="postgresql://eur_pg:nee9eiGh@localhost:5432/test"

    # dbname="test",
    # user="eur_pg",
    # password="nee9eiGh",
    # host="localhost",
    # port="5432",
# conn = psycopg.connect(
#     dbname="test",
#     user="eur_pg",
#     password="nee9eiGh",
#     host="74.248.128.241",
#     port="5432",
#     connect_timeout=15,  # czas w sekundach
#     autocommit=True
# )




# Example of using psycopg3 to connect to a PostgreSQL database
def test_psycopg3():
    with psycopg.connect(dsn, autocommit=True
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            record = cur.fetchone()
            print("You are connected to - ", record, "\n")

    print("Psycopg3 connection test successful.")

if __name__ == "__main__":
    test_psycopg3()

