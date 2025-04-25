import pandas as pd
import sqlite3
import logging
import os
import argparse
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sqlalchemy import create_engine, text # Using SQLAlchemy for easier type mapping
import psycopg2
from data_utils.general_utils import _create_table_if_not_exists, insertIntoTable

# os.environ['GOOGLE_CREDENTIALS_PATH'] = "~/airflow/credentials/google_credentials.json"

# --- Configuration ---
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# SHEET_ID = os.environ.get("GOOGLE_SHEET_ID") 
SHEET_NUMBER = os.environ.get("GOOGLE_SHEET_NUMBER", "Sheet3") 
# CREDENTIALS_PATH = os.path.expanduser(os.environ.get("GOOGLE_CREDENTIALS_PATH", "/opt/airflow/credentials/google_credentials.json")) 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 


def fetch_data_from_google_sheets(sheet_id, expected_columns):
    """
    Fetches data from the configured Google Sheet and converts specific columns to appropriate types.
    """
    if not sheet_id:
        logging.error("Missing required environment variable: GOOGLE_SHEET_ID")
        raise ValueError("GOOGLE_SHEET_ID environment variable not set.")
    # if not os.path.exists(CREDENTIALS_PATH):
    #      logging.error(f"Google credentials file not found at: {CREDENTIALS_PATH}")
    #      raise FileNotFoundError(f"Credentials file not found: {CREDENTIALS_PATH}")

    # logging.info(f"Attempting to authorize Google Sheets API using {CREDENTIALS_PATH}")
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={SHEET_NUMBER}"
        df = pd.read_csv(url)
    except gspread.exceptions.APIError as e:
        logging.error(f"Google Sheets API error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error connecting to or reading from Google Sheets: {e}")
        raise

    if len(df)==0:
        logging.warning("No data found in the Google Sheet.")
        return pd.DataFrame()

    # df = pd.DataFrame(data)
    logging.info(f"Fetched {len(df)} rows from Google Sheet.")

    # Verify required columns exist
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logging.warning(f"Missing expected columns in fetched data: {missing_cols}. Proceeding with available columns.")

    # Convert empty strings to None 
    df = df.replace('', None)

    # --- Specific Data Type Conversion ---
    logging.info("Converting data types for specific columns...")

    # Columns expected to be numeric 
    # numeric_float_cols = [
    #     'listing_ceiling', 'listing_floor', 'listing_ref_price', 'listing_prior_close_price',
    #     'match_match_price', 'match_accumulated_value', 'match_avg_match_price',
    #     'match_highest', 'match_lowest', 'match_reference_price',
    #     'bid_ask_bid_1_price', 'bid_ask_bid_2_price', 'bid_ask_bid_3_price',
    #     'bid_ask_ask_1_price', 'bid_ask_ask_2_price', 'bid_ask_ask_3_price'
    # ]
    # Columns expected to be numeric 
    # numeric_int_cols = [
    #     'listing_listed_share', 'match_match_vol', 'match_accumulated_volume',
    #     'match_foreign_sell_volume', 'match_foreign_buy_volume',
    #     'match_current_room', 'match_total_room', 'match_total_accumulated_value', # Check if value can be float
    #     'match_total_accumulated_volume',
    #     'bid_ask_bid_1_volume', 'bid_ask_bid_2_volume', 'bid_ask_bid_3_volume',
    #     'bid_ask_ask_1_volume', 'bid_ask_ask_2_volume', 'bid_ask_ask_3_volume'
    # ]
    # Columns expected to be datetime objects
    # datetime_cols = [
    #     'Time', 'listing_last_trading_date', 'listing_trading_date'
    # ]

    # for col in numeric_float_cols:
    #     if col in df.columns:
    #         df[col] = pd.to_numeric(df[col], errors='coerce')

    # for col in numeric_int_cols:
    #      if col in df.columns:
    #         # Coerce to float first to handle potential decimals in source, then Int64 (nullable int)
    #         df[col] = pd.to_numeric(df[col], errors='coerce')
    #         df[col] = df[col].astype('Int64') # Use nullable integer type

    # for col in datetime_cols:
    #     if col in df.columns:
    #         df[col] = pd.to_datetime(df[col], errors='coerce')
    #         if df[col].isnull().any():
    #             logging.warning(f"Column '{col}' contained values that could not be parsed as datetime.")

    # # Ensure string columns are explicitly strings 
    # string_cols = [
    #     'listing_symbol', 'listing_stock_type', 'listing_exchange', 'listing_type',
    #     'listing_id', 'listing_organ_name', 'listing_benefit', 
    #     'match_match_type'
    # ]
    # for col in string_cols:
    #     if col in df.columns:
    #         df[col] = df[col].astype(str).replace({'nan': None, 'None': None})


    # logging.info(f"Data types converted. DataFrame info:\n")
    df.info(verbose=True, show_counts=True) 
    logging.info(f"DataFrame head:\n{df.head().to_string()}")

    # Reorder columns to expected order if necessary 
    present_expected_cols = [col for col in expected_columns if col in df.columns]
    df = df[present_expected_cols]

    return df


# def _create_table_if_not_exists(db_name, df, db_uri, table_type = 'train'):
#     db_config = {
#     'dbname': db_name,
#     'user': 'huyvu',
#     'password': 'password',
#     'host': 'localhost',  # Use 'localhost' or your DB host
#     'port': 5432  # Default PostgreSQL port
#     }
#     conn = psycopg2.connect(**db_config)
#     cur = conn.cursor()
#     # === 1. Create the table based on DataFrame columns ===

#     # Generate column definitions dynamically based on DataFrame types
#     columns = []
#     for col, dtype in zip(df.columns, df.dtypes):
#         if dtype == 'int64':
#             col_type = 'INTEGER'
#         elif dtype == 'float64':
#             col_type = 'FLOAT'
#         elif dtype == 'datetime64[ns]':
#             col_type = 'TIMESTAMP'
#         else:
#             col_type = 'TEXT'  # Default to TEXT for string-like columns

#         columns.append(f"{col} {col_type}")

#     # Combine columns to form the CREATE TABLE statement
#     create_table_query = f'''
#     CREATE TABLE IF NOT EXISTS {table_type}_data (
#         {', '.join(columns)}
#     );
#     '''
#     engine = create_engine(db_uri)

#     with engine.connect() as conn:
#         conn.execute(create_table_query)  # Creating the table if it doesn't exist


# def insertIntoTable(db_name, df, table):
#     db_config = {
#         'dbname': db_name,
#         'user': 'huyvu',
#         'password': 'password',
#         'host': 'localhost',
#         'port': 5432
#     }

#     conn = psycopg2.connect(**db_config)
#     cur = conn.cursor()

#     # Fetch existing 'time' values
#     cur.execute(f"SELECT * FROM {table}")
#     existing_times = set(str(row[0]) for row in cur.fetchall())
#     print(existing_times)

#     # Filter out rows with existing 'time' values
#     new_rows = df[~df['Time'].isin(existing_times)]

#     if new_rows.empty:
#         print("No new rows to insert.")
#         cur.close()
#         conn.close()
#         return

#     tuples = [tuple(x) for x in new_rows.to_numpy()]
#     cols = ', '.join(new_rows.columns)
#     placeholders = ', '.join(['%s'] * len(new_rows.columns))
#     query = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"

#     try:
#         cur.executemany(query, tuples)
#         conn.commit()
#         print(f"{len(tuples)} new rows inserted.")
#     except Exception as error:
#         print("Error:", error)
#         conn.rollback()
#     finally:
#         cur.close()
#         conn.close()

#     return len(new_rows)


# --- Main Function for DAG Task ---
def ingest_data(db_name, sheet_id, table_name):
    """
    Orchestrator function called by the Airflow task.
    Fetches data from Google Sheets, converts types, and appends (deduplicated)
    into the golden SQLite database.
    """
    logging.info("Starting ingestion process (append, deduplicated)...")
    # Define the list of expected columns based on user input
    expected_columns = [
        'Time', 'listing_symbol', 'listing_ceiling', 'listing_floor', 'listing_ref_price',
        'listing_stock_type', 'listing_exchange', 'listing_last_trading_date',
        'listing_listed_share', 'listing_type', 'listing_id', 'listing_organ_name',
        'listing_prior_close_price', 'listing_benefit', 'listing_trading_date',
        'match_match_price', 'match_match_vol', 'match_accumulated_volume',
        'match_accumulated_value', 'match_avg_match_price', 'match_highest', 'match_lowest',
        'match_match_type', 'match_foreign_sell_volume', 'match_foreign_buy_volume',
        'match_current_room', 'match_total_room', 'match_total_accumulated_value',
        'match_total_accumulated_volume', 'match_reference_price', 'bid_ask_bid_1_price',
        'bid_ask_bid_1_volume', 'bid_ask_bid_2_price', 'bid_ask_bid_2_volume',
        'bid_ask_bid_3_price', 'bid_ask_bid_3_volume', 'bid_ask_ask_1_price',
        'bid_ask_ask_1_volume', 'bid_ask_ask_2_price', 'bid_ask_ask_2_volume',
        'bid_ask_ask_3_price', 'bid_ask_ask_3_volume'
    ]
    try:
        print('Fetching Data')
        df = fetch_data_from_google_sheets(sheet_id, expected_columns)
        df.rename(columns={'Time': 'time'}, inplace= True)
        print('Creating table')
        _create_table_if_not_exists(db_name, df, table_type = table_name)
        print('Inserting Data')
        inserted_count = insertIntoTable(db_name, df, table_name)
        logging.info(f"Ingestion process completed. Approximately {inserted_count} new rows inserted.")
    except Exception as e:
        logging.error(f"Ingestion process failed: {e}")
        raise


# --- Command Line Execution (for testing) ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Fetch data from Google Sheets and append (deduplicated) into SQLite database."
#     )
#     parser.add_argument(
#         "--output-db", required=True, help="Path to the output SQLite database file (Golden DB)."
#     )
#     parser.add_argument(
#         "--table", default="stock_data", help="SQLite table name (default: stock_data)."
#     )
#     parser.add_argument(
#         "--key-cols", required=True, nargs='+', help="Column name(s) used as the unique key for deduplication."
#     )

#     args = parser.parse_args()

#     os.makedirs(os.path.dirname(args.output_db), exist_ok=True)
#     ingest_data(args.output_db, args.table, args.key_cols)
