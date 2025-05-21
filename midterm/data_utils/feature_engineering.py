import os
import logging
import pandas as pd
import numpy as np
import sqlite3
import argparse
from datetime import datetime
from sqlalchemy import create_engine, text # Using SQLAlchemy for easier type mapping
import psycopg2
from data_utils.general_utils import _create_table_if_not_exists, insertIntoTable

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
USER = os.getenv("USER") 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_features(df):
    """
    Engineers features based on listing info and order book depth.
    Assumes input DataFrame has relevant columns and 'Time' column is datetime.
    """
    if df.empty:
        logging.warning("Input DataFrame for feature creation is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df = df.copy()
    logging.info("Starting feature engineering...")
    start_time = datetime.now()

    # --- Date and Time Column Conversions ---
    # Ensure 'time' column is datetime type
    if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
        logging.warning("Converting 'time' column to datetime.")
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if df['time'].isnull().any():
             logging.warning("Some 'time' values could not be parsed and are set to NaT.")

    # Convert other relevant date columns
    date_cols_to_convert = ['listing_last_trading_date', 'listing_trading_date', 
                            'listing_sending_time', 'bid_ask_transaction_time', 'match_sending_time']
    for col_name in date_cols_to_convert:
        if col_name in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
                logging.info(f"Converting '{col_name}' column to datetime.")
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                if df[col_name].isnull().any():
                    logging.warning(f"Some '{col_name}' values could not be parsed and are set to NaT.")
        else:
            logging.debug(f"Date column '{col_name}' not found in DataFrame for conversion, will result in NaN for related features.")


    # --- Define required columns for feature engineering ---
    # These columns are essential for the calculations that follow.
    required_cols = [
        'time',
        'bid_ask_bid_1_price', 'bid_ask_ask_1_price', 'bid_ask_bid_1_volume', 'bid_ask_ask_1_volume',
        'bid_ask_bid_2_price', 'bid_ask_ask_2_volume', 'bid_ask_ask_2_price', 'bid_ask_ask_2_volume',
        'bid_ask_bid_3_price', 'bid_ask_bid_3_volume', 'bid_ask_ask_3_price', 'bid_ask_ask_3_volume',
        'listing_ceiling', 'listing_floor', 'listing_ref_price', 'match_match_price',
        'match_foreign_buy_volume', 'match_foreign_sell_volume',
        'match_foreign_buy_value', 'match_foreign_sell_value',
        'listing_trading_status', 'listing_last_trading_date', 'listing_trading_date'
    ]
    
    # Check for essential columns, error out if missing
    essential_for_core_features = [
        'bid_ask_bid_1_price', 'bid_ask_ask_1_price', 'bid_ask_bid_1_volume', 'bid_ask_ask_1_volume'
    ]
    missing_essential_cols = [col for col in essential_for_core_features if col not in df.columns]
    if missing_essential_cols:
        logging.error(f"Missing essential columns for feature engineering: {missing_essential_cols}")
        raise ValueError(f"Missing essential columns: {missing_essential_cols}")

    # For other columns, if missing, features relying on them will be NaN
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"Missing some columns for feature engineering: {missing_cols}. Related features will be NaN.")
        for col in missing_cols: # Add missing columns as NaN to prevent KeyErrors later
            df[col] = np.nan


    epsilon = 1e-9  # Small value to avoid division by zero

    # --- Feature Engineering Logic ---

    # 1. Basic Price and Spread Features (L1, L2, L3)
    logging.info("Calculating L1, L2, L3 price & spread features...")
    # Level 1
    df['mid_price_l1'] = (df['bid_ask_bid_1_price'] + df['bid_ask_ask_1_price']) / 2
    vol_sum_l1 = df['bid_ask_bid_1_volume'] + df['bid_ask_ask_1_volume']
    numerator_microprice_l1 = (df['bid_ask_bid_1_price'] * df['bid_ask_ask_1_volume'] +
                               df['bid_ask_ask_1_price'] * df['bid_ask_bid_1_volume'])
    df['microprice_l1'] = np.where(vol_sum_l1 > epsilon, numerator_microprice_l1 / vol_sum_l1, df['mid_price_l1'])
    df['spread_l1'] = df['bid_ask_ask_1_price'] - df['bid_ask_bid_1_price']
    df['relative_spread_l1'] = np.where(df['mid_price_l1'] > epsilon, df['spread_l1'] / df['mid_price_l1'], np.nan)

    # Level 2
    df['mid_price_l2'] = (df['bid_ask_bid_2_price'] + df['bid_ask_ask_2_price']) / 2
    df['spread_l2'] = df['bid_ask_ask_2_price'] - df['bid_ask_bid_2_price']
    df['relative_spread_l2'] = np.where(df['mid_price_l2'] > epsilon, df['spread_l2'] / df['mid_price_l2'], np.nan)

    # Level 3
    df['mid_price_l3'] = (df['bid_ask_bid_3_price'] + df['bid_ask_ask_3_price']) / 2
    df['spread_l3'] = df['bid_ask_ask_3_price'] - df['bid_ask_bid_3_price']
    df['relative_spread_l3'] = np.where(df['mid_price_l3'] > epsilon, df['spread_l3'] / df['mid_price_l3'], np.nan)

    # 2. Volume and Liquidity Features
    logging.info("Calculating volume & liquidity features...")
    df['total_bid_volume_l1'] = df['bid_ask_bid_1_volume']
    df['total_ask_volume_l1'] = df['bid_ask_ask_1_volume']
    df['total_bid_volume_3lv'] = df['bid_ask_bid_1_volume'] + df['bid_ask_bid_2_volume'] + df['bid_ask_bid_3_volume']
    df['total_ask_volume_3lv'] = df['bid_ask_ask_1_volume'] + df['bid_ask_ask_2_volume'] + df['bid_ask_ask_3_volume']

    df['market_depth_value_bid_l1'] = df['bid_ask_bid_1_price'] * df['bid_ask_bid_1_volume']
    df['market_depth_value_ask_l1'] = df['bid_ask_ask_1_price'] * df['bid_ask_ask_1_volume']
    df['market_depth_value_bid_3lv'] = (df['bid_ask_bid_1_price'] * df['bid_ask_bid_1_volume'] +
                                      df['bid_ask_bid_2_price'] * df['bid_ask_bid_2_volume'] +
                                      df['bid_ask_bid_3_price'] * df['bid_ask_bid_3_volume'])
    df['market_depth_value_ask_3lv'] = (df['bid_ask_ask_1_price'] * df['bid_ask_ask_1_volume'] +
                                      df['bid_ask_ask_2_price'] * df['bid_ask_ask_2_volume'] +
                                      df['bid_ask_ask_3_price'] * df['bid_ask_ask_3_volume'])

    # 3. Weighted Average Prices and Imbalances
    logging.info("Calculating weighted average prices and imbalances...")
    total_bid_vol_3lv_safe = df['total_bid_volume_3lv'].replace(0, epsilon)
    total_ask_vol_3lv_safe = df['total_ask_volume_3lv'].replace(0, epsilon)

    df['weighted_avg_price_3lv_bid'] = df['market_depth_value_bid_3lv'] / total_bid_vol_3lv_safe
    df.loc[df['total_bid_volume_3lv'] == 0, 'weighted_avg_price_3lv_bid'] = np.nan
    df['weighted_avg_price_3lv_ask'] = df['market_depth_value_ask_3lv'] / total_ask_vol_3lv_safe
    df.loc[df['total_ask_volume_3lv'] == 0, 'weighted_avg_price_3lv_ask'] = np.nan

    sum_total_volume_3lv = df['total_bid_volume_3lv'] + df['total_ask_volume_3lv']
    df['volume_imbalance_3lv'] = np.where(
        sum_total_volume_3lv > epsilon,
        (df['total_bid_volume_3lv'] - df['total_ask_volume_3lv']) / sum_total_volume_3lv,
        np.nan
    )
    sum_market_depth_value_3lv = df['market_depth_value_bid_3lv'] + df['market_depth_value_ask_3lv']
    df['value_imbalance_3lv'] = np.where(
        sum_market_depth_value_3lv > epsilon,
        (df['market_depth_value_bid_3lv'] - df['market_depth_value_ask_3lv']) / sum_market_depth_value_3lv,
        np.nan
    )

    # 4. Price-to-Listing Ratios
    logging.info("Calculating price-to-listing ratios...")
    df['price_to_listing_ceiling_ratio'] = np.where(df['listing_ceiling'].fillna(0) > epsilon, df['match_match_price'] / df['listing_ceiling'], np.nan)
    df['price_to_listing_floor_ratio'] = np.where(df['listing_floor'].fillna(0) > epsilon, df['match_match_price'] / df['listing_floor'], np.nan)
    df['price_to_listing_ref_ratio'] = np.where(df['listing_ref_price'].fillna(0) > epsilon, df['match_match_price'] / df['listing_ref_price'], np.nan)

    # 5. Foreign Trade Imbalances
    logging.info("Calculating foreign trade imbalances...")
    sum_foreign_volume = df['match_foreign_buy_volume'].fillna(0) + df['match_foreign_sell_volume'].fillna(0)
    df['foreign_volume_imbalance'] = np.where(
        sum_foreign_volume > epsilon,
        (df['match_foreign_buy_volume'].fillna(0) - df['match_foreign_sell_volume'].fillna(0)) / sum_foreign_volume,
        np.nan
    )
    sum_foreign_value = df['match_foreign_buy_value'].fillna(0) + df['match_foreign_sell_value'].fillna(0)
    df['foreign_value_imbalance'] = np.where(
        sum_foreign_value > epsilon,
        (df['match_foreign_buy_value'].fillna(0) - df['match_foreign_sell_value'].fillna(0)) / sum_foreign_value,
        np.nan
    )
    
    # 6. Accumulated Trade Features
    if 'match_accumulated_volume' in df.columns and 'match_accumulated_value' in df.columns:
        df['avg_trade_price_accumulated'] = np.where(
            df['match_accumulated_volume'].fillna(0) > epsilon,
            df['match_accumulated_value'] / df['match_accumulated_volume'],
            np.nan
        )
    else:
        df['avg_trade_price_accumulated'] = np.nan


    # 7. Trading Status and Date-Related Features
    logging.info("Calculating trading status and date-related features...")
    if 'listing_trading_status' in df.columns:
        # Example: 'H' for Halted. This needs to be specific to your data.
        # 1 if halted, 0 if not halted (and status is known), NaN otherwise
        df['is_trading_halted'] = df['listing_trading_status'].apply(
            lambda x: 1 if x == 'H' else (0 if pd.notna(x) else np.nan)
        )
    else:
        df['is_trading_halted'] = np.nan

    if 'listing_last_trading_date' in df.columns and 'listing_trading_date' in df.columns and \
       pd.api.types.is_datetime64_any_dtype(df['listing_last_trading_date']) and \
       pd.api.types.is_datetime64_any_dtype(df['listing_trading_date']):
        df['days_to_last_trading_date'] = (df['listing_last_trading_date'] - df['listing_trading_date']).dt.days
    else:
        df['days_to_last_trading_date'] = np.nan

    # 8. Time-Based Features (Extended)
    logging.info("Calculating time-based features...")
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time_hour'] = df['time'].dt.hour
        df['time_minute'] = df['time'].dt.minute
        df['time_second'] = df['time'].dt.second
        df['day_of_week'] = df['time'].dt.dayofweek  # Monday=0, Sunday=6
        df['time_since_market_open'] = (df['time'] - df['time'].dt.normalize().replace(hour=9, minute=30)).dt.total_seconds() # Example: 9:30 AM open
    else:
         logging.warning("'time' column not available or not datetime type for extended time-based features.")
         df['time_hour'] = np.nan
         df['time_minute'] = np.nan
         df['time_second'] = np.nan
         df['day_of_week'] = np.nan
         df['time_since_market_open'] = np.nan
    
    end_time = datetime.now()
    logging.info(f"Feature engineering core calculations complete. Duration: {end_time - start_time}")

    # --- Select and Finalize Features ---
    feature_columns = [
        'time', # Keep timestamp for joining/analysis
        # L1 features
        'mid_price_l1', 'microprice_l1', 'spread_l1', 'relative_spread_l1',
        'total_bid_volume_l1', 'total_ask_volume_l1',
        'market_depth_value_bid_l1', 'market_depth_value_ask_l1',
        # L2 features
        'mid_price_l2', 'spread_l2', 'relative_spread_l2',
        # L3 features
        'mid_price_l3', 'spread_l3', 'relative_spread_l3',
        # 3-level aggregate features
        'total_bid_volume_3lv', 'total_ask_volume_3lv',
        'market_depth_value_bid_3lv', 'market_depth_value_ask_3lv',
        'weighted_avg_price_3lv_bid', 'weighted_avg_price_3lv_ask',
        'volume_imbalance_3lv', 'value_imbalance_3lv',
        # Price-to-listing ratios
        'price_to_listing_ceiling_ratio', 'price_to_listing_floor_ratio', 'price_to_listing_ref_ratio',
        # Foreign trade
        'foreign_volume_imbalance', 'foreign_value_imbalance',
        # Accumulated trade
        'avg_trade_price_accumulated',
        # Status and date-based
        'is_trading_halted', 'days_to_last_trading_date',
        # Time-based
        'time_hour', 'time_minute', 'time_second', 'day_of_week', 'time_since_market_open'
    ]

    # Filter to only include columns that were actually created and are in the feature_columns list
    final_feature_cols_present = [col for col in feature_columns if col in df.columns]
    final_features_df = df[final_feature_cols_present].copy()

    # Handle potential infinities resulting from calculations
    final_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    logging.info(f"Selected {len(final_features_df.columns)} features. Final feature DataFrame shape: {final_features_df.shape}")
    logging.info(f"Feature engineering complete. Total duration: {datetime.now() - start_time}")
    
    return final_features_df


# --- Main Function for DAG Task ---
def engineering_features(golden_database, golden_table_name, feature_store_database, feature_store_table_name):
    """
    Main feature engineering task called by Airflow.
    Reads from golden DB, creates features, saves to feature store DB.
    """
    logging.info("Starting feature engineering task...")
    logging.info(f"Reading data from Golden DB: {golden_database}, Table: {golden_table_name}")

    db_config = {
        'dbname': golden_database,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'host': 'localhost',
        'port': 5432
    }

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Fetch existing 'time' values
    cur.execute(f"SELECT * FROM {golden_table_name}")
    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    features_df = create_features(df)

    conn.close()

    print('Creating table')
    _create_table_if_not_exists(feature_store_database, features_df, feature_store_table_name)
    print('Inserting Data')
    inserted_count = insertIntoTable(feature_store_database, features_df, feature_store_table_name)
    logging.info(f"Ingestion process completed. Approximately {inserted_count} new rows inserted.")