import argparse
import pandas as pd
import numpy as np
import mlflow.sklearn
import mlflow.keras
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sqlalchemy import create_engine
import os

def create_sequences(X, y, ts = 30):
    Xs, ys = [], []
    for i in range(ts, len(X)-ts):
        Xs.append(X[i-ts:i])
        ys.append(y[i:i+ts])
    return np.array(Xs), np.array(ys)

def get_data_from_postgres(query: str, db_uri: str):
    engine = create_engine(db_uri)
    conn = engine.raw_connection()
    df = pd.read_sql(query, con=conn)
    conn.close()
    return df


def evaluate_model(model_path: str, run_id: str, db_uri: str, query: str):
    """
    Validate the model using test data from PostgreSQL.

    Args:
        model_path (str): MLflow model URI (pyfunc), e.g. "runs:/<run_id>/distilbert_sentiment"
        db_uri (str): Database URI for PostgreSQL.
        query (str): SQL query to fetch test data.

    Returns:
        None
    """
    # 1. Load test data
    df = get_data_from_postgres(query, db_uri)
    FEATURES = [
        'listing_ceiling',
        'listing_floor',
        'listing_ref_price',
        'listing_listed_share',
        'listing_prior_close_price',

        'match_match_vol',
        'match_accumulated_volume',
        'match_accumulated_value',
        'match_avg_match_price',
        'match_highest',
        'match_lowest',

        'match_foreign_sell_volume',
        'match_foreign_buy_volume',
        'match_current_room',
        'match_total_room',

        'match_total_accumulated_value',
        'match_total_accumulated_volume',
        'match_reference_price' 
    ]
    X = df[FEATURES]
    y = df['match_match_price']

    # Load feature scaler
    feat_scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/feat_scaler")
    X = feat_scaler.inverse_transform(X)

    # Load target scaler
    tgt_scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/tgt_scaler")
    y = tgt_scaler.inverse_transform(y.values.reshape(-1,1))

    # 2. Load the pyfunc model
    model = model = mlflow.keras.load_model(f"runs:/{run_id}/LSTM")
        
    X, y = create_sequences(X, y)
    y = y.reshape(-1, 30)
    
    # 4. Derive hard predictions
    y_pred = model.predict(X)
    
    print('y_pred shape:', y_pred.shape)
    print('y shape:', y.shape)
    # 5. Compute metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    
    # 6. Compiling metrics
    metrics = {'mae': mae, 'mape': mape, 'mse': mse}
    
    # 7. Logging to MLFLOW
    mlflow.log_metrics(metrics)
    # 8. Print & save report
    print('MAE: ', mae)
    print('MSE: ', mape)
    print('MAPE: ', mape)

    report = {
        "mae": mae,
        "mse": mse,
        "mape": mape,
    }
    pd.DataFrame([report]).to_csv("validation_report.csv", index=False)
    print("Saved validation_report.csv")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Validate a pyfunc MLflow model using PostgreSQL data.")
#     parser.add_argument("--model-path", type=str, required=True,
#                         help="MLflow model URI, e.g. runs:/<run_id>/distilbert_sentiment")
#     parser.add_argument("--db-uri", type=str, required=True, help="PostgreSQL connection URI.")
#     parser.add_argument("--query", type=str, required=True, help="SQL query to fetch test data.")
#     args = parser.parse_args()
#     evaluate_model(args.model_path, args.db_uri, args.query)
