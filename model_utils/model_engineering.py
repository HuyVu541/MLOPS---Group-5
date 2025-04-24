import os
import mlflow
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import mlflow.sklearn
import joblib
import mlflow.keras

def load_data(query: str, db_uri: str):
    engine = create_engine(db_uri)
    conn = engine.raw_connection()
    df = pd.read_sql(query, con=conn)
    conn.close()
    return df

def preprocess_data(df):
    df_numerical = df.select_dtypes(include = 'number')

    df_numerical = df_numerical.dropna(axis=1)
    df = df_numerical
    df['match_match_price'] = df['match_match_price'].replace(0, np.nan)
    df.fillna(method='ffill', inplace=True)
    return df  

def create_sequences(X, y, ts):
    Xs, ys = [], []
    for i in range(ts, len(X)-ts):
        Xs.append(X[i-ts:i])
        ys.append(y[i:i+ts])
    return np.array(Xs), np.array(ys)

def create_lstm_model(input_shape, num_layers=1, units=256, dropout_rate=0.2, time_steps=30):
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers based on num_layers
    for _ in range(num_layers - 1):  # subtract 1 because the first layer is already added
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    # Final LSTM layer (without return_sequences)
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    
    # Dense layer
    model.add(Dense(time_steps))
    
    # Compile model
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    
    return model

def train_model(model_name, query, db_uri, num_layers = 1):

    df = load_data(query, db_uri)
    df = preprocess_data(df)
    TARGET = ['match_match_price']

    # Feature set
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

    # 2. Tạo experiment và start run
    experiment_name = "LSTM"
    try:
        mlflow.create_experiment(experiment_name)  
    except mlflow.exceptions.MlflowException:
        pass  
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        n = len(df)
        n_train = int(0.6 * n)
        n_val   = int(0.2 * n)

        train_df = df.iloc[:n_train]
        val_df   = df.iloc[n_train:n_train + n_val]
        test_df  = df.iloc[n_train + n_val:]
        feat_scaler = StandardScaler()
        tgt_scaler  = StandardScaler()

        X_train_s = feat_scaler.fit_transform(train_df[FEATURES])
        X_val_s   = feat_scaler.transform(val_df[FEATURES])
        X_test_s  = feat_scaler.transform(test_df[FEATURES])

        y_train_s = tgt_scaler.fit_transform(train_df[TARGET]).flatten()
        y_val_s   = tgt_scaler.transform(val_df[TARGET]).flatten()
        y_test_s  = tgt_scaler.transform(test_df[TARGET]).flatten()

        TIME_STEPS = 30
        print("TIME_STEPS =", TIME_STEPS)

        X_train, y_train = create_sequences(X_train_s, y_train_s, TIME_STEPS)
        X_test, y_test = create_sequences(X_test_s, y_test_s, TIME_STEPS)

        # 3. Train
        n_features = X_train.shape[2]
        model = create_lstm_model(input_shape=(TIME_STEPS, n_features), num_layers= num_layers)
        model.fit(X_train, y_train, epochs=5, batch_size=256, verbose=2)

        y_pred = model.predict(X_test)
        print(f"y_pred shape: ", y_pred.shape)
        print(f"y_test shape: ", y_test.shape)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
    	
        print('MAE: ', mae)
        print('MSE: ', mape)
        print('MAPE: ', mape)

        metrics = {'mae': mae, 'mape': mape, 'mse': mse}
        mlflow.log_metrics(metrics)

        # 4. Log params/metrics tuỳ thích
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", 5)
        # mlflow.log_param("lr", 2e-5)

        mlflow.keras.log_model(model, model_name)

        # Save locally
        joblib.dump(feat_scaler, "feat_scaler.pkl")
        joblib.dump(tgt_scaler, "tgt_scaler.pkl")

        # Log to MLflow
        mlflow.sklearn.log_model(feat_scaler, "feat_scaler")
        mlflow.sklearn.log_model(tgt_scaler, "tgt_scaler")


    # 7. Trả về run_id & các URI nếu cần
    run_id = run.info.run_id
    # pytorch_uri = f"runs:/{run_id}/{model_name}_pytorch"
    model_uri  = f"runs:/{run_id}/{model_name}"
    return {"run_id": run_id, 'model_uri': model_uri}
