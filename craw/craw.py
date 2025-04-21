import numpy as np
import os, json
from datetime import datetime
import pandas as pd
from vnstock import Vnstock
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from apscheduler.schedulers.background import BackgroundScheduler
import gradio as gr
import pytz

json = {} #Google auth key

# --------------------------
# 1. Setup Google Sheets access
# --------------------------
SHEET_KEY = os.environ["GOOGLE_SHEET_KEY"]

# If your Google credentials are stored as JSON (like below), this should work;
# otherwise, load from file or environment as needed.
google_creds = json  # replace with your credentials loading logic if necessary

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(google_creds, scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_KEY).worksheet('Sheet2')


# Insert a header row that only shows the fundamental fields you need.
stock = Vnstock().stock(symbol='VCI', source='VCI')
# Here we retrieve the price board for a list of symbols. In this case, for symbol 'FPT'
df = stock.trading.price_board(['FPT'])

header = ['Time'] + ['_'.join(col) for col in df.columns]
# header = ["Time", "Symbol", "Price", "Volume"]

# It’s best to clear your sheet or check if the header exists before inserting.
try:
    # Insert header at the top if the sheet is empty.
    sheet.insert_row(header, index=1)
except Exception as e:
    print("Could not insert header (possibly already exists):", e)

last_trading_date = None

def fetch_and_append():
    global last_trading_date

    stock = Vnstock().stock(symbol='VCI', source='VCI')
    df = stock.trading.price_board(['FPT'])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)

    # take the first row as a Series
    s = df.iloc[0]

    # cast any numpy scalar to native Python
    row_data = s.apply(
        lambda x: x.item() if isinstance(x, np.generic) else x
    ).tolist()

    # get Vietnam timestamp
    bar_ts_vn = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
    if bar_ts_vn == last_trading_date:
        return
    last_trading_date = bar_ts_vn
    readable_time = bar_ts_vn.strftime("%Y-%m-%d %H:%M:%S")

    # prepend timestamp and append
    row = [readable_time] + row_data
    try:
        sheet.append_row(row, value_input_option="USER_ENTERED")
        print("Appended row:", row)
    except Exception as e:
        print("Error appending row:", e)

# --------------------------
# 3. Schedule the job
# --------------------------
# The frequency here can be adjusted. vnstock data may not update every second,
# so for instance, scheduling every 5 seconds or every minute might be more reasonable.
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_and_append, 'interval', seconds=1)
scheduler.start()

# --------------------------
# 4. Minimal Gradio UI to keep the process alive and display status
# --------------------------
def status():
    return "✅ Fetching fundamental VNStock data every 1 seconds and appending to Google Sheets..."

demo = gr.Interface(fn=status, inputs=None, outputs="text", live=True)
demo.launch()