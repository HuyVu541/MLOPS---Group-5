import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# 1. AUTO-REFRESH
st_autorefresh(interval=30_000, key="ticker")

# 2. LOAD & PREP DATA
@st.cache_data(ttl=20)
def load_data(url):
    df = pd.read_csv(url, parse_dates=['Time (VN)'])
    df = df.sort_values('Time (VN)').reset_index(drop=True)
    # compute diff to find gaps >5m
    df['dt'] = df['Time (VN)'].diff()
    return df

csv_url = "https://docs.google.com/spreadsheets/d/1yjmPxKbNBRD6DACtkq4l_Xp9O7ldmWujypKE9NhC6Z0/gviz/tq?tqx=out:csv&sheet=Sheet1"
df = load_data(csv_url)

# 3. SET UP FIGURE (use index as x so spacing is uniform)
fig = go.Figure(
    data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="AAPL"
    )]
)

# 4. FIND GAPS AND ADD BLOCK + LABEL
gap_threshold = pd.Timedelta('5min')
gaps = df.index[df['dt'] > gap_threshold]

for i in gaps:
    x0, x1 = i, i+1
    # rectangle block spanning full height
    fig.add_shape(
        type="rect",
        xref="x", yref="paper",
        x0=x0, x1=x1,
        y0=0, y1=1,
        fillcolor="White",
        opacity=1,
        layer="below",
        line_width=0
    )

# # 5. TICK LABELS: show real times every N points
# tick_every = max(1, len(df)//10)
# fig.update_xaxes(
#     tickmode="array",
#     tickvals=df.index[::tick_every],
#     ticktext=df['Time (VN)'].dt.strftime('%H:%M')[::tick_every],
#     title="Time (VN)"
# )
# 5. TICK LABELS: show real date + time every N points
tick_every = max(1, len(df)//10)

# format as “YYYY‑MM‑DD\nHH:MM”
ticktext = df['Time (VN)'].dt.strftime('%d-%m<br>%H:%M')

fig.update_xaxes(
    tickmode="array",
    tickvals=df.index[::tick_every],
    ticktext=ticktext[::tick_every],
    title="Time (VN)"
)


# 6. LAYOUT
fig.update_layout(
    title="AAPL Live Candlestick",
    yaxis_title="Price (USD)",
    dragmode="pan",
    margin=dict(l=0, r=0, t=20, b=20),
    width=1000,
    height=500
)

# 7. RENDER
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Last updated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
