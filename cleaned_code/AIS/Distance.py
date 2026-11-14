
from pathlib import Path
import numpy as np
import pandas as pd

# 修改为你实际的文件路径
AIS_FILE = Path(
    r"C:\Users\user\Desktop\cleaned_code\AIS\data\matched\AIS_matched.csv"
)

# 根据实际列名修改
TIME_COL = "timestamp"
LAT_COL = "lat"
LON_COL = "lon"

df = pd.read_csv(AIS_FILE)

df = df[df["imo"].astype(str) == "1016252"].copy()
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")

df = df.dropna(subset=[TIME_COL, LAT_COL, LON_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)

if len(df) < 2:
    raise ValueError("有效AIS点不足2个")

lat1 = np.radians(df[LAT_COL].shift())
lat2 = np.radians(df[LAT_COL])
dlon = np.radians(df[LON_COL] - df[LON_COL].shift())
dlat = lat2 - lat1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

a = np.clip(a, 0, 1)

# 地球半径为3440.065海里
df["segment_nm"] = (
    2 * 3440.065 * np.arcsin(np.sqrt(a))
).fillna(0)

distance_nm = df["segment_nm"].sum()

print("IMO:", 1016252)
print("AIS点数:", len(df))
print("累计航程(NM):", round(distance_nm, 3))
print("起点:", df.iloc[0][[LAT_COL, LON_COL]].to_dict())
print("终点:", df.iloc[-1][[LAT_COL, LON_COL]].to_dict())
