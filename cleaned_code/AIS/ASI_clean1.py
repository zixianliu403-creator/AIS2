"""将截图格式的 AIS CSV 清洗成 calculate_ais_co2_only.py 可读取的活动数据。

用法: python clean_ais_for_co2.py --input AIS_raw.csv --output AIS_cleaned.csv
注意: 没有 IMO 的 AIS 记录无法与基于 IMO 的船舶参数表匹配，程序不会捏造 IMO。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--input',
    default='./data/aisdk-2025-02-27.csv',
    help='原始 AIS CSV 文件路径'
)
    parser.add_argument('--output', default='./data/ais_cleaned.csv', help='./data/ais_cleaned.csv')
    parser.add_argument('--max-gap-seconds', type=int, default=900)
    args = parser.parse_args()

    # dtype=str 既保留 MMSI/IMO 原始形式，也避免类似 1234567.0 的转型问题。
    raw = pd.read_csv(args.input, dtype='string', low_memory=False, encoding='utf-8-sig')
    raw.columns = raw.columns.str.strip().str.lstrip('#').str.strip()
    required = ['Timestamp', 'MMSI', 'IMO', 'Latitude', 'Longitude', 'SOG']
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'原始 AIS 缺少字段: {missing}\n实际字段: {list(raw.columns)}')

    print(f'原始记录: {len(raw):,}')
    # 保留 MMSI 只供核查；原排放脚本真正使用的是 IMO。
    df = raw[required].rename(columns={
        'Timestamp': 'timestamp', 'MMSI': 'mmsi', 'IMO': 'imo',
        'Latitude': 'lat', 'Longitude': 'lon', 'SOG': 'v',
    }).copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, utc=True, errors='coerce')
    df['imo'] = (df['imo'].str.strip().str.upper()
                 .str.replace(r'^IMO\s*', '', regex=True)
                 .str.replace(r'\.0$', '', regex=True))
    for c in ['lat', 'lon', 'v']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 只筛选可以用于现有脚本的七位 IMO；Unknown/空白不做假匹配。
    has_imo = df['imo'].str.fullmatch(r'\d{7}', na=False)
    print(f'有七位 IMO 的记录: {int(has_imo.sum()):,}; 无有效 IMO: {int((~has_imo).sum()):,}')
    valid = (has_imo & df['timestamp'].notna()
             & df['lat'].between(-90, 90) & df['lon'].between(-180, 180)
             & df['v'].between(0, 102.2))
    df = df.loc[valid].copy()
    print(f'基础字段有效: {len(df):,}')

    # 同船同秒仅保留一条；当前脚本假定每艘船具有唯一时间线。
    df = df.sort_values(['imo', 'timestamp']).drop_duplicates(
        subset=['imo', 'timestamp'], keep='first')
    # delta_time = 当前 AIS 记录到同一船舶下一记录的秒数（不是前一条到当前）。
    next_timestamp = df.groupby('imo', sort=False)['timestamp'].shift(-1)
    df['delta_time'] = (next_timestamp - df['timestamp']).dt.total_seconds()
    df['valid_segment'] = (df['delta_time'].gt(0)
                           & df['delta_time'].le(args.max_gap_seconds))
    print(f'有效时间间隔 (0, {args.max_gap_seconds}] 秒: {int(df["valid_segment"].sum()):,}')

    # CO2 排放脚本无需 timestamp/lat/lon/mmsi 计算，但保留用于后续核查航次。
    columns = ['imo', 'mmsi', 'timestamp', 'lat', 'lon', 'v', 'delta_time', 'valid_segment']
    clean = df.loc[df['valid_segment'], columns].copy()
    clean['timestamp'] = clean['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False, encoding='utf-8-sig')
    print(f'已输出: {output.resolve()}')
    print(f'可用于核算的记录: {len(clean):,}; 船舶数: {clean["imo"].nunique():,}')
    if clean.empty:
        print('警告: 结果为空。请重点检查原始 IMO 列是否多为 Unknown，或是否缺少连续 AIS 时间点。')
        print('如只有 MMSI，需先获取可靠 MMSI→IMO 船舶身份匹配表。')


if __name__ == '__main__':
    main()
