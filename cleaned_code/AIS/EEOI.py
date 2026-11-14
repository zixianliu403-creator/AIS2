"""EEOI calculator: one AIS observation sequence = one test voyage per IMO.
Place this file beside data/ and run via IDE. Requires pandas, numpy.

Read inputs: data/matched/co2_by_ship.csv, data/matched/AIS_matched.csv
             data/matched/vessels.csv OR data/STSD.csv
             (either text vessel_type/dwt columns, or IMO numeric codes
              imo4shiptype/imo4shipsize as produced by STSD_match.py)
             data/TCU.csv, data/BAF.csv,
             reference/auxiliary_engine_power.csv (code -> name/DWT band)
Write:       data/matched/EEOI_by_ship.csv and EEOI_audit.csv

Important: summary EEOI is exploratory when AIS is not a complete identified voyage.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ===== Edit only this section if necessary =====
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
REFERENCE = ROOT / 'reference'
TARGET_IMO = None  # None=all ships; 1016252=just this ship
BAF_MODE = 'none'  # 'none' (recommended until CO2 coverage audited) or 'table_multiplier'
MAX_SEGMENT_SPEED_KN = 60  # flag/remove obviously implausible AIS jumps
# ===============================================

SHIP_ALIASES = {
    'imo': ['imo', 'imo_number', 'imonumber', 'imo no', 'imo no.', 'IMO Number'],
    'vessel_type': ['vessel_type', 'ship_type', 'type', 'vesseltype', 'shiptype', 'ship type', 'vessel type'],
    'dwt': ['dwt', 'deadweight', 'deadweight_tonnage', 'deadweighttonnage', 'dead weight', 'dead weight tonnage'],
}
AIS_ALIASES = {
    'imo': SHIP_ALIASES['imo'],
    'time': ['timestamp', 'time', 'datetime', 'base_date_time', 'basedatetime', 'date_time', 'date time', 'utc', 'datetime_utc', 'received_time', 'timestamp_utc'],
    'lat': ['lat', 'latitude', 'y'],
    'lon': ['lon', 'longitude', 'lng', 'long', 'x'],
}
CO2_ALIASES = {
    'imo': SHIP_ALIASES['imo'],
    'co2_total_tonnes': ['co2_total_tonnes', 'co2_total_tons', 'co2_tonnes', 'co2_tons', 'total_co2_tonnes', 'co2_total_t', 'co2_ton', 'co2_tonne'],
    'co2_total_g': ['co2_total_g', 'total_co2_g', 'co2_g'],
}


def norm(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())


def pick_col(frame, aliases, required=True):
    name_map = {norm(c): c for c in frame.columns}
    for alias in aliases:
        if norm(alias) in name_map:
            return name_map[norm(alias)]
    if required:
        raise ValueError(f'Cannot find any of {aliases}; actual fields: {list(frame.columns)}')
    return None


def read(path):
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    for encoding in ['utf-8-sig', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            pass
    raise ValueError(f'Cannot decode {path}')


def clean_imo(series):
    return series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def parse_time(series):
    # Supports ISO strings and Unix timestamps in seconds/milliseconds.
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().mean() > 0.95:
        median = numeric.dropna().abs().median()
        unit = 'ms' if median > 1e11 else 's'
        return pd.to_datetime(numeric, unit=unit, errors='coerce', utc=True)
    return pd.to_datetime(series, errors='coerce', utc=True)


def classify_type(raw):
    value = norm(raw)
    rules = [
        ('chemical_tanker', ['chemicaltanker', 'chemicalcarrier']),
        ('liquid_gas_tanker', ['lng', 'lpg', 'liquefiedgas', 'liquidgas', 'gastanker', 'gascarrier']),
        ('oil_tanker', ['oiltanker', 'crudeoiltanker', 'producttanker', 'petroleumtanker', 'tanker']),
        ('bulk_carrier', ['bulkcarrier', 'bulker', 'orecarrier']),
        ('container', ['containership', 'container', 'boxship']),
        ('general_cargo', ['generalcargo', 'generalfreight', 'multipurposecargo', 'multipurposevessel']),
        ('reefer', ['reefer', 'refrigerated']),
        ('ro_ro', ['roro', 'rollonrolloff', 'vehiclecarrier', 'carcarrier']),
        ('miscellaneous', ['miscellaneous', 'othercargo']),
    ]
    for name, terms in rules:
        if any(term in value for term in terms):
            return name
    return None

def load_lookup(path):
    table = read(path)
    # Supported file format: dwt_min,dwt_max + 9 named vessel-type columns.
    lo = pick_col(table, ['dwt_min', 'min_dwt', 'dwt_from'])
    hi = pick_col(table, ['dwt_max', 'max_dwt', 'dwt_to'])
    table['_low'] = pd.to_numeric(table[lo], errors='coerce')
    table['_high'] = pd.to_numeric(table[hi], errors='coerce')
    return table


def lookup(table, vessel_type, dwt):
    if not vessel_type or not np.isfinite(dwt) or dwt <= 0:
        return np.nan
    col = pick_col(table, [vessel_type], required=False)
    if col is None:
        raise ValueError(f'Missing vessel type column {vessel_type} in parameter file')
    matches = table.loc[table['_low'].le(dwt) & (table['_high'].ge(dwt) | table['_high'].isna()), col]
    return float(pd.to_numeric(matches, errors='coerce').iloc[0]) if len(matches) else np.nan


def size_midpoint(text):
    """'10,000-34,999' -> 22499.5; '200,000+' -> 200000 (like upstream scripts); NaN if empty."""
    if text is None or pd.isna(text):
        return np.nan
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', str(text).replace(',', ''))]
    if not nums:
        return np.nan
    return nums[0] if len(nums) == 1 else (nums[0] + nums[1]) / 2


def load_imo4_map(path):
    """From reference table: code -> ship type name, and (type,size) -> DWT band text."""
    ref = read(path)
    tc = pick_col(ref, ['imo4shiptype'])
    sc = pick_col(ref, ['imo4shipsize'])
    tn = pick_col(ref, ['shiptype', 'vessel_type'])
    sn = pick_col(ref, ['shipsize'])
    ref = ref[[tc, sc, tn, sn]].copy()
    ref.columns = ['type_code', 'size_code', 'shiptype', 'shipsize']
    ref['type_code'] = pd.to_numeric(ref['type_code'], errors='coerce')
    ref['size_code'] = pd.to_numeric(ref['size_code'], errors='coerce')
    names = {int(t): n for t, n in ref.dropna(subset=['type_code']).groupby('type_code')['shiptype'].first().items()}
    bands = {(int(t), int(s)): b for t, s, b in zip(ref['type_code'], ref['size_code'], ref['shipsize'])
             if pd.notna(t) and pd.notna(s)}
    return names, bands


def distance_by_ship(ais):
    imo_col = pick_col(ais, AIS_ALIASES['imo'])
    tm_col = pick_col(ais, AIS_ALIASES['time'])
    lat_col = pick_col(ais, AIS_ALIASES['lat'])
    lon_col = pick_col(ais, AIS_ALIASES['lon'])
    ais = ais[[imo_col, tm_col, lat_col, lon_col]].copy()
    ais.columns = ['imo', 'time', 'lat', 'lon']
    ais['imo'] = clean_imo(ais['imo'])
    ais['time'] = parse_time(ais['time'])
    for col in ['lat', 'lon']:
        ais[col] = pd.to_numeric(ais[col], errors='coerce')
    ais = ais.dropna(subset=['imo', 'time', 'lat', 'lon'])
    ais = ais[ais['lat'].between(-90, 90) & ais['lon'].between(-180, 180)]
    ais = ais.sort_values(['imo', 'time']).drop_duplicates(['imo', 'time', 'lat', 'lon'])
    if TARGET_IMO is not None:
        ais = ais[ais['imo'] == str(TARGET_IMO)]
    out = []
    for imo, group in ais.groupby('imo', sort=False):
        g = group.sort_values('time')
        lat1 = np.radians(g['lat'].shift().to_numpy(dtype=float))
        lat2 = np.radians(g['lat'].to_numpy(dtype=float))
        dlat = lat2 - lat1
        dlon = np.radians((g['lon'] - g['lon'].shift()).to_numpy(dtype=float))
        aa = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        segments = 2 * 3440.065 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
        hours = g['time'].diff().dt.total_seconds().to_numpy(dtype=float) / 3600
        invalid = (~np.isfinite(hours)) | (hours <= 0)
        speed = np.divide(segments, hours, out=np.full(len(g), np.nan), where=~invalid)
        jumps = np.isfinite(speed) & (speed > MAX_SEGMENT_SPEED_KN)
        segments[invalid | jumps] = np.nan
        out.append({'imo': imo, 'distance_nm': float(np.nansum(segments)),
                    'ais_points': len(g), 'excluded_jumps': int(jumps.sum()),
                    'first_time': str(g['time'].iloc[0]), 'last_time': str(g['time'].iloc[-1])})
    return pd.DataFrame(out, columns=['imo', 'distance_nm', 'ais_points', 'excluded_jumps', 'first_time', 'last_time'])


def main():
    if BAF_MODE not in ['none', 'table_multiplier']:
        raise ValueError("BAF_MODE should be 'none' or 'table_multiplier'")
    co2 = read(DATA / 'matched' / 'co2_by_ship.csv')
    ais = read(DATA / 'matched' / 'AIS_matched.csv')
    # Prefer the pre-matched technical vessel table.
    specs_path = next((p for p in [DATA / 'matched' / 'vessels.csv', DATA / 'STSD.csv'] if p.exists()), None)
    if specs_path is None:
        raise FileNotFoundError('Neither matched/vessels.csv nor STSD.csv exists')
    specs = read(specs_path)
    tcu = load_lookup(DATA / 'TCU.csv')
    # BAF.csv is only needed in table_multiplier mode (file may be empty otherwise).
    baf = load_lookup(DATA / 'BAF.csv') if BAF_MODE == 'table_multiplier' else None

    ci = pick_col(co2, CO2_ALIASES['imo'])
    cton = pick_col(co2, CO2_ALIASES['co2_total_tonnes'], required=False)
    cg = pick_col(co2, CO2_ALIASES['co2_total_g'], required=False)
    if cton is None and cg is None:
        raise ValueError('co2_by_ship.csv must contain co2_total_tonnes or co2_total_g')
    emis = pd.DataFrame({'imo': clean_imo(co2[ci]),
                         'co2_tonnes': pd.to_numeric(co2[cton], errors='coerce') if cton else pd.to_numeric(co2[cg], errors='coerce') / 1e6})
    emis = emis.groupby('imo', as_index=False)['co2_tonnes'].sum(min_count=1)

    si = pick_col(specs, SHIP_ALIASES['imo'])
    st = pick_col(specs, SHIP_ALIASES['vessel_type'], required=False)
    sd = pick_col(specs, SHIP_ALIASES['dwt'], required=False)
    sty = pick_col(specs, ['imo4shiptype'], required=False)
    ssz = pick_col(specs, ['imo4shipsize'], required=False)
    if st is None and sty is None:
        raise ValueError('vessels.csv/STSD.csv needs vessel_type or imo4shiptype')
    ship = pd.DataFrame({'imo': clean_imo(specs[si])})
    if st is not None:
        ship['vessel_type_original'] = specs[st].astype('string')
    else:
        # STSD_match.py format: numeric IMO codes; resolve via reference table.
        names, bands = load_imo4_map(REFERENCE / 'auxiliary_engine_power.csv')
        tcode = pd.to_numeric(specs[sty], errors='coerce')
        ship['vessel_type_original'] = tcode.map(lambda v: names.get(int(v)) if pd.notna(v) else None)
    if sd is not None:
        ship['dwt'] = pd.to_numeric(specs[sd], errors='coerce')
        ship['dwt_source'] = 'table'
    elif st is None and ssz is not None:
        tcode = pd.to_numeric(specs[sty], errors='coerce')
        scode = pd.to_numeric(specs[ssz], errors='coerce')
        ship['dwt'] = [size_midpoint(bands.get((int(t), int(s))))
                       if pd.notna(t) and pd.notna(s) else np.nan
                       for t, s in zip(tcode, scode)]
        ship['dwt_source'] = 'imo4_size_band_midpoint'  # approximate, not measured DWT
    else:
        raise ValueError('vessels.csv/STSD.csv needs dwt, or imo4shiptype+imo4shipsize codes')
    ship = ship.drop_duplicates('imo')
    route = distance_by_ship(ais)
    merged = emis.merge(ship, on='imo', how='left').merge(route, on='imo', how='left')
    if TARGET_IMO is not None:
        merged = merged[merged['imo'] == str(TARGET_IMO)].copy()
    merged['vessel_type'] = merged['vessel_type_original'].map(classify_type)
    merged['tcu'] = [lookup(tcu, typ, dw) for typ, dw in zip(merged['vessel_type'], merged['dwt'])]
    merged['baf_table_value'] = ([lookup(baf, typ, dw) for typ, dw in zip(merged['vessel_type'], merged['dwt'])]
                                 if baf is not None else np.nan)

    # Table 6 entries > 1 are treated as multipliers ONLY in explicitly chosen mode.
    merged['baf_multiplier_used'] = 1.0 if BAF_MODE == 'none' else merged['baf_table_value']
    denominator = merged['dwt'] * merged['tcu'] * merged['distance_nm']
    merged['eeoi_g_per_t_nm'] = (merged['co2_tonnes'] * merged['baf_multiplier_used'] * 1e6 / denominator).where(denominator > 0)
    merged['status'] = 'ok'
    merged.loc[merged['vessel_type'].isna(), 'status'] = 'unrecognized_vessel_type'
    merged.loc[merged['dwt'].isna() | (merged['dwt'] <= 0), 'status'] = 'missing_dwt'
    merged.loc[merged['tcu'].isna(), 'status'] = 'missing_tcu'
    merged.loc[merged['distance_nm'].isna() | (merged['distance_nm'] <= 0), 'status'] = 'missing_ais_distance'
    if BAF_MODE == 'table_multiplier':
        merged.loc[merged['baf_table_value'].isna(), 'status'] = 'missing_baf'
    merged.loc[merged['co2_tonnes'].isna() | (merged['co2_tonnes'] < 0), 'status'] = 'invalid_co2'
    merged.loc[merged['status'] != 'ok', 'eeoi_g_per_t_nm'] = np.nan

    dest = DATA / 'matched'
    merged.to_csv(dest / 'EEOI_by_ship.csv', index=False, encoding='utf-8-sig')
    merged[['imo', 'status', 'vessel_type_original', 'vessel_type', 'dwt', 'dwt_source', 'ais_points',
            'excluded_jumps']].to_csv(
        dest / 'EEOI_audit.csv', index=False, encoding='utf-8-sig')
    print(f'Completed {len(merged)} vessel(s). Successful: {(merged.status == "ok").sum()}')
    print('BAF mode:', BAF_MODE, '(do not double-count port emissions)')
    print(merged[['imo', 'co2_tonnes', 'distance_nm', 'dwt', 'vessel_type', 'tcu', 'baf_table_value', 'eeoi_g_per_t_nm', 'status']].to_string(index=False))
    print('Saved to:', dest / 'EEOI_by_ship.csv')
    print('Audit:', dest / 'EEOI_audit.csv')


if __name__ == '__main__':
    main()
