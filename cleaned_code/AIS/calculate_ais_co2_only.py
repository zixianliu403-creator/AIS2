"""根据清洗后的 AIS 活动数据计算船舶 CO2 排放。

本脚本只完成活动法排放核算，不识别航次，不计算 EEOI/BAF，也不匹配贸易。

区间排放公式（单位：g CO2）：
    main = main_power_kw * load_factor * delta_time_h * ef_main
    auxiliary = auxiliary_power_kw[phase] * delta_time_h * ef_auxiliary
    boiler = boiler_power_kw[phase] * delta_time_h * ef_boiler

    load_factor = clip((SOG / design_speed)^3, 0, 1)

输入 AIS 必须已经按 IMO、时间排序，并包含当前记录到下一记录的
``delta_time``（秒）。大时间间隔会被丢弃，而不是假设船舶持续保持当前状态。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PHASES = ["at berth", "anchored", "manoeuvring", "at sea"]


def normalize_imo(series: pd.Series) -> pd.Series:
    """将 IMO9529968、9529968.0 等统一为七位数字字符串。"""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^IMO\s*", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def size_bounds(value: object) -> tuple[float, float] | None:
    text = str(value).replace(",", "").replace(" ", "")
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    if "+" in text:
        return numbers[0], float("inf")
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    return numbers[0], numbers[0]


def prepare_vessels(vessels: pd.DataFrame, power_reference: pd.DataFrame) -> pd.DataFrame:
    """校验船舶技术参数，并把文本船型/尺度映射成 IMO4 代码。"""
    aliases = {
        "dwt": "dwt_tonnes",
        "powerkw": "main_power_kw",
        "design_speed": "design_speed_knots",
        "enginetype": "engine_type",
        "fueltype": "fuel_type",
    }
    vessels = vessels.rename(columns={k: v for k, v in aliases.items() if k in vessels})
    required = {
        "imo",
        "imo4shiptype",
        "imo4shipsize",
        "main_power_kw",
        "design_speed_knots",
        "engine_type",
        "fuel_type",
    }
    missing = sorted(required - set(vessels.columns))
    if missing:
        raise ValueError(f"船舶参数表缺少字段：{missing}")

    vessels = vessels.copy()
    vessels["imo"] = normalize_imo(vessels["imo"])
    vessels = vessels[vessels["imo"].str.fullmatch(r"\d{7}", na=False)].copy()
    if vessels["imo"].duplicated().any():
        duplicated = vessels.loc[vessels["imo"].duplicated(False), "imo"].unique()
        raise ValueError(f"船舶参数表 IMO 必须唯一，重复值示例：{duplicated[:5].tolist()}")

    reference = power_reference[
        ["imo4shiptype", "imo4shipsize", "shiptype", "shipsize"]
    ].drop_duplicates()
    reference["type_key"] = reference["shiptype"].map(normalize_label)
    reference["size_key"] = reference["shipsize"].map(normalize_label)

    type_codes: list[int] = []
    size_codes: list[int] = []
    for row_number, row in vessels.iterrows():
        raw_type = row["imo4shiptype"]
        raw_size = row["imo4shipsize"]
        numeric_type = pd.to_numeric(pd.Series([raw_type]), errors="coerce").iloc[0]
        numeric_size = pd.to_numeric(pd.Series([raw_size]), errors="coerce").iloc[0]
        if pd.notna(numeric_type) and pd.notna(numeric_size):
            type_codes.append(int(numeric_type))
            size_codes.append(int(numeric_size))
            continue

        candidates = reference[reference["type_key"] == normalize_label(raw_type)]
        if candidates.empty:
            raise ValueError(f"船舶参数第 {row_number + 2} 行船型无法映射：{raw_type!r}")
        exact = candidates[candidates["size_key"] == normalize_label(raw_size)]
        if not exact.empty:
            match = exact.iloc[0]
        else:
            requested = size_bounds(raw_size)
            if requested is None:
                raise ValueError(f"船舶参数第 {row_number + 2} 行尺度无法解析：{raw_size!r}")
            midpoint = requested[0] if np.isinf(requested[1]) else sum(requested) / 2
            matches = []
            for _, candidate in candidates.iterrows():
                bounds = size_bounds(candidate["shipsize"])
                if bounds and bounds[0] <= midpoint <= bounds[1]:
                    matches.append(candidate)
            if not matches:
                raise ValueError(
                    f"船舶参数第 {row_number + 2} 行尺度无法映射："
                    f"{raw_type!r}/{raw_size!r}"
                )
            match = matches[0]
        type_codes.append(int(match["imo4shiptype"]))
        size_codes.append(int(match["imo4shipsize"]))

    vessels["imo4shiptype"] = type_codes
    vessels["imo4shipsize"] = size_codes
    for column in ["main_power_kw", "design_speed_knots"]:
        vessels[column] = pd.to_numeric(vessels[column], errors="coerce")
    invalid_numeric = (
        vessels["main_power_kw"].isna()
        | vessels["main_power_kw"].le(0)
        | vessels["design_speed_knots"].isna()
        | vessels["design_speed_knots"].le(0)
    )
    if invalid_numeric.any():
        raise ValueError("main_power_kw 和 design_speed_knots 必须为正数")
    vessels["engine_type"] = vessels["engine_type"].str.strip().str.lower()
    vessels["fuel_type"] = vessels["fuel_type"].str.strip().str.lower()
    return vessels


def load_parameters(vessel_file: Path, reference_dir: Path) -> pd.DataFrame:
    factors = pd.read_csv(reference_dir / "emission_factor.csv")
    factors.columns = factors.columns.str.strip()
    factors["engine_type"] = factors["enginetype"].str.strip().str.lower()
    factors["fuel_type"] = factors["fueltype"].str.strip().str.lower()

    auxiliary = pd.read_csv(reference_dir / "auxiliary_engine_power.csv")
    boiler = pd.read_csv(reference_dir / "boiler_engine_power.csv")
    vessels = prepare_vessels(pd.read_csv(vessel_file, dtype={"imo": "string"}), auxiliary)

    parameters = vessels.merge(
        factors[["engine_type", "fuel_type", "ef_main", "ef_auxiliary", "ef_boiler"]],
        on=["engine_type", "fuel_type"],
        how="left",
        validate="many_to_one",
    )
    if parameters[["ef_main", "ef_auxiliary", "ef_boiler"]].isna().any(axis=None):
        missing = parameters.loc[
            parameters[["ef_main", "ef_auxiliary", "ef_boiler"]].isna().any(axis=1),
            ["engine_type", "fuel_type"],
        ].drop_duplicates()
        raise ValueError(f"排放因子表没有这些发动机/燃料组合：{missing.to_dict('records')}")

    auxiliary_long = auxiliary.melt(
        id_vars=["imo4shiptype", "imo4shipsize"],
        value_vars=PHASES,
        var_name="phase",
        value_name="auxiliary_power_kw",
    )
    boiler_long = boiler.melt(
        id_vars=["imo4shiptype", "imo4shipsize"],
        value_vars=PHASES,
        var_name="phase",
        value_name="boiler_power_kw",
    )
    phase_power = auxiliary_long.merge(
        boiler_long,
        on=["imo4shiptype", "imo4shipsize", "phase"],
        validate="one_to_one",
    )
    return parameters.merge(
        phase_power,
        on=["imo4shiptype", "imo4shipsize"],
        how="left",
        validate="one_to_many",
    )


def truthy(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def calculate_chunk(
    chunk: pd.DataFrame,
    parameters_by_phase: pd.DataFrame,
    max_gap_seconds: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    aliases = {"sog": "v", "SOG": "v"}
    chunk = chunk.rename(columns={k: v for k, v in aliases.items() if k in chunk})
    required = {"imo", "v", "delta_time"}
    missing = sorted(required - set(chunk.columns))
    if missing:
        raise ValueError(f"AIS 文件缺少字段：{missing}")

    input_rows = len(chunk)
    chunk = chunk.copy()
    chunk["imo"] = normalize_imo(chunk["imo"])
    chunk["v"] = pd.to_numeric(chunk["v"], errors="coerce")
    chunk["delta_time"] = pd.to_numeric(chunk["delta_time"], errors="coerce")
    valid = (
        chunk["imo"].str.fullmatch(r"\d{7}", na=False)
        & chunk["v"].between(0, 102.2, inclusive="both")
        & chunk["delta_time"].gt(0)
        & chunk["delta_time"].le(max_gap_seconds)
    )
    if "valid_segment" in chunk.columns:
        valid &= truthy(chunk["valid_segment"])
    chunk = chunk.loc[valid].copy()

    # 先和不分阶段的船舶参数连接，再根据该记录的运行阶段连接功率表。
    static_columns = [
        "imo", "main_power_kw", "design_speed_knots", "engine_type", "fuel_type",
        "ef_main", "ef_auxiliary", "ef_boiler", "imo4shiptype", "imo4shipsize",
    ]
    static = parameters_by_phase[static_columns].drop_duplicates("imo")
    data = chunk.merge(static, on="imo", how="inner", validate="many_to_one")
    matched_rows = len(data)

    data["load_factor"] = ((data["v"] / data["design_speed_knots"]) ** 3).clip(0, 1)
    data["phase"] = np.select(
        [data["v"] < 1, data["v"] < 3, data["load_factor"] < 0.2],
        ["at berth", "anchored", "manoeuvring"],
        default="at sea",
    )
    phase_power = parameters_by_phase[
        ["imo", "phase", "auxiliary_power_kw", "boiler_power_kw"]
    ]
    data = data.merge(phase_power, on=["imo", "phase"], how="left", validate="many_to_one")
    if data[["auxiliary_power_kw", "boiler_power_kw"]].isna().any(axis=None):
        raise ValueError("部分船型/尺度/阶段没有辅机或锅炉功率参数")

    hours = data["delta_time"] / 3600
    data["co2_main_g"] = (
        data["main_power_kw"] * data["load_factor"] * hours * data["ef_main"]
    )
    data["co2_auxiliary_g"] = data["auxiliary_power_kw"] * hours * data["ef_auxiliary"]
    data["co2_boiler_g"] = data["boiler_power_kw"] * hours * data["ef_boiler"]
    data["co2_total_g"] = data[["co2_main_g", "co2_auxiliary_g", "co2_boiler_g"]].sum(axis=1)

    audit = {
        "input_rows": input_rows,
        "valid_activity_rows": int(valid.sum()),
        "matched_vessel_rows": matched_rows,
        "unmatched_or_invalid_rows": input_rows - matched_rows,
    }
    return data, audit


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    matched = script_dir / "data" / "matched"
    parser = argparse.ArgumentParser(description=__doc__)
    # 默认指向 STSD_match.py 的输出；也可以点击运行后不加参数，
    # 或在命令行覆盖。
    parser.add_argument("--ais-file", type=Path, default=matched / "AIS_matched.csv")
    parser.add_argument("--vessel-file", type=Path, default=matched / "vessels.csv")
    parser.add_argument("--reference-dir", type=Path, default=script_dir / "reference")
    parser.add_argument("--output-prefix", type=Path, default=matched / "co2")
    parser.add_argument("--max-gap-seconds", type=float, default=900)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_gap_seconds <= 0 or args.chunksize <= 0:
        raise ValueError("max-gap-seconds 和 chunksize 必须为正数")
    parameters = load_parameters(args.vessel_file, args.reference_dir)
    output_rows = args.output_prefix.with_name(args.output_prefix.name + "_rows.csv")
    output_ships = args.output_prefix.with_name(args.output_prefix.name + "_by_ship.csv")
    output_audit = args.output_prefix.with_name(args.output_prefix.name + "_audit.json")
    output_rows.parent.mkdir(parents=True, exist_ok=True)

    if output_rows.exists():
        output_rows.unlink()
    aggregates: list[pd.DataFrame] = []
    audit_total = {
        "input_rows": 0,
        "valid_activity_rows": 0,
        "matched_vessel_rows": 0,
        "unmatched_or_invalid_rows": 0,
    }
    wrote_header = False
    for chunk in pd.read_csv(args.ais_file, chunksize=args.chunksize, dtype={"imo": "string"}):
        calculated, audit = calculate_chunk(chunk, parameters, args.max_gap_seconds)
        for key in audit_total:
            audit_total[key] += audit[key]
        if calculated.empty:
            continue
        calculated.to_csv(
            output_rows,
            mode="a",
            header=not wrote_header,
            index=False,
            encoding="utf-8-sig" if not wrote_header else "utf-8",
        )
        wrote_header = True
        aggregates.append(
            calculated.groupby("imo", as_index=False).agg(
                observation_intervals=("imo", "size"),
                activity_hours=("delta_time", lambda x: x.sum() / 3600),
                co2_main_g=("co2_main_g", "sum"),
                co2_auxiliary_g=("co2_auxiliary_g", "sum"),
                co2_boiler_g=("co2_boiler_g", "sum"),
                co2_total_g=("co2_total_g", "sum"),
            )
        )

    if not aggregates:
        raise ValueError("没有可核算的 AIS 记录：请检查 IMO、时间间隔和船舶参数匹配")
    by_ship = pd.concat(aggregates, ignore_index=True).groupby("imo", as_index=False).sum()
    for column in ["co2_main_g", "co2_auxiliary_g", "co2_boiler_g", "co2_total_g"]:
        by_ship[column.replace("_g", "_tonnes")] = by_ship[column] / 1_000_000
    by_ship.to_csv(output_ships, index=False, encoding="utf-8-sig")

    audit_total["matched_ships"] = int(by_ship["imo"].nunique())
    audit_total["total_co2_tonnes"] = float(by_ship["co2_total_tonnes"].sum())
    audit_total["method"] = {
        "main_engine_load": "clip((SOG/design_speed)^3, 0, 1)",
        "co2": "power_kW * load_factor * hours * emission_factor_g_per_kWh",
        "phase_rule": "SOG<1 berth; 1<=SOG<3 anchored; otherwise LF<0.2 manoeuvring; else at sea",
        "excluded_gap_seconds_above": args.max_gap_seconds,
    }
    output_audit.write_text(json.dumps(audit_total, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"逐记录结果：{output_rows.resolve()}")
    print(f"逐船汇总：{output_ships.resolve()}")
    print(f"质量审计：{output_audit.resolve()}")
    print(f"总 CO2：{audit_total['total_co2_tonnes']:.6f} tonnes")


if __name__ == "__main__":
    main()
