"""AIS_cleaned.csv 与 STSD 船舶参数表按 IMO 匹配。

【这个脚本做什么】
把两部分数据“接”起来：
  1. AIS_cleaned.csv —— 船舶的实时动态数据（位置、航速、时间戳等），
     由上游 ASI_clean1.py 清洗生成，行数非常大，因此全程分块读取；
  2. STSD.csv —— 船舶静态技术参数表（船型、船尺、主机功率、设计航速、
     发动机类型、燃料类型等），来自 Sea桅 Trade Seeds / STSD 数据库导出。
两部分数据通过唯一的船舶识别号 IMO（7 位数字）做主键关联。

【为什么需要这一步】
后续的碳排放核算（calculate_ais_co2_only.py）要求每条 AIS 记录都能查到
对应的完整技术参数。但 STSD 表中存在缺列、缺值、IMO 格式不统一、重复行
等脏数据，本脚本负责把这些情况全部甄别清楚：参数完整的船进入核算清单，
参数不完整的船单独列出、绝不猜测或编造缺失值，等人工补齐后重跑即可。

【输出四个文件，均在 data/matched/ 下】
  vessels.csv            参数完整、且在 AIS 中出现过的船舶清单（IMO 唯一），
                         是排放核算的“船舶档案”输入；
  vessels_incomplete.csv 在 AIS 中出现、但 STSD 参数不全的船舶，附带
                         missing_fields 列说明缺了哪些字段，便于人工补录；
  AIS_matched.csv        只保留“完整船舶”的 AIS 记录，规模远小于原始文件，
                         供排放核算直接读取；
  match_audit.json       匹配过程审计数字（各阶段行数/船数），用于核对
                         数据在流转过程中是否有丢失。

【匹配的基本流程】
  读取 STSD → 列名标准化（COLUMN_MAP 手工映射 + ALIASES 别名识别）
  → 逐行校验参数完整性 → 第一遍分块扫描 AIS 收集出现过的 IMO
  → 求交集并输出船舶清单 → 第二遍分块扫描 AIS 导出匹配记录 → 写审计 JSON。

示例：
python STSD_match.py
或直接点击 VS Code 的“运行 Python 文件”，路径在下方常量中修改。
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import pandas as pd

# ============= 只需修改这里的文件名/路径 =============
# 运行本脚本前，一般只需要确认下面几个路径是否正确：
#   AIS_FILE   —— 清洗后的 AIS 动态数据（ASI_clean1.py 的输出）
#   STSD_FILE  —— STSD 静态参数表，支持 .csv 或 .xlsx/.xls
#   OUTPUT_DIR —— 四个输出文件的存放目录，不存在会自动创建
#   CHUNK_SIZE —— 分块读取 AIS 时每块的行数。AIS 文件动辄数百万行，
#                 一次全部读入内存不现实，故按块流式处理；内存紧张可调小。
BASE_DIR = Path(__file__).resolve().parent
AIS_FILE = BASE_DIR / "data" / "ais_cleaned.csv"
STSD_FILE = BASE_DIR / "data" / "STSD.csv"
OUTPUT_DIR = BASE_DIR / "data" / "matched"
STSD_SHEET = "0"   # Excel 用：工作表序号或名称；CSV 无需修改
CHUNK_SIZE = 300_000
# ==================================================


# 如果 STSD 的表头与常见名称不同，在这里添加：'STSD实际列名': '目标字段名'
# COLUMN_MAP 是最高优先级的“手工指定”通道：别名识别不到、或识别错了字段时，
# 在这里写死映射即可，程序会原样采纳，不做任何猜测。
COLUMN_MAP = {
    # 'IMO Number': 'imo',
    # 'Installed Power (kW)': 'main_power_kw',
}

# ALIASES 是自动识别用的“别名词典”：把 STSD 表头去掉大小写、空格和
# 标点（见 key() 函数）后，若恰好命中其中某个写法，就重命名为标准字段名。
# 每个标准字段的含义：
#   imo                —— 7 位 IMO 编号，匹配主键
#   imo4shiptype       —— IMO 4 位船型代码（如化学品船、散货船）
#   imo4shipsize       —— IMO 4 位船尺分级代码
#   main_power_kw      —— 主机额定功率 MCR（kW），排放模型的核心输入
#   design_speed_knots —— 设计/服务航速（节），用于功率修正
#   engine_type        —— 主机类型（二冲程/四冲程柴油机等）
#   fuel_type          —— 主要燃料类型（HFO/MDO/LNG 等）
ALIASES = {
    'imo': ['imo', 'imonumber', 'imono', 'shipimo', 'vesselimo', 'imoid'],
    'imo4shiptype': ['imo4shiptype', 'imo4type', 'shiptypecode', 'shiptype', 'vesseltype'],
    'imo4shipsize': ['imo4shipsize', 'imo4size', 'shipsizecode', 'shipsize', 'sizeclass'],
    'main_power_kw': ['mainpowerkw', 'mainenginepowerkw', 'mainenginepower', 'powerkw', 'mepowerkw', 'installedpowerkw', 'mcrkw', 'mcr'],
    'design_speed_knots': ['designspeedknots', 'designspeed', 'servicespeedknots', 'servicespeed', 'speedknots'],
    'engine_type': ['enginetype', 'mainenginetype', 'machinerytype'],
    'fuel_type': ['fueltype', 'mainfueltype', 'fuel', 'primaryfuel'],
}
REQUIRED = ['imo', 'imo4shiptype', 'imo4shipsize', 'main_power_kw', 'design_speed_knots', 'engine_type', 'fuel_type']


def key(v):
    """把列名“压平”成小写字母数字串，用于与 ALIASES 词典比对。

    例如 'Installed Power (kW)' → 'installedpowerkw'，
    'IMO Number' → 'imonumber'。这样表头里的空格、括号、大小写差异
    都不再影响识别。
    """
    return re.sub(r'[^a-z0-9]', '', str(v).strip().lower())


def norm_imo(s):
    """把一列 IMO 编号规范化为统一的纯数字字符串。

    AIS 和 STSD 两个来源的写法可能不一致，例如 'IMO 1234567'、
    ' imo1234567 '、甚至 pandas 读数字列时带出的 '1234567.0'。
    统一去掉前后空格、'IMO' 前缀和小数尾巴并转大写，
    保证两边能按同一口径对得上。
    """
    return (s.astype('string').str.strip().str.upper()
            .str.replace(r'^IMO\s*', '', regex=True)
            .str.replace(r'\.0$', '', regex=True))


def load_stsd(path, sheet):
    """读取 STSD 参数表，按扩展名自动选择 CSV 或 Excel 读取方式。

    所有列一律按字符串读入（dtype='string'），避免 pandas 自动猜类型
    把 IMO 之类编号读成浮点数；数值列后续再显式转换。
    CSV 优先按 UTF-8（带 BOM 也可）解码，失败则回退到国内常见的
    GB18030 编码，兼容中文导出的表格。
    """
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path, sheet_name=sheet, dtype='string')
    try:
        return pd.read_csv(path, dtype='string', encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype='string', encoding='gb18030', low_memory=False)


def main():
    """串起整个匹配流程：读表 → 认列 → 校验 → 两遍扫描 AIS → 输出。

    各步骤详细说明见函数体内以 “----------” 分隔的阶段注释。
    """
    # 不使用 argparse，直接点击 VS Code 的“运行 Python 文件”即可
    # （这里手动拼一个 Namespace，是为了让后面的代码统一用 args.xxx 取参数，
    #   改路径只需动文件顶部的常量，不必碰命令行。）
    args = argparse.Namespace(
        ais=AIS_FILE, stsd=STSD_FILE, sheet=STSD_SHEET,
        outdir=OUTPUT_DIR, chunksize=CHUNK_SIZE,
    )
    print("AIS 文件：", args.ais)
    print("STSD 文件：", args.stsd)
    print("输出目录：", args.outdir)
    if not args.ais.is_file():
        raise FileNotFoundError(f'AIS 文件不存在: {args.ais}')
    if not args.stsd.is_file():
        raise FileNotFoundError(f'STSD 文件不存在: {args.stsd}')
    args.outdir.mkdir(parents=True, exist_ok=True)
    sheet = int(args.sheet) if args.sheet.isdecimal() else args.sheet
    stsd = load_stsd(args.stsd, sheet)
    stsd.columns = [str(c).strip() for c in stsd.columns]
    print('STSD 原始列名:', list(stsd.columns))

    # ---------- 阶段 1：识别 STSD 列名 ----------
    # 先应用手工映射（COLUMN_MAP，最高优先级），再匹配通用别名（ALIASES）。
    # 别名识别采取“唯一命中才采纳”的保守策略：某个标准字段若在表头中
    # 恰好只对应上一列，才重命名；命中 0 列或多列都视为识别失败，
    # 绝不猜测模糊含义的字段，把不确定性交还给使用者用 COLUMN_MAP 解决。
    stsd = stsd.rename(columns=COLUMN_MAP)
    renames = {}
    occupied = set(stsd.columns)
    for target, possibilities in ALIASES.items():
        if target in occupied:
            continue
        matches = [col for col in stsd.columns if key(col) in possibilities]
        if len(matches) == 1:
            renames[matches[0]] = target
            occupied.add(target)
    stsd = stsd.rename(columns=renames)
    if 'imo' not in stsd:
        raise ValueError('STSD 没有识别出 IMO 列。请在代码顶部 COLUMN_MAP 指明，例如 {"IMO Number": "imo"}')
    absent = [f for f in REQUIRED if f not in stsd]
    if absent:
        raise ValueError(
            f'STSD 缺少字段 {absent}。如列名不同请在 COLUMN_MAP 中补充映射，'
            '程序不会猜测或编造缺失的技术参数。')

    # ---------- 阶段 2：校验 STSD 各行，分成完整/不完整 ----------
    # 判定标准逐字段给出（checks 字典），全部通过才算“完整”：
    #   imo                —— 规范化后必须是恰好 7 位纯数字；
    #   功率 / 设计航速     —— 必须能转成数字且大于 0（空值、'N/A' 之类
    #                          转不成数字的按缺失处理）；
    #   其余文本字段        —— 去掉空格后非空即可。
    # complete_mask 标记可参与排放核算的行；同时为每行生成 missing_fields，
    # 把缺失的字段名逗号拼接，方便后续人工照着补录。
    stsd['imo'] = norm_imo(stsd['imo'])
    valid_imo = stsd['imo'].str.fullmatch(r'\d{7}', na=False)
    power = pd.to_numeric(stsd['main_power_kw'], errors='coerce')
    speed = pd.to_numeric(stsd['design_speed_knots'], errors='coerce')
    checks = {
        'imo': valid_imo,
        'imo4shiptype': stsd['imo4shiptype'].fillna('').str.strip().ne(''),
        'imo4shipsize': stsd['imo4shipsize'].fillna('').str.strip().ne(''),
        'main_power_kw': power.gt(0),
        'design_speed_knots': speed.gt(0),
        'engine_type': stsd['engine_type'].fillna('').str.strip().ne(''),
        'fuel_type': stsd['fuel_type'].fillna('').str.strip().ne(''),
    }
    complete_mask = pd.concat(checks.values(), axis=1).all(axis=1)
    fail = pd.DataFrame({name: ~ok.rename(name) for name, ok in checks.items()})
    stsd['missing_fields'] = fail.apply(
        lambda row: ','.join(col for col in fail.columns if row[col]), axis=1)

    # ---------- 阶段 3：第一遍分块扫描 AIS，收集出现过活动的 IMO ----------
    # AIS 文件太大无法一次读入，这里只做轻量统计：逐块提取去重后的 IMO，
    # 汇成一个集合 ais_imos，代表“考察期内真正在海上跑过的船”。
    # STSD 里参数再全，船没出现在 AIS 里也无从核算，故后面取交集。
    print('扫描 AIS 文件…')
    ais_imos: set[str] = set()
    ais_rows = 0
    first_chunk = pd.read_csv(args.ais, dtype='string', chunksize=args.chunksize,
                              low_memory=False, encoding='utf-8-sig')
    for i, chunk in enumerate(first_chunk, 1):
        if 'imo' not in chunk.columns:
            raise ValueError(f'AIS 文件缺少 imo 列，实际列: {list(chunk.columns)}\n'
                             '请先运行 ASI_clean1.py 生成清洗后的 ais_cleaned.csv')
        ais_rows += len(chunk)
        ais_imos.update(norm_imo(chunk['imo']).dropna().unique().tolist())
        if i % 20 == 0:
            print(f'  已扫描 {ais_rows:,} 行')
    print(f'AIS 记录 {ais_rows:,} 行，涉及船舶 {len(ais_imos):,} 艘')

    # 至此两侧数据已对齐：in_ais 标记该 STSD 行的船在 AIS 中出现过；
    # matched_imo 是两边 IMO 的交集，即“有动态记录、也有静态档案”的船。
    stsd['in_ais'] = valid_imo & stsd['imo'].isin(ais_imos)
    matched_imo = ais_imos & set(stsd.loc[valid_imo, 'imo'])
    print(f'STSD 与 AIS 按 IMO 匹配到 {len(matched_imo):,} 艘船')

    # ---------- 阶段 4：输出船舶清单 ----------
    # vessels.csv 收录“参数完整 ∩ 在 AIS 中出现”的船。下游排放核算要求
    # IMO 唯一，所以同一 IMO 若在 STSD 里重复出现（如换名、补录造成的重复行），
    # 只保留第一行（keep='first'）。dwt、lf 两列若 STSD 恰好提供则一并带上，
    # 没有也不强制——核心必填字段已在阶段 1 把关。
    out_cols = [c for c in REQUIRED + ['dwt', 'lf'] if c in stsd.columns]
    complete = (stsd.loc[complete_mask & stsd['in_ais'], out_cols]
                .drop_duplicates('imo', keep='first')
                .sort_values('imo'))
    vessels_path = args.outdir / 'vessels.csv'
    complete.to_csv(vessels_path, index=False, encoding='utf-8-sig')
    print(f'完整船舶清单（用于排放核算）: {vessels_path.resolve()}  '
          f'共 {len(complete):,} 艘')

    # 不完整清单同样只针对“在 AIS 中出现”的船——没活动的船即使参数缺失
    # 也不影响本期核算，不必列入待补录清单干扰视线。
    incomplete = stsd.loc[valid_imo & ~complete_mask & stsd['in_ais'],
                          out_cols + ['missing_fields']]
    incomplete_path = args.outdir / 'vessels_incomplete.csv'
    incomplete.to_csv(incomplete_path, index=False, encoding='utf-8-sig')
    print(f'参数不完整、暂不核算的船舶: {incomplete_path.resolve()}  '
          f'共 {incomplete["imo"].nunique():,} 艘')
    if complete.empty:
        print('警告: 没有可核算的完整船舶。请补充 STSD 技术参数后重跑。')

    # ---------- 阶段 5：第二次分块扫描 AIS，导出匹配记录 ----------
    # 第一遍只收集了 IMO 集合，没有保留原始记录；这里重扫一遍，
    # 把属于“完整船舶”（keep 集合）的 AIS 行逐块筛出、追加写入
    # AIS_matched.csv。写第一块时用 'w' 模式带表头，其后用 'a' 模式
    # 只追加数据，拼成一个完整的 CSV。若最终一行都没写出（比如没有
    # 任何完整船），也要生成空文件，避免下游因文件缺失而报错。
    keep = set(complete['imo'])
    ais_out_path = args.outdir / 'AIS_matched.csv'
    matched_rows = 0
    wrote_header = False
    if keep:
        for i, chunk in enumerate(pd.read_csv(
                args.ais, dtype='string', chunksize=args.chunksize,
                low_memory=False, encoding='utf-8-sig'), 1):
            sel = chunk.loc[chunk['imo'].isin(keep)]
            matched_rows += len(sel)
            if not sel.empty:
                sel.to_csv(ais_out_path, mode='w' if not wrote_header else 'a',
                           header=not wrote_header, index=False,
                           encoding='utf-8-sig' if not wrote_header else 'utf-8')
                wrote_header = True
            if i % 20 == 0:
                print(f'  已写出 {matched_rows:,} 行匹配 AIS')
    if not wrote_header:
        ais_out_path.write_text('', encoding='utf-8')

    # ---------- 阶段 6：匹配审计 ----------
    # 记录全流程的关键数字，形成一条可对账的“数据链路”：
    #   STSD 总行数 → 有效 IMO 行数 → 去重后船数 → 与 AIS 交集船数
    #   → 导出核算的完整船数 / 待补录的不完整船数 → 匹配 AIS 记录行数。
    # 任何一个环节数字异常（例如匹配率过低），都能据此快速定位是
    # STSD 缺字段、IMO 格式问题，还是 AIS 本身覆盖不足。
    audit = {
        'stsd_rows': len(stsd),
        'stsd_valid_imo_rows': int(valid_imo.sum()),
        'stsd_duplicate_imo_rows': int(valid_imo.sum())
            - int(stsd.loc[valid_imo, 'imo'].nunique()),
        'ais_rows': ais_rows,
        'ais_unique_ships': len(ais_imos),
        'matched_ships': len(matched_imo),
        'complete_ships_exported': len(keep),
        'incomplete_ships_not_exported': int(incomplete['imo'].nunique()),
        'ais_records_matched_to_complete_ships': matched_rows,
        'stsd_ships_without_ais_activity': int(
            len(set(stsd.loc[valid_imo, 'imo']) - ais_imos)),
    }
    audit_path = args.outdir / 'match_audit.json'
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                          encoding='utf-8')
    print(f'匹配审计: {audit_path.resolve()}')
    print('\n下一步运行排放核算：')
    print(f'python calculate_ais_co2_only.py --ais-file "{ais_out_path}" '
          f'--vessel-file "{vessels_path}" --reference-dir reference '
          f'--output-prefix data/matched/co2')


if __name__ == '__main__':
    main()