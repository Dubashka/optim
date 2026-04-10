"""
Ядро оптимизатора логистической сети (Polars-версия).
Парсинг данных → построение LP/MILP → решение → извлечение результатов.
"""
import polars as pl
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from collections import defaultdict
import time


# ============================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================

def load_data(shipments_file: str, inventory_file: str, config):
    """Загружает и подготавливает все входные данные."""
    t0 = time.time()

    if shipments_file.endswith(".xlsb"):
        engine = "pyxlsb"
    else:
        engine = "openpyxl"

    # Читаем через pandas (Excel), конвертируем в polars
    df_pd = pd.read_excel(shipments_file, sheet_name="Отгрузки", engine=engine, header=1)
    df = pl.from_pandas(df_pd)
    wh_costs_pd = pd.read_excel(shipments_file, sheet_name="Склады компании", engine=engine)
    wh_costs = pl.from_pandas(wh_costs_pd)
    wh_cap_pd = pd.read_excel(shipments_file, sheet_name="Мощность складов", engine=engine)
    wh_cap = pl.from_pandas(wh_cap_pd)

    # Даты
    if df.schema["Дата"] == pl.Float64:
        df = df.with_columns(
            (pl.lit(pd.Timestamp("1899-12-30")) + pl.duration(days=pl.col("Дата").cast(pl.Int64))).alias("Дата")
        )
    df = df.with_columns(pl.col("Дата").dt.month().alias("month_idx"))

    # Исправление названий
    for col in ["Склад Отправления", "Склад Назначения"]:
        for old, new in config.WAREHOUSE_NAME_FIXES.items():
            df = df.with_columns(
                pl.when(pl.col(col) == old).then(pl.lit(new)).otherwise(pl.col(col)).alias(col)
            )

    # Самовывоз: город назначения = город склада отправления
    df = df.with_columns(
        pl.when(
            (pl.col("Вид Доставки") == "Самовывоз") & pl.col("Город Назначения").is_null()
        ).then(pl.col("Город Отправления")).otherwise(pl.col("Город Назначения")).alias("Город Назначения")
    )

    # Остатки
    inv_pd = pd.read_excel(inventory_file, sheet_name="Остатки")
    inv_raw = pl.from_pandas(inv_pd)
    inv_raw = inv_raw.with_columns(
        pl.col("Склад / Переработчик").str.replace(config.INVENTORY_SUFFIX, "").alias("Склад")
    )
    for old, new in config.WAREHOUSE_NAME_FIXES.items():
        inv_raw = inv_raw.with_columns(
            pl.when(pl.col("Склад") == old).then(pl.lit(new)).otherwise(pl.col("Склад")).alias("Склад")
        )

    print(f"  Загрузка данных: {time.time()-t0:.0f}с")
    return df, wh_costs, wh_cap, inv_raw


def prepare_model_data(df, wh_costs, wh_cap, inv_raw, config):
    """Подготавливает все структуры данных для модели (Polars-версия)."""
    t0 = time.time()

    # Если пришёл pandas DataFrame — конвертируем
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    if isinstance(wh_costs, pd.DataFrame):
        wh_costs = pl.from_pandas(wh_costs)
    if isinstance(wh_cap, pd.DataFrame):
        wh_cap = pl.from_pandas(wh_cap)
    if isinstance(inv_raw, pd.DataFrame):
        inv_raw = pl.from_pandas(inv_raw)

    warehouses = sorted(wh_costs["Склад"].unique().to_list())
    wh_set = set(warehouses)
    pgs = sorted(df["Группа Номенклатуры"].drop_nulls().unique().to_list())
    months = list(range(1, 13))
    days_in_month = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30,
                     7:31, 8:31, 9:30, 10:31, 11:30, 12:29}

    # Справочники затрат
    wh_var = dict(zip(
        wh_costs["Склад"].to_list(),
        wh_costs["переменные затраты на обработку тонны продукции в рублях"].to_list()
    ))
    wh_fix = {}
    for row in wh_costs.iter_rows(named=True):
        v = row["Постоянные затраты рублей в день"]
        wh_fix[row["Склад"]] = v if v is not None else 0.0
    cap_lk = dict(zip(
        wh_cap["Названия строк"].to_list(),
        wh_cap["Пропускная способность в месяц тонн или на вход или на выход"].to_list()
    ))

    # Начальные остатки
    init_inv = {}
    inv_filtered = inv_raw.filter(pl.col("Склад").is_in(list(wh_set)))
    for row in inv_filtered.iter_rows(named=True):
        k = (row["Склад"], row["Продукт"])
        init_inv[k] = init_inv.get(k, 0) + row["Объем"]

    # Разделение потоков
    is_samovyvoz_no_mode = (pl.col("Вид Доставки") == "Самовывоз") & pl.col("Мода").is_null()
    df = df.with_columns(
        pl.when(is_samovyvoz_no_mode).then(pl.lit("СВ")).otherwise(pl.col("Мода")).alias("Мода")
    )

    opt_flow_types = config.OPTIMIZABLE_FLOW_TYPES
    fix_flow_types = config.FIXED_FLOW_TYPES
    all_optim_modes = config.OPTIMIZABLE_MODES + ["СВ"]
    fixed_modes = config.FIXED_MODES

    df_optim = df.filter(
        pl.col("Вид Отправки").is_in(opt_flow_types) &
        pl.col("Мода").is_in(all_optim_modes)
    )

    is_fixed_type = pl.col("Вид Отправки").is_in(fix_flow_types)
    if fixed_modes:
        is_fixed_mode = pl.col("Вид Отправки").is_in(opt_flow_types) & pl.col("Мода").is_in(fixed_modes)
    else:
        is_fixed_mode = pl.lit(False)
    df_fixed = df.filter(is_fixed_type | (is_fixed_mode & ~is_fixed_type))

    sw_df = df_optim.filter(pl.col("Вид Отправки") == "Поставщик - Склад")
    wc_df = df_optim.filter(pl.col("Вид Отправки") == "Склад - Клиент")
    ww_df = df_optim.filter(pl.col("Вид Отправки") == "Склад - Склад")

    # --- Построение дуг с тарифами (Polars groupby — основное ускорение) ---
    def build_mode_arcs_pl(sub, mode, groupby_cols):
        filtered = sub.filter(pl.col("Мода") == mode)
        if filtered.height == 0:
            return []
        agg = filtered.group_by(groupby_cols).agg([
            pl.col("Затраты на транспортировку (LTL), в рублях").sum().alias("c"),
            pl.col("Количество").sum().alias("v"),
        ]).with_columns(
            pl.when(pl.col("v") > 0).then(pl.col("c") / pl.col("v")).otherwise(0.0).alias("t")
        )
        return agg

    def build_direction_arcs_pl(sub, mode, origin_col, dest_col, extra_cols=None):
        """Строит дуги для ЖД/ММ: тариф по направлению, разбивка по группам."""
        filtered = sub.filter(pl.col("Мода") == mode)
        if filtered.height == 0:
            return []
        d = filtered.group_by([origin_col, dest_col]).agg([
            pl.col("Затраты на транспортировку (LTL), в рублях").sum().alias("c"),
            pl.col("Количество").sum().alias("v"),
        ]).with_columns((pl.col("c") / pl.col("v")).alias("t"))

        grp_cols = [origin_col, dest_col, "Группа Номенклатуры"]
        if extra_cols:
            grp_cols += extra_cols
        p = filtered.group_by(grp_cols).agg(
            pl.col("Количество").sum()
        ).join(d.select([origin_col, dest_col, "t"]), on=[origin_col, dest_col])
        return p

    # SW arcs
    swa = build_mode_arcs_pl(sw_df, "Авто", ["Город Отправления", "Склад Назначения", "Группа Номенклатуры"])
    sw_auto = [(r["Город Отправления"], r["Склад Назначения"], r["Группа Номенклатуры"], r["t"], "A")
               for r in swa.iter_rows(named=True) if r["Склад Назначения"] in wh_set] if isinstance(swa, pl.DataFrame) else []

    swj = build_direction_arcs_pl(sw_df, "ЖД", "Город Отправления", "Склад Назначения")
    sw_jd = [(r["Город Отправления"], r["Склад Назначения"], r["Группа Номенклатуры"], r["t"], "J")
             for r in swj.iter_rows(named=True) if r["Склад Назначения"] in wh_set] if isinstance(swj, pl.DataFrame) else []

    swm = build_direction_arcs_pl(sw_df, "ММ", "Город Отправления", "Склад Назначения")
    sw_mm = [(r["Город Отправления"], r["Склад Назначения"], r["Группа Номенклатуры"], r["t"], "M")
             for r in swm.iter_rows(named=True) if r["Склад Назначения"] in wh_set] if isinstance(swm, pl.DataFrame) else []

    # WC arcs
    wca = build_mode_arcs_pl(wc_df, "Авто", ["Склад Отправления", "Город Назначения", "Вид Доставки", "Группа Номенклатуры"])
    wc_auto = [(r["Склад Отправления"], r["Город Назначения"], r["Вид Доставки"], r["Группа Номенклатуры"], r["t"], "A")
               for r in wca.iter_rows(named=True) if r["Склад Отправления"] in wh_set] if isinstance(wca, pl.DataFrame) else []

    def _build_wc_dir_pl(wc_sub, mode, mode_code):
        p = build_direction_arcs_pl(wc_sub, mode, "Склад Отправления", "Город Назначения", ["Вид Доставки"])
        if not isinstance(p, pl.DataFrame):
            return []
        return [(r["Склад Отправления"], r["Город Назначения"], r["Вид Доставки"],
                 r["Группа Номенклатуры"], r["t"], mode_code)
                for r in p.iter_rows(named=True) if r["Склад Отправления"] in wh_set]

    wc_jd = _build_wc_dir_pl(wc_df, "ЖД", "J")
    wc_mm = _build_wc_dir_pl(wc_df, "ММ", "M")

    # Самовывоз
    wc_sv_raw = wc_df.filter(pl.col("Мода") == "СВ")
    wc_sv = []
    if wc_sv_raw.height > 0:
        sv_grp = wc_sv_raw.group_by(["Склад Отправления", "Город Назначения", "Вид Доставки", "Группа Номенклатуры"]).agg(
            pl.col("Количество").sum()
        )
        wc_sv = [(r["Склад Отправления"], r["Город Назначения"], r["Вид Доставки"],
                  r["Группа Номенклатуры"], 0.0, "S")
                 for r in sv_grp.iter_rows(named=True) if r["Склад Отправления"] in wh_set]

    # WW arcs
    wwa = build_mode_arcs_pl(ww_df, "Авто", ["Склад Отправления", "Склад Назначения", "Группа Номенклатуры"])
    ww_auto = [(r["Склад Отправления"], r["Склад Назначения"], r["Группа Номенклатуры"], r["t"], "A")
               for r in wwa.iter_rows(named=True)
               if r["Склад Отправления"] in wh_set and r["Склад Назначения"] in wh_set] if isinstance(wwa, pl.DataFrame) else []

    wwj = build_direction_arcs_pl(ww_df, "ЖД", "Склад Отправления", "Склад Назначения")
    ww_jd = [(r["Склад Отправления"], r["Склад Назначения"], r["Группа Номенклатуры"], r["t"], "J")
             for r in wwj.iter_rows(named=True)
             if r["Склад Отправления"] in wh_set and r["Склад Назначения"] in wh_set] if isinstance(wwj, pl.DataFrame) else []

    wwm = build_direction_arcs_pl(ww_df, "ММ", "Склад Отправления", "Склад Назначения")
    ww_mm = [(r["Склад Отправления"], r["Склад Назначения"], r["Группа Номенклатуры"], r["t"], "M")
             for r in wwm.iter_rows(named=True)
             if r["Склад Отправления"] in wh_set and r["Склад Назначения"] in wh_set] if isinstance(wwm, pl.DataFrame) else []

    all_sw = sw_auto + sw_jd + sw_mm
    all_wc = wc_auto + wc_jd + wc_mm + wc_sv
    all_ww = ww_auto + ww_jd + ww_mm

    # Спрос (Polars groupby)
    dd_agg = wc_df.group_by(["Город Назначения", "Вид Доставки", "Группа Номенклатуры", "month_idx"]).agg(
        pl.col("Количество").sum()
    )
    dd = {(r["Город Назначения"], r["Вид Доставки"], r["Группа Номенклатуры"], r["month_idx"]): r["Количество"]
          for r in dd_agg.iter_rows(named=True)}

    # Фикс. спрос
    mm_wc = df_fixed.filter(pl.col("Вид Отправки") == "Склад - Клиент")
    dd_mm = {}
    if mm_wc.height > 0:
        mm_agg = mm_wc.group_by(["Город Назначения", "Вид Доставки", "Группа Номенклатуры", "month_idx"]).agg(
            pl.col("Количество").sum()
        )
        dd_mm = {(r["Город Назначения"], r["Вид Доставки"], r["Группа Номенклатуры"], r["month_idx"]): r["Количество"]
                 for r in mm_agg.iter_rows(named=True)}
    for k, v in dd_mm.items():
        dd[k] = dd.get(k, 0) + v

    # Ограничения поставщиков
    sa_agg = sw_df.group_by(["Город Отправления", "Группа Номенклатуры"]).agg(
        pl.col("Количество").sum()
    )
    supply_annual = {(r["Город Отправления"], r["Группа Номенклатуры"]): r["Количество"]
                     for r in sa_agg.iter_rows(named=True)}

    # Фиксированные потоки: нетто на складах
    fn = defaultdict(float)
    for row in df_fixed.iter_rows(named=True):
        pg, m, vol = row["Группа Номенклатуры"], row["month_idx"], row["Количество"]
        wd, ws = row["Склад Назначения"], row["Склад Отправления"]
        if wd is not None and wd in wh_set:
            fn[(wd, pg, m)] += vol
        if ws is not None and ws in wh_set:
            fn[(ws, pg, m)] -= vol

    # Фиксированная пропускная способность
    fixed_tp_in = defaultdict(float)
    fixed_tp_out = defaultdict(float)
    for row in df_fixed.iter_rows(named=True):
        m, vol = row["month_idx"], row["Количество"]
        wd, ws = row["Склад Назначения"], row["Склад Отправления"]
        if wd is not None and wd in wh_set:
            fixed_tp_in[(wd, m)] += vol
        if ws is not None and ws in wh_set:
            fixed_tp_out[(ws, m)] += vol

    # Фиксированные затраты
    fixed_transport_cost = df_fixed["Затраты на транспортировку (LTL), в рублях"].sum()
    fixed_var_wh_cost = 0.0
    for row in df_fixed.iter_rows(named=True):
        vol = row["Количество"]
        wd, ws = row["Склад Назначения"], row["Склад Отправления"]
        if wd is not None and wd in wh_set:
            fixed_var_wh_cost += wh_var.get(wd, 0) * vol
        if ws is not None and ws in wh_set:
            fixed_var_wh_cost += wh_var.get(ws, 0) * vol

    # Расчётный исходящий остаток (AS-IS)
    ann_in = defaultdict(float)
    ann_out = defaultdict(float)
    in_types = {"Поставщик - Склад", "Склад - Склад", "Переработчик - Склад"}
    out_types = {"Склад - Клиент", "Склад - Склад", "Склад - Переработчик"}
    for row in df.iter_rows(named=True):
        vt, pg, vol = row["Вид Отправки"], row["Группа Номенклатуры"], row["Количество"]
        wd, ws = row["Склад Назначения"], row["Склад Отправления"]
        if wd is not None and wd in wh_set and vt in in_types:
            ann_in[(wd, pg)] += vol
        if ws is not None and ws in wh_set and vt in out_types:
            ann_out[(ws, pg)] += vol
    end_inv_target = {}
    for k in set(list(init_inv.keys()) + list(ann_in.keys()) + list(ann_out.keys())):
        ei = init_inv.get(k, 0) + ann_in.get(k, 0) - ann_out.get(k, 0)
        if ei > 0.01:
            end_inv_target[k] = ei

    # AS-IS затраты
    as_is_transport = df["Затраты на транспортировку (LTL), в рублях"].sum()

    # Throughput по складам (Polars)
    out_df = df.filter(pl.col("Склад Отправления").is_not_null()).group_by("Склад Отправления").agg(
        pl.col("Количество").sum().alias("o")
    )
    in_df = df.filter(pl.col("Склад Назначения").is_not_null()).group_by("Склад Назначения").agg(
        pl.col("Количество").sum().alias("i")
    )
    out_dict = dict(zip(out_df["Склад Отправления"].to_list(), out_df["o"].to_list()))
    in_dict = dict(zip(in_df["Склад Назначения"].to_list(), in_df["i"].to_list()))
    all_whs = set(out_dict.keys()) | set(in_dict.keys())
    as_is_var_wh = sum(wh_var.get(w, 0) * (out_dict.get(w, 0) + in_dict.get(w, 0)) for w in all_whs)
    as_is_fix_wh = sum(v * sum(days_in_month.values()) for v in wh_fix.values())
    as_is_total = as_is_transport + as_is_var_wh + as_is_fix_wh

    as_is_sw = sw_df["Затраты на транспортировку (LTL), в рублях"].sum()
    as_is_wc = wc_df["Затраты на транспортировку (LTL), в рублях"].sum()
    as_is_ww = ww_df["Затраты на транспортировку (LTL), в рублях"].sum()

    # Конвертируем df_fixed обратно в pandas для solve()
    df_fixed_pd = df_fixed.to_pandas()

    # Сохраняем исходный df отгрузок для тарифного модуля
    df_shipments_pd = df.to_pandas()

    data = {
        "warehouses": warehouses, "wh_set": wh_set, "pgs": pgs,
        "months": months, "days_in_month": days_in_month,
        "wh_var": wh_var, "wh_fix": wh_fix, "cap_lk": cap_lk,
        "init_inv": init_inv, "end_inv_target": end_inv_target,
        "all_sw": all_sw, "all_wc": all_wc, "all_ww": all_ww,
        "dd": dd, "dd_mm": dd_mm,
        "supply_annual": supply_annual,
        "fn": fn, "fixed_tp_in": fixed_tp_in, "fixed_tp_out": fixed_tp_out,
        "fixed_transport_cost": fixed_transport_cost,
        "fixed_var_wh_cost": fixed_var_wh_cost,
        "as_is": {
            "total": as_is_total,
            "transport_sw": as_is_sw,
            "transport_wc": as_is_wc,
            "transport_ww": as_is_ww,
            "transport_fixed": fixed_transport_cost,
            "var_wh_opt": as_is_var_wh - fixed_var_wh_cost,
            "var_wh_fixed": fixed_var_wh_cost,
            "fix_wh": as_is_fix_wh,
        },
        "sw_df": sw_df.to_pandas(), "wc_df": wc_df.to_pandas(), "ww_df": ww_df.to_pandas(),
        "df_fixed": df_fixed_pd,
        "df_shipments": df_shipments_pd,
    }
    print(f"  Подготовка данных (Polars): {time.time()-t0:.0f}с")
    print(f"  Дуги: SW={len(all_sw)}, WC={len(all_wc)}, WW={len(all_ww)}")
    return data


# ============================================================
# 2. ПОСТРОЕНИЕ И РЕШЕНИЕ МОДЕЛИ
# ============================================================

def solve(data, config):
    """Строит LP, решает, возвращает результаты."""
    t0 = time.time()

    D = data
    months = D["months"]
    warehouses = D["warehouses"]
    wh_set = D["wh_set"]
    pgs = D["pgs"]
    wh_var = D["wh_var"]
    wh_fix = D["wh_fix"]
    cap_lk = D["cap_lk"]
    dim = D["days_in_month"]

    PU = config.PENALTY_UNMET_DEMAND
    slv = pywraplp.Solver.CreateSolver(config.SOLVER_NAME)
    if not slv:
        raise RuntimeError(f"Солвер {config.SOLVER_NAME} недоступен")

    # --- Переменные ---
    svv = {}; wcv = {}; wwv = {}; uv = {}; iv = {}
    wip = defaultdict(list); wop = defaultdict(list)
    ci_ = defaultdict(list); sav_ = defaultdict(list)
    wh_tp_in = defaultdict(list)
    wh_tp_out = defaultdict(list)

    for s, w, pg, t, md in D["all_sw"]:
        for m in months:
            v = slv.NumVar(0, slv.infinity(), "")
            svv[(s, w, pg, m, md)] = v
            wip[(w, pg, m)].append(v)
            sav_[(s, pg)].append(v)
            wh_tp_in[(w, m)].append(v)

    for w, c, dt, pg, t, md in D["all_wc"]:
        for m in months:
            v = slv.NumVar(0, slv.infinity(), "")
            wcv[(w, c, dt, pg, m, md)] = v
            wop[(w, pg, m)].append(v)
            ci_[(c, dt, pg, m)].append(v)
            wh_tp_out[(w, m)].append(v)

    for w1, w2, pg, t, md in D["all_ww"]:
        for m in months:
            v = slv.NumVar(0, slv.infinity(), "")
            wwv[(w1, w2, pg, m, md)] = v
            wop[(w1, pg, m)].append(v)
            wip[(w2, pg, m)].append(v)
            wh_tp_out[(w1, m)].append(v)
            wh_tp_in[(w2, m)].append(v)

    for wh in warehouses:
        for pg in pgs:
            for m in months:
                iv[(wh, pg, m)] = slv.NumVar(0, slv.infinity(), "")

    for k, vol in D["dd"].items():
        uv[k] = slv.NumVar(0, vol, "")

    print(f"  Переменные: {slv.NumVariables():,}")

    # --- Целевая функция ---
    obj = slv.Objective()
    st_ = {(s, w, pg, md): t for s, w, pg, t, md in D["all_sw"]}
    wct_ = {(w, c, dt, pg, md): t for w, c, dt, pg, t, md in D["all_wc"]}
    wwt_ = {(w1, w2, pg, md): t for w1, w2, pg, t, md in D["all_ww"]}

    for k, v in svv.items():
        obj.SetCoefficient(v, st_.get((k[0], k[1], k[2], k[4]), 0) + wh_var.get(k[1], 0))
    for k, v in wcv.items():
        obj.SetCoefficient(v, wct_.get((k[0], k[1], k[2], k[3], k[5]), 0) + wh_var.get(k[0], 0))
    for k, v in wwv.items():
        obj.SetCoefficient(v, wwt_.get((k[0], k[1], k[2], k[4]), 0)
                           + wh_var.get(k[0], 0) + wh_var.get(k[1], 0))
    for k, v in uv.items():
        obj.SetCoefficient(v, PU)
    obj.SetMinimization()

    # --- Ограничения ---
    dd_mm = D["dd_mm"]
    for k, vol in D["dd"].items():
        mm_s = dd_mm.get(k, 0)
        rem = vol - mm_s
        if rem < 0.01:
            continue
        c = slv.Constraint(rem, slv.infinity())
        for v in ci_.get(k, []):
            c.SetCoefficient(v, 1)
        if k in uv:
            c.SetCoefficient(uv[k], 1)

    fn = D["fn"]
    init_inv = D["init_inv"]
    for wh in warehouses:
        for pg in pgs:
            for m in months:
                f = fn.get((wh, pg, m), 0)
                i0 = init_inv.get((wh, pg), 0) if m == 1 else 0
                c = slv.Constraint(-(f + i0), -(f + i0))
                for v in wip.get((wh, pg, m), []):
                    c.SetCoefficient(v, 1)
                for v in wop.get((wh, pg, m), []):
                    c.SetCoefficient(v, -1)
                c.SetCoefficient(iv[(wh, pg, m)], -1)
                if m > 1:
                    c.SetCoefficient(iv[(wh, pg, m - 1)], 1)

    fixed_tp_in = D["fixed_tp_in"]
    fixed_tp_out = D["fixed_tp_out"]
    for wh in warehouses:
        cap = cap_lk.get(wh, None)
        if cap is None or (isinstance(cap, float) and np.isnan(cap)):
            continue
        for m in months:
            ft_in = fixed_tp_in.get((wh, m), 0)
            rc_in = max(cap - ft_in, 0)
            ct_in = slv.Constraint(0, rc_in)
            for v in wh_tp_in.get((wh, m), []):
                ct_in.SetCoefficient(v, 1)
            ft_out = fixed_tp_out.get((wh, m), 0)
            rc_out = max(cap - ft_out, 0)
            ct_out = slv.Constraint(0, rc_out)
            for v in wh_tp_out.get((wh, m), []):
                ct_out.SetCoefficient(v, 1)

    supply_annual = D["supply_annual"]
    for (s, pg), av in supply_annual.items():
        vl = sav_.get((s, pg), [])
        if not vl:
            continue
        c = slv.Constraint(0, av)
        for v in vl:
            c.SetCoefficient(v, 1)
        if s in config.MANDATORY_SUPPLIERS:
            c2 = slv.Constraint(av, slv.infinity())
            for v in vl:
                c2.SetCoefficient(v, 1)

    if config.ENFORCE_ENDING_INVENTORY:
        for (wh, pg), tgt in D["end_inv_target"].items():
            if (wh, pg, 12) in iv:
                c = slv.Constraint(tgt, slv.infinity())
                c.SetCoefficient(iv[(wh, pg, 12)], 1)

    print(f"  Ограничения: {slv.NumConstraints():,}")
    print(f"  Построение: {time.time()-t0:.0f}с")

    # --- Решение ---
    t1 = time.time()
    slv.SetTimeLimit(config.SOLVER_TIME_LIMIT * 1000)
    status = slv.Solve()
    solve_time = time.time() - t1

    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        print(f"  [X] Решение не найдено (статус: {status})")
        return None

    status_str = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"
    print(f"  Решение: {status_str}, {solve_time:.1f}с")

    # --- Извлечение результатов ---
    FW = sum(wh_fix.get(wh, 0) * sum(dim.values()) for wh in warehouses)

    tobe_sw = sum(st_.get((k[0], k[1], k[2], k[4]), 0) * v.solution_value()
                  for k, v in svv.items())
    tobe_wc = sum(wct_.get((k[0], k[1], k[2], k[3], k[5]), 0) * v.solution_value()
                  for k, v in wcv.items())
    tobe_ww = sum(wwt_.get((k[0], k[1], k[2], k[4]), 0) * v.solution_value()
                  for k, v in wwv.items())
    tobe_var = (sum(wh_var.get(k[1], 0) * v.solution_value() for k, v in svv.items())
                + sum(wh_var.get(k[0], 0) * v.solution_value() for k, v in wcv.items())
                + sum((wh_var.get(k[0], 0) + wh_var.get(k[1], 0)) * v.solution_value()
                      for k, v in wwv.items()))
    tobe_penalty = sum(PU * v.solution_value() for v in uv.values())
    tobe_unmet = sum(v.solution_value() for v in uv.values())

    tobe_total = (tobe_sw + tobe_wc + tobe_ww + tobe_var
                  + D["fixed_transport_cost"] + D["fixed_var_wh_cost"]
                  + FW + tobe_penalty)

    # Потоки TO-BE
    mode_map = {"A": "Авто", "J": "ЖД", "M": "ММ", "S": "Самовывоз"}
    flows = []

    for (s, w, pg, m, md), v in svv.items():
        val = v.solution_value()
        if val > 0.01:
            tariff = st_.get((s, w, pg, md), 0)
            wh_in = wh_var.get(w, 0)
            flows.append({
                "Отправитель": s, "Тип отправителя": "Поставщик",
                "Получатель": w, "Тип получателя": "Склад",
                "Вид Отправки": "Поставщик - Склад",
                "Вид Доставки": "Доставка",
                "Группа": pg, "Месяц": m, "Мода": mode_map.get(md, md),
                "Объём": val,
                "Исх. склад. обработка, руб/т": 0.0,
                "Исх. склад. обработка, руб": 0.0,
                "Транспорт, руб/т": tariff,
                "Транспорт, руб": tariff * val,
                "Вх. склад. обработка, руб/т": wh_in,
                "Вх. склад. обработка, руб": wh_in * val,
                "Оптимизировано": True,
            })

    for (w, c, dt, pg, m, md), v in wcv.items():
        val = v.solution_value()
        if val > 0.01:
            tariff = wct_.get((w, c, dt, pg, md), 0)
            wh_out = wh_var.get(w, 0)
            flows.append({
                "Отправитель": w, "Тип отправителя": "Склад",
                "Получатель": c, "Тип получателя": "Клиент",
                "Вид Отправки": "Склад - Клиент",
                "Вид Доставки": dt,
                "Группа": pg, "Месяц": m,
                "Мода": mode_map.get(md, md),
                "Объём": val,
                "Исх. склад. обработка, руб/т": wh_out,
                "Исх. склад. обработка, руб": wh_out * val,
                "Транспорт, руб/т": tariff,
                "Транспорт, руб": tariff * val,
                "Вх. склад. обработка, руб/т": 0.0,
                "Вх. склад. обработка, руб": 0.0,
                "Оптимизировано": True,
            })

    for (w1, w2, pg, m, md), v in wwv.items():
        val = v.solution_value()
        if val > 0.01:
            tariff = wwt_.get((w1, w2, pg, md), 0)
            wh_out = wh_var.get(w1, 0)
            wh_in = wh_var.get(w2, 0)
            flows.append({
                "Отправитель": w1, "Тип отправителя": "Склад",
                "Получатель": w2, "Тип получателя": "Склад",
                "Вид Отправки": "Склад - Склад",
                "Вид Доставки": "Доставка",
                "Группа": pg, "Месяц": m, "Мода": mode_map.get(md, md),
                "Объём": val,
                "Исх. склад. обработка, руб/т": wh_out,
                "Исх. склад. обработка, руб": wh_out * val,
                "Транспорт, руб/т": tariff,
                "Транспорт, руб": tariff * val,
                "Вх. склад. обработка, руб/т": wh_in,
                "Вх. склад. обработка, руб": wh_in * val,
                "Оптимизировано": True,
            })

    # Фиксированные потоки
    df_fixed_report = D.get("df_fixed")
    if df_fixed_report is not None and len(df_fixed_report) > 0:
        fix_rpt = df_fixed_report.copy()
        for col in ["Склад Отправления", "Склад Назначения", "Город Отправления",
                     "Город Назначения", "Мода", "Группа Номенклатуры", "Вид Доставки"]:
            if col in fix_rpt.columns:
                fix_rpt[col] = fix_rpt[col].fillna("")
        grp_cols = ["Вид Отправки", "Вид Доставки", "Склад Отправления", "Город Отправления",
                     "Склад Назначения", "Город Назначения", "Мода", "Группа Номенклатуры", "month_idx"]
        for _, r in fix_rpt.groupby(grp_cols).agg({
            "Количество": "sum",
            "Затраты на транспортировку (LTL), в рублях": "sum",
        }).reset_index().iterrows():
            vt = r["Вид Отправки"]
            parts = vt.split(" - ")
            type_from = parts[0] if len(parts) == 2 else "Неизвестно"
            type_to = parts[1] if len(parts) == 2 else "Неизвестно"
            sender = r["Склад Отправления"] if r["Склад Отправления"] else r["Город Отправления"]
            receiver = r["Склад Назначения"] if r["Склад Назначения"] else r["Город Назначения"]
            vol = r["Количество"]
            transport_cost = r["Затраты на транспортировку (LTL), в рублях"]
            transport_rate = transport_cost / vol if vol > 0 else 0
            wh_out_rate = wh_var.get(r["Склад Отправления"], 0) if r["Склад Отправления"] in wh_set else 0
            wh_in_rate = wh_var.get(r["Склад Назначения"], 0) if r["Склад Назначения"] in wh_set else 0
            moda = r["Мода"]
            vid_dostavki = r["Вид Доставки"]
            if not moda and vid_dostavki == "Самовывоз":
                moda = "Самовывоз"
            flows.append({
                "Отправитель": sender, "Тип отправителя": type_from,
                "Получатель": receiver, "Тип получателя": type_to,
                "Вид Отправки": vt, "Вид Доставки": vid_dostavki,
                "Группа": r["Группа Номенклатуры"], "Месяц": r["month_idx"],
                "Мода": moda, "Объём": vol,
                "Исх. склад. обработка, руб/т": wh_out_rate,
                "Исх. склад. обработка, руб": wh_out_rate * vol,
                "Транспорт, руб/т": transport_rate,
                "Транспорт, руб": transport_cost,
                "Вх. склад. обработка, руб/т": wh_in_rate,
                "Вх. склад. обработка, руб": wh_in_rate * vol,
                "Оптимизировано": False,
            })

    flows_df = pd.DataFrame(flows) if flows else pd.DataFrame()

    # --- Тарифный модуль ---
    tariff_df = pd.DataFrame()
    neighbors_path = getattr(config, "TARIFF_NEIGHBORS_PATH", None)
    city_region_path = getattr(config, "TARIFF_CITY_REGION_PATH", None)
    if not flows_df.empty:
        try:
            from tariff_calculator import (
                build_tariff_reference,
                load_neighbors,
                load_city_region,
                assign_tariffs,
            )
            df_shipments = D.get("df_shipments")
            if df_shipments is not None and len(df_shipments) > 0:
                refs = build_tariff_reference(df_shipments)
                neighbors = load_neighbors(neighbors_path) if neighbors_path else {}
                city_region = load_city_region(city_region_path) if city_region_path else {}
                flows_df, tariff_df = assign_tariffs(
                    flows_df, refs, neighbors, city_region,
                    tariff_col=getattr(config, "TARIFF_COLUMN_NAME", "Тариф авто, руб/т"),
                )
                print(f"  Тарифный модуль: {len(tariff_df)} новых авто-маршрутов")
        except Exception as e:
            print(f"  [WARN] Тарифный модуль пропущен: {e}")

    # Таблица спроса
    demand_rows = []
    for k, vol in D["dd"].items():
        city, dt, pg, m = k
        mm_s = dd_mm.get(k, 0)
        opt_supplied = sum(v.solution_value() for v in ci_.get(k, []))
        total_supplied = opt_supplied + mm_s
        unmet_val = uv[k].solution_value() if k in uv else 0.0
        penalty_val = unmet_val * PU
        pct = (total_supplied / vol * 100) if vol > 0.01 else 100.0
        demand_rows.append({
            "Клиент": city, "Вид доставки": dt, "Месяц": m,
            "Группа номенклатуры": pg,
            "Требуемый спрос, т": round(vol, 2),
            "Покрытый спрос, т": round(total_supplied, 2),
            "Невыполненный спрос, т": round(unmet_val, 2),
            "% удовлетворения": round(pct, 1),
            "Штраф, руб": round(penalty_val, 2),
        })
    demand_df = pd.DataFrame(demand_rows) if demand_rows else pd.DataFrame()

    results = {
        "status": status_str,
        "solve_time": solve_time,
        "to_be": {
            "total": tobe_total,
            "transport_sw": tobe_sw,
            "transport_wc": tobe_wc,
            "transport_ww": tobe_ww,
            "transport_fixed": D["fixed_transport_cost"],
            "var_wh_opt": tobe_var,
            "var_wh_fixed": D["fixed_var_wh_cost"],
            "fix_wh": FW,
            "penalty": tobe_penalty,
            "unmet_demand": tobe_unmet,
        },
        "as_is": D["as_is"],
        "flows": flows_df,
        "demand": demand_df,
        "tariffs": tariff_df,
    }
    return results


# ============================================================
# 3. ФОРМИРОВАНИЕ ОТЧЁТА
# ============================================================

def build_summary(results: dict) -> pd.DataFrame:
    a = results["as_is"]
    t = results["to_be"]
    rows = [
        ("Входящий транспорт (Пост->Склад)", a["transport_sw"], t["transport_sw"]),
        ("Исходящий транспорт (Склад->Клиент)", a["transport_wc"], t["transport_wc"]),
        ("Внутр. перемещения (Склад->Склад)", a["transport_ww"], t["transport_ww"]),
        ("Фикс. транспорт (ММ+перераб.)", a["transport_fixed"], t["transport_fixed"]),
        ("Перем. складские (оптимизируемые)", a["var_wh_opt"], t["var_wh_opt"]),
        ("Перем. складские (фиксированные)", a["var_wh_fixed"], t["var_wh_fixed"]),
        ("Постоянные складские", a["fix_wh"], t["fix_wh"]),
    ]
    import numpy as np
    df = pd.DataFrame(rows, columns=["Статья затрат", "AS-IS", "TO-BE"])
    df["D"] = df["AS-IS"] - df["TO-BE"]
    df["D%"] = np.where(df["AS-IS"] > 0, df["D"] / df["AS-IS"] * 100, 0)
    total = pd.DataFrame([{
        "Статья затрат": "ИТОГО",
        "AS-IS": a["total"],
        "TO-BE": t["total"] - t["penalty"],
        "D": a["total"] - (t["total"] - t["penalty"]),
        "D%": (a["total"] - (t["total"] - t["penalty"])) / a["total"] * 100,
    }])
    df = pd.concat([df, total], ignore_index=True)
    return df


def save_results(results: dict, output_file: str, config=None):
    summary = build_summary(results)
    export_tariff_sheet = getattr(config, "TARIFF_EXPORT_SHEET", True) if config else True
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Сводка", index=False)
        flows = results.get("flows")
        if flows is not None and len(flows) > 0:
            flows.to_excel(writer, sheet_name="Потоки", index=False)
        demand = results.get("demand")
        if demand is not None and len(demand) > 0:
            demand.to_excel(writer, sheet_name="Спрос", index=False)
        if export_tariff_sheet:
            tariffs = results.get("tariffs")
            if tariffs is not None and isinstance(tariffs, pd.DataFrame) and len(tariffs) > 0:
                tariffs.to_excel(writer, sheet_name="Тарифы авто", index=False)
    print(f"  Результаты сохранены: {output_file}")
