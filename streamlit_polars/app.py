"""
Streamlit-интерфейс для оптимизатора логистической сети.
Загрузка файла -> настройка параметров -> запуск AS-IS / TO-BE -> сравнение экспериментов.
"""
import sys, os, time, io, copy
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import streamlit as st

# Добавляем путь к модулю оптимизатора
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "logistics_optimizer_polars"))
import config as base_config
from validate import validate_data
from optimizer import load_data, prepare_model_data, solve, build_summary, save_results


# ============================================================
# Вспомогательные функции
# ============================================================

class Config:
    """Копия параметров из base_config с возможностью переопределения."""
    def __init__(self, **overrides):
        # Копируем все публичные атрибуты из модуля
        for attr in dir(base_config):
            if not attr.startswith("_"):
                val = getattr(base_config, attr)
                if not callable(val):
                    setattr(self, attr, copy.deepcopy(val))
        # Применяем переопределения
        for k, v in overrides.items():
            setattr(self, k, v)


def make_config(**overrides):
    """Создаёт копию конфига с переопределёнными параметрами."""
    return Config(**overrides)


CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _cache_path(filename):
    """Путь к кэш-файлу для данного исходного файла."""
    base = os.path.splitext(filename)[0]
    return os.path.join(CACHE_DIR, f"{base}.parquet_cache")


def _save_cache(filename, df, wh_costs, wh_cap, inv_raw):
    """Сохраняет загруженные данные в быстрый формат."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = _cache_path(filename)
    os.makedirs(cache, exist_ok=True)
    df.to_parquet(os.path.join(cache, "shipments.parquet"))
    wh_costs.to_parquet(os.path.join(cache, "wh_costs.parquet"))
    wh_cap.to_parquet(os.path.join(cache, "wh_cap.parquet"))
    if inv_raw is not None:
        inv_raw.to_parquet(os.path.join(cache, "inv_raw.parquet"))


def _load_cache(filename):
    """Загружает данные из кэша. Возвращает None если кэша нет."""
    cache = _cache_path(filename)
    shipments_path = os.path.join(cache, "shipments.parquet")
    if not os.path.exists(shipments_path):
        return None
    df = pd.read_parquet(shipments_path)
    wh_costs = pd.read_parquet(os.path.join(cache, "wh_costs.parquet"))
    wh_cap = pd.read_parquet(os.path.join(cache, "wh_cap.parquet"))
    inv_path = os.path.join(cache, "inv_raw.parquet")
    inv_raw = pd.read_parquet(inv_path) if os.path.exists(inv_path) else None
    return df, wh_costs, wh_cap, inv_raw


def _load_parquet_set(parquet_dir):
    """Загружает 4 parquet-файла из папки."""
    names = {"Отгрузки": "shipments", "Склады компании": "wh_costs",
             "Мощность складов": "wh_cap", "Остатки": "inv_raw"}
    result = {}
    for ru_name, en_name in names.items():
        for candidate in [f"{ru_name}.parquet", f"{en_name}.parquet"]:
            path = os.path.join(parquet_dir, candidate)
            if os.path.exists(path):
                result[ru_name] = pd.read_parquet(path)
                break
    return result


def load_from_upload(uploaded_file):
    """Загружает данные из загруженного файла. Поддерживает Excel и Parquet."""
    fname = uploaded_file.name

    # --- Parquet ---
    if fname.endswith(".parquet"):
        buf = io.BytesIO(uploaded_file.read())
        first_df = pd.read_parquet(buf)

        parquet_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "input", "parquet"),
            os.path.join(os.path.dirname(__file__), "..", "input"),
        ]
        loaded = {}
        for d in parquet_dirs:
            if os.path.isdir(d):
                loaded = _load_parquet_set(d)
                if len(loaded) >= 3:
                    break

        base = os.path.splitext(fname)[0]
        for ru_name in ["Отгрузки", "Склады компании", "Мощность складов", "Остатки"]:
            if base == ru_name or base in ["shipments", "wh_costs", "wh_cap", "inv_raw"]:
                loaded[ru_name] = first_df
                break
        else:
            cols = set(first_df.columns)
            if "Количество" in cols and "Вид Отправки" in cols:
                loaded["Отгрузки"] = first_df
            elif "Склад" in cols and "переменные" in " ".join(str(c) for c in cols).lower():
                loaded["Склады компании"] = first_df

        df = loaded.get("Отгрузки")
        wh_costs = loaded.get("Склады компании")
        wh_cap = loaded.get("Мощность складов")
        inv_raw = loaded.get("Остатки")

        if df is None or wh_costs is None:
            raise ValueError("Не удалось найти все parquet-файлы. "
                             "Убедитесь что в папке input/parquet лежат: "
                             "Отгрузки.parquet, Склады компании.parquet, "
                             "Мощность складов.parquet, Остатки.parquet")
        return df, wh_costs, wh_cap, inv_raw

    # --- Excel ---
    cached = _load_cache(fname)
    if cached is not None:
        return cached

    buf = io.BytesIO(uploaded_file.read())

    if fname.endswith(".xlsb"):
        engine = "pyxlsb"
    else:
        engine = "openpyxl"

    df = pd.read_excel(buf, sheet_name="Отгрузки", engine=engine, header=1)
    buf.seek(0)
    wh_costs = pd.read_excel(buf, sheet_name="Склады компании", engine=engine)
    buf.seek(0)
    wh_cap = pd.read_excel(buf, sheet_name="Мощность складов", engine=engine)
    buf.seek(0)

    try:
        inv_raw = pd.read_excel(buf, sheet_name="Остатки", engine=engine)
    except Exception:
        inv_raw = None

    _save_cache(fname, df, wh_costs, wh_cap, inv_raw)

    return df, wh_costs, wh_cap, inv_raw


# ============================================================
# Валидация входных данных
# ============================================================

# Ожидаемые колонки для каждого листа
EXPECTED_COLUMNS = {
    "Отгрузки": {
        "required": [
            "Вид Отправки", "Вид Доставки", "Дата", "Мода",
            "Склад Отправления", "Город Отправления",
            "Склад Назначения", "Город Назначения",
            "Группа Номенклатуры", "Количество",
            "Затраты на транспортировку (LTL), в рублях",
        ],
        "optional": [
            "Грузоотправитель", "Код Грузоотправителя",
            "Регион Отправления", "Грузополучатель",
            "Код Грузополучателя", "Регион Назначения",
            "Номенклатура", "Характеристика",
            "Перевозчик", "НомерТС", "Грузоподъемность В Тоннах",
        ],
    },
    "Склады компании": {
        "required": [
            "Склад",
            "переменные затраты на обработку тонны продукции в рублях",
            "Постоянные затраты рублей в день",
        ],
        "optional": [],
    },
    "Мощность складов": {
        "required": [
            "Названия строк",
            "Пропускная способность в месяц тонн или на вход или на выход",
        ],
        "optional": ["Емкость хранения тонн", "Тип склада"],
    },
    "Остатки": {
        "required": ["Склад / Переработчик", "Продукт", "Объем"],
        "optional": [],
    },
}


def _find_similar(target, candidates, top_n=3, threshold=0.4):
    """Находит похожие строки по коэффициенту совпадения."""
    scored = []
    target_lower = target.lower().strip()
    for c in candidates:
        # Точное совпадение без учёта регистра
        if c.lower().strip() == target_lower:
            scored.append((c, 1.0))
            continue
        ratio = SequenceMatcher(None, target_lower, c.lower().strip()).ratio()
        if ratio >= threshold:
            scored.append((c, ratio))
    scored.sort(key=lambda x: -x[1])
    return [name for name, _ in scored[:top_n]]


def validate_columns(sheets: dict) -> list:
    """
    Проверяет наличие обязательных листов и колонок.
    Возвращает список ошибок (пустой = всё ок).
    sheets: {"Отгрузки": df, "Склады компании": df, ...}
    """
    errors = []

    for sheet_name, spec in EXPECTED_COLUMNS.items():
        if sheet_name not in sheets or sheets[sheet_name] is None:
            if sheet_name == "Остатки":
                continue  # остатки могут быть в отдельном файле
            errors.append(f"Не найден лист **'{sheet_name}'**")
            continue

        df = sheets[sheet_name]
        actual_cols = list(df.columns)

        for req_col in spec["required"]:
            if req_col in actual_cols:
                continue

            # Ищем похожие
            similar = _find_similar(req_col, actual_cols)
            msg = f"Лист **'{sheet_name}'**: не найдена колонка **'{req_col}'**"
            if similar:
                suggestions = ", ".join([f"'{s}'" for s in similar])
                msg += f". Похожие: {suggestions}"
            else:
                msg += f". Имеющиеся колонки: {', '.join(actual_cols[:10])}"
                if len(actual_cols) > 10:
                    msg += f" ... (ещё {len(actual_cols)-10})"
            errors.append(msg)

    return errors


def try_auto_rename(sheets: dict) -> tuple:
    """
    Пытается автоматически переименовать колонки, если есть точное совпадение
    без учёта регистра/пробелов. Возвращает (sheets, warnings).
    """
    warnings = []

    for sheet_name, spec in EXPECTED_COLUMNS.items():
        if sheet_name not in sheets or sheets[sheet_name] is None:
            continue

        df = sheets[sheet_name]
        rename_map = {}

        for req_col in spec["required"] + spec["optional"]:
            if req_col in df.columns:
                continue
            # Ищем совпадение без учёта регистра и лишних пробелов
            for actual in df.columns:
                if actual.lower().strip() == req_col.lower().strip() and actual != req_col:
                    rename_map[actual] = req_col
                    break

        if rename_map:
            sheets[sheet_name] = df.rename(columns=rename_map)
            for old, new in rename_map.items():
                warnings.append(f"Лист '{sheet_name}': '{old}' -> '{new}'")

    return sheets, warnings


def get_missing_columns(sheets: dict) -> dict:
    """
    Возвращает словарь {sheet_name: [(missing_col, [similar_cols]), ...]}.
    Только листы/колонки с проблемами.
    """
    missing = {}
    for sheet_name, spec in EXPECTED_COLUMNS.items():
        if sheet_name not in sheets or sheets[sheet_name] is None:
            continue
        actual_cols = list(sheets[sheet_name].columns)
        sheet_missing = []
        for req_col in spec["required"]:
            if req_col not in actual_cols:
                similar = _find_similar(req_col, actual_cols)
                sheet_missing.append((req_col, similar))
        if sheet_missing:
            missing[sheet_name] = sheet_missing
    return missing


def apply_column_mapping(sheets: dict, mapping: dict) -> dict:
    """
    Применяет маппинг колонок.
    mapping: {sheet_name: {actual_col: expected_col, ...}, ...}
    """
    for sheet_name, col_map in mapping.items():
        if sheet_name in sheets and sheets[sheet_name] is not None:
            rename = {actual: expected for actual, expected in col_map.items() if actual and actual != expected}
            if rename:
                sheets[sheet_name] = sheets[sheet_name].rename(columns=rename)
    return sheets


def preprocess(df, wh_costs, inv_raw, cfg):
    """Предобработка: даты, названия, самовывоз, суффиксы."""
    # Даты
    if df["Дата"].dtype == np.float64:
        df["Дата"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["Дата"], unit="D")
    df["month_idx"] = df["Дата"].dt.month

    # Исправление названий
    for col in ["Склад Отправления", "Склад Назначения"]:
        for old, new in cfg.WAREHOUSE_NAME_FIXES.items():
            df[col] = df[col].replace(old, new)

    # Самовывоз
    mask = (df["Вид Доставки"] == "Самовывоз") & (df["Город Назначения"].isna())
    df.loc[mask, "Город Назначения"] = df.loc[mask, "Город Отправления"]

    # Остатки
    if inv_raw is not None and "Склад / Переработчик" in inv_raw.columns:
        inv_raw["Склад"] = inv_raw["Склад / Переработчик"].str.replace(
            cfg.INVENTORY_SUFFIX, "", regex=False
        )
        for old, new in cfg.WAREHOUSE_NAME_FIXES.items():
            inv_raw["Склад"] = inv_raw["Склад"].replace(old, new)

    return df, wh_costs, inv_raw


def apply_cost_markup(wh_costs, var_markup_pct, fix_markup_pct):
    """Применяет наценку к складским затратам."""
    wh = wh_costs.copy()
    col_var = "переменные затраты на обработку тонны продукции в рублях"
    col_fix = "Постоянные затраты рублей в день"
    if var_markup_pct != 0:
        wh[col_var] = wh[col_var] * (1 + var_markup_pct / 100)
    if fix_markup_pct != 0:
        wh[col_fix] = wh[col_fix] * (1 + fix_markup_pct / 100)
    return wh


def calc_as_is(df, wh_costs, wh_cap, inv_raw, cfg):
    """Считает AS-IS затраты (без оптимизации, просто факт). Возвращает (costs_dict, flows_df)."""
    warehouses = sorted(wh_costs["Склад"].unique())
    wh_set = set(warehouses)

    wh_var = dict(zip(wh_costs["Склад"],
                      wh_costs["переменные затраты на обработку тонны продукции в рублях"]))
    wh_fix = {}
    for _, r in wh_costs.iterrows():
        v = r["Постоянные затраты рублей в день"]
        wh_fix[r["Склад"]] = v if pd.notna(v) else 0.0

    days_in_month = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30,
                     7:31, 8:31, 9:30, 10:31, 11:30, 12:29}

    # Транспортные затраты по типам
    # Самовывоз без моды — оптимизируемый (тариф = 0)
    is_samovyvoz_no_mode = (df["Вид Доставки"] == "Самовывоз") & (df["Мода"].isna())
    df.loc[is_samovyvoz_no_mode, "Мода"] = "СВ"

    is_fixed_type = df["Вид Отправки"].isin(cfg.FIXED_FLOW_TYPES)
    is_fixed_mode = (df["Вид Отправки"].isin(cfg.OPTIMIZABLE_FLOW_TYPES)
                     & df["Мода"].isin(cfg.FIXED_MODES)) if cfg.FIXED_MODES else pd.Series(False, index=df.index)

    all_optim_modes = cfg.OPTIMIZABLE_MODES + ["СВ"]
    df_optim = df[
        df["Вид Отправки"].isin(cfg.OPTIMIZABLE_FLOW_TYPES)
        & df["Мода"].isin(all_optim_modes)
    ]
    df_fixed = df[is_fixed_type | (is_fixed_mode & ~is_fixed_type)]

    sw_df = df_optim[df_optim["Вид Отправки"] == "Поставщик - Склад"]
    wc_df = df_optim[df_optim["Вид Отправки"] == "Склад - Клиент"]
    ww_df = df_optim[df_optim["Вид Отправки"] == "Склад - Склад"]

    as_is_sw = sw_df["Затраты на транспортировку (LTL), в рублях"].sum()
    as_is_wc = wc_df["Затраты на транспортировку (LTL), в рублях"].sum()
    as_is_ww = ww_df["Затраты на транспортировку (LTL), в рублях"].sum()
    fixed_transport = df_fixed["Затраты на транспортировку (LTL), в рублях"].sum()

    # Складские затраты
    as_is_transport = df["Затраты на транспортировку (LTL), в рублях"].sum()
    out_ = df[df["Склад Отправления"].notna()].groupby("Склад Отправления")["Количество"].sum()
    in_ = df[df["Склад Назначения"].notna()].groupby("Склад Назначения")["Количество"].sum()
    tp = pd.DataFrame({"o": out_, "i": in_}).fillna(0)
    tp["t"] = tp["o"] + tp["i"]
    as_is_var_wh = sum(wh_var.get(w, 0) * row["t"] for w, row in tp.iterrows())
    as_is_fix_wh = sum(v * sum(days_in_month.values()) for v in wh_fix.values())

    # Фиксированные переменные складские
    fixed_var_wh = 0.0
    for _, r in df_fixed.iterrows():
        vol = r["Количество"]
        wd, ws = r["Склад Назначения"], r["Склад Отправления"]
        if pd.notna(wd) and wd in wh_set:
            fixed_var_wh += wh_var.get(wd, 0) * vol
        if pd.notna(ws) and ws in wh_set:
            fixed_var_wh += wh_var.get(ws, 0) * vol

    total = as_is_transport + as_is_var_wh + as_is_fix_wh

    # Фактические потоки — единая таблица (Отправитель/Получатель)
    all_df = pd.concat([df_optim, df_fixed]).copy()
    for col in ["Склад Отправления", "Склад Назначения", "Город Отправления",
                 "Город Назначения", "Мода", "Группа Номенклатуры", "Вид Доставки"]:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna("")
    grp_cols = ["Вид Отправки", "Вид Доставки", "Склад Отправления", "Город Отправления",
                 "Склад Назначения", "Город Назначения", "Мода", "Группа Номенклатуры", "month_idx"]
    flows = []
    for _, r in all_df.groupby(grp_cols).agg({
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
            "Вид Отправки": vt,
            "Вид Доставки": vid_dostavki,
            "Группа": r["Группа Номенклатуры"], "Месяц": r["month_idx"],
            "Мода": moda,
            "Объём": vol,
            "Исх. склад. обработка, руб/т": wh_out_rate,
            "Исх. склад. обработка, руб": wh_out_rate * vol,
            "Транспорт, руб/т": transport_rate,
            "Транспорт, руб": transport_cost,
            "Вх. склад. обработка, руб/т": wh_in_rate,
            "Вх. склад. обработка, руб": wh_in_rate * vol,
            "Оптимизировано": False,
        })
    flows_df = pd.DataFrame(flows) if flows else pd.DataFrame()

    costs = {
        "total": total,
        "transport_sw": as_is_sw,
        "transport_wc": as_is_wc,
        "transport_ww": as_is_ww,
        "transport_fixed": fixed_transport,
        "var_wh_opt": as_is_var_wh - fixed_var_wh,
        "var_wh_fixed": fixed_var_wh,
        "fix_wh": as_is_fix_wh,
    }
    return costs, flows_df


def format_summary_table(results_list):
    """Формирует сводную таблицу для списка экспериментов."""
    categories = [
        ("Входящий транспорт (Пост->Склад)", "transport_sw"),
        ("Исходящий транспорт (Склад->Клиент)", "transport_wc"),
        ("Внутр. перемещения (Склад->Склад)", "transport_ww"),
        ("Фикс. транспорт (ММ+перераб.)", "transport_fixed"),
        ("Перем. складские (оптимизируемые)", "var_wh_opt"),
        ("Перем. складские (фиксированные)", "var_wh_fixed"),
        ("Постоянные складские", "fix_wh"),
    ]

    rows = []
    for cat_name, key in categories:
        row = {"Статья затрат": cat_name}
        for exp in results_list:
            row[exp["name"]] = exp["data"].get(key, 0)
        rows.append(row)

    # Итого
    row_total = {"Статья затрат": "ИТОГО"}
    for exp in results_list:
        penalty = exp["data"].get("penalty", 0)
        row_total[exp["name"]] = exp["data"]["total"] - penalty
    rows.append(row_total)

    return pd.DataFrame(rows)


def experiment_to_excel(exp):
    """Экспортирует один эксперимент в Excel."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Сводка
        categories = [
            ("Входящий транспорт (Пост->Склад)", "transport_sw"),
            ("Исходящий транспорт (Склад->Клиент)", "transport_wc"),
            ("Внутр. перемещения (Склад->Склад)", "transport_ww"),
            ("Фикс. транспорт (ММ+перераб.)", "transport_fixed"),
            ("Перем. складские (оптимизируемые)", "var_wh_opt"),
            ("Перем. складские (фиксированные)", "var_wh_fixed"),
            ("Постоянные складские", "fix_wh"),
        ]
        rows = []
        for cat_name, key in categories:
            rows.append({"Статья затрат": cat_name, "Сумма, руб": exp["data"].get(key, 0)})
        penalty = exp["data"].get("penalty", 0)
        rows.append({"Статья затрат": "ИТОГО", "Сумма, руб": exp["data"]["total"] - penalty})
        pd.DataFrame(rows).to_excel(writer, sheet_name="Сводка", index=False)

        # Параметры
        params_rows = [{"Параметр": k, "Значение": v} for k, v in exp.get("params_dict", {}).items()]
        if params_rows:
            pd.DataFrame(params_rows).to_excel(writer, sheet_name="Параметры", index=False)

        # Потоки
        flows = exp.get("flows")
        if isinstance(flows, pd.DataFrame) and not flows.empty:
            try:
                flows.to_excel(writer, sheet_name="Потоки", index=False)
            except Exception as e:
                pd.DataFrame([{"Ошибка": str(e)}]).to_excel(
                    writer, sheet_name="Потоки_ошибка", index=False)

        # Спрос
        demand = exp.get("demand")
        if isinstance(demand, pd.DataFrame) and not demand.empty:
            try:
                demand.to_excel(writer, sheet_name="Спрос", index=False)
            except Exception as e:
                pd.DataFrame([{"Ошибка": str(e)}]).to_excel(
                    writer, sheet_name="Спрос_ошибка", index=False)

    return buf.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Оптимизатор логистической сети (Polars)",
    page_icon="🏭",
    layout="wide",
)

st.title("Оптимизатор логистической сети (Polars)")
st.caption("Аналог AnyLogistix Network Optimization на Google OR-Tools")

# Инициализация состояния
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "exp_counter" not in st.session_state:
    st.session_state.exp_counter = 0
if "selected_exp" not in st.session_state:
    st.session_state.selected_exp = None

# ── Сайдбар ──
with st.sidebar:
    st.header("1. Загрузка данных")

    uploaded = st.file_uploader(
        "Файл с данными (.xlsb / .xlsx / .parquet)",
        type=["xlsb", "xlsx", "parquet"],
        help="Excel: один файл с листами Отгрузки, Склады компании, Мощность складов, Остатки. "
             "Parquet: папка из 4 файлов (загрузите любой, остальные подтянутся автоматически)."
    )

    if uploaded:
        # Загрузка сырых данных (при смене файла)
        if st.session_state.get("_file_name") != uploaded.name:
            st.session_state.data_loaded = False
            st.session_state.pop("_needs_mapping", None)
            st.session_state.pop("_raw_sheets", None)
            try:
                with st.status("Загрузка данных...", expanded=True) as load_status:
                    t_load = time.time()
                    df, wh_costs, wh_cap, inv_raw = load_from_upload(uploaded)
                    t_loaded = time.time()
                    fmt = "parquet" if uploaded.name.endswith(".parquet") else "Excel"
                    st.write(f"Чтение {fmt} — **{t_loaded - t_load:.1f}с**")
                    load_status.update(label=f"Данные загружены за {t_loaded - t_load:.1f}с",
                                       state="complete", expanded=False)

                if inv_raw is None or len(inv_raw) == 0:
                    st.error("Не найден лист 'Остатки' в файле.")
                    st.stop()

                sheets = {
                    "Отгрузки": df,
                    "Склады компании": wh_costs,
                    "Мощность складов": wh_cap,
                    "Остатки": inv_raw,
                }

                sheets, rename_warnings = try_auto_rename(sheets)
                if rename_warnings:
                    st.info("Автоисправление колонок:\n- " + "\n- ".join(rename_warnings))

                missing = get_missing_columns(sheets)
                if missing:
                    st.session_state._raw_sheets = sheets
                    st.session_state._needs_mapping = True
                    st.session_state._file_name = uploaded.name
                else:
                    st.session_state._needs_mapping = False
                    st.session_state._file_name = uploaded.name
                    cfg_temp = make_config()
                    df = sheets["Отгрузки"]
                    wh_costs = sheets["Склады компании"]
                    inv_raw = sheets["Остатки"]
                    df, wh_costs, inv_raw = preprocess(df, wh_costs, inv_raw, cfg_temp)
                    st.session_state.df = df
                    st.session_state.wh_costs_orig = wh_costs.copy()
                    st.session_state.wh_cap = sheets["Мощность складов"]
                    st.session_state.inv_raw = inv_raw
                    st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")

        # Маппинг колонок (если нужен)
        if st.session_state.get("_needs_mapping") and "_raw_sheets" in st.session_state:
            sheets = st.session_state._raw_sheets
            missing = get_missing_columns(sheets)
            st.warning("Укажите соответствие колонок:")

            user_mapping = {}
            for sheet_name, cols_info in missing.items():
                st.markdown(f"**Лист '{sheet_name}'**")
                actual_cols = ["-- не выбрано --"] + list(sheets[sheet_name].columns)
                sheet_map = {}
                for req_col, similar in cols_info:
                    default_idx = 0
                    if similar:
                        try:
                            default_idx = actual_cols.index(similar[0])
                        except ValueError:
                            pass
                    chosen = st.selectbox(f"'{req_col}' =", actual_cols, index=default_idx,
                                          key=f"map_{sheet_name}_{req_col}")
                    if chosen != "-- не выбрано --":
                        sheet_map[chosen] = req_col
                if sheet_map:
                    user_mapping[sheet_name] = sheet_map

            if st.button("Применить маппинг", type="primary"):
                sheets = apply_column_mapping(sheets, user_mapping)
                still_missing = get_missing_columns(sheets)
                if still_missing:
                    for sn, cols in still_missing.items():
                        for col, _ in cols:
                            st.error(f"Лист '{sn}': колонка '{col}' не назначена")
                else:
                    cfg_temp = make_config()
                    df = sheets["Отгрузки"]
                    wh_costs = sheets["Склады компании"]
                    inv_raw = sheets["Остатки"]
                    df, wh_costs, inv_raw = preprocess(df, wh_costs, inv_raw, cfg_temp)
                    st.session_state.df = df
                    st.session_state.wh_costs_orig = wh_costs.copy()
                    st.session_state.wh_cap = sheets["Мощность складов"]
                    st.session_state.inv_raw = inv_raw
                    st.session_state.data_loaded = True
                    st.session_state._needs_mapping = False
                    st.session_state.pop("_raw_sheets", None)
                    st.rerun()

        if st.session_state.data_loaded:
            df = st.session_state.df
            wh_costs = st.session_state.wh_costs_orig
            st.success(f"{len(df):,} отгрузок, {wh_costs['Склад'].nunique()} складов")

    # ── Тип эксперимента ──
    if st.session_state.data_loaded:
        st.divider()
        st.header("2. Настройка эксперимента")

        exp_type = st.radio("Тип модели", ["AS-IS", "TO-BE"],
                            help="AS-IS — расчёт затрат по факту. TO-BE — оптимизация потоков.")

        # Параметры TO-BE (солвер)
        solver = "GLOP"
        if exp_type == "TO-BE":
            solver = st.selectbox("Солвер", ["GLOP", "SCIP"], index=0,
                                  help="GLOP — быстрый LP, SCIP — MILP")

        # Анализ чувствительности
        with st.expander("Анализ чувствительности"):
            var_markup = st.slider("Наценка на переменные складские, %",
                                   min_value=-50, max_value=100, value=0, step=5)
            fix_markup = st.slider("Наценка на постоянные складские, %",
                                   min_value=-50, max_value=100, value=0, step=5)
            transport_markup = st.slider("Наценка на транспортные затраты, %",
                                          min_value=-50, max_value=100, value=0, step=5)

        # Доп. параметры
        penalty = 20_000
        enforce_ending = True
        supply_mode = "annual"
        with st.expander("Доп. параметры"):
            penalty = st.number_input("Штраф за невыполненный спрос, руб/т",
                                       min_value=0, max_value=1_000_000, value=20_000, step=1_000)
            enforce_ending = st.checkbox("Контроль исходящего остатка", value=True)
            supply_mode = st.radio("Ограничения поставщиков", ["annual", "monthly"],
                                    format_func=lambda x: "Годовые" if x == "annual" else "Помесячные")

        # Кнопка запуска
        st.divider()
        run_experiment = st.button("Запустить эксперимент", use_container_width=True, type="primary")

    # Шаблон
    st.divider()
    template_path = os.path.join(os.path.dirname(__file__), "input_template.xlsx")
    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            st.download_button("Скачать шаблон", data=f.read(),
                               file_name="input_template.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ── Запуск эксперимента ──

if st.session_state.data_loaded and "run_experiment" in dir() and run_experiment:
    cfg = make_config(
        PENALTY_UNMET_DEMAND=penalty,
        SOLVER_NAME=solver,
        ENFORCE_ENDING_INVENTORY=enforce_ending,
        SUPPLY_CONSTRAINT_MODE=supply_mode,
    )
    wh_costs_mod = apply_cost_markup(st.session_state.wh_costs_orig, var_markup, fix_markup)

    # Наценка на транспорт — применяем к данным
    df_run = st.session_state.df.copy()
    if transport_markup != 0:
        df_run["Затраты на транспортировку (LTL), в рублях"] = (
            df_run["Затраты на транспортировку (LTL), в рублях"] * (1 + transport_markup / 100)
        )

    # Авто-нумерация
    st.session_state.exp_counter += 1
    n = st.session_state.exp_counter
    auto_name = f"{exp_type}_Эксперимент {n}"

    # Параметры для отчёта
    params_dict = {"Тип": exp_type, "Солвер": solver}
    if var_markup != 0:
        params_dict["Наценка перем. склад."] = f"{var_markup:+d}%"
    if fix_markup != 0:
        params_dict["Наценка пост. склад."] = f"{fix_markup:+d}%"
    if transport_markup != 0:
        params_dict["Наценка транспорт"] = f"{transport_markup:+d}%"
    params_dict["Штраф невып. спроса"] = f"{penalty:,} руб/т"
    params_dict["Контроль исх. остатка"] = "Да" if enforce_ending else "Нет"
    params_dict["Ограничения поставщ."] = "Годовые" if supply_mode == "annual" else "Помесячные"

    if exp_type == "AS-IS":
        with st.status("Расчёт AS-IS...", expanded=True) as status:
            t0 = time.time()

            status.update(label="[1/2] Расчёт затрат и формирование потоков...")
            result, flows_df = calc_as_is(df_run, wh_costs_mod, st.session_state.wh_cap,
                                          st.session_state.inv_raw, cfg)
            t1 = time.time()
            n_flows = len(flows_df) if isinstance(flows_df, pd.DataFrame) else 0
            st.write(f"[1/2] Расчёт затрат и потоков — **{t1-t0:.1f}с** ({n_flows:,} потоков)")

            status.update(label="[2/2] Сохранение результатов...")
            experiment = {
                "name": auto_name,
                "type": "AS-IS",
                "data": result,
                "params_dict": params_dict,
                "flows": flows_df,
            }
            st.session_state.experiments.append(experiment)
            st.session_state.selected_exp = len(st.session_state.experiments) - 1
            t2 = time.time()
            st.write(f"[2/2] Сохранение — **{t2-t1:.1f}с**")

            status.update(label=f"AS-IS завершён за {t2-t0:.1f}с", state="complete", expanded=False)

    else:  # TO-BE
        with st.status("Оптимизация TO-BE...", expanded=True) as status:
            t0 = time.time()

            status.update(label="[1/4] Подготовка данных модели...")
            data = prepare_model_data(df_run, wh_costs_mod, st.session_state.wh_cap,
                                       st.session_state.inv_raw, cfg)
            t1 = time.time()
            st.write(f"[1/4] Подготовка данных — **{t1-t0:.1f}с**")

            status.update(label="[2/4] Решение оптимизационной модели...")
            results = solve(data, cfg)
            t2 = time.time()
            solve_time = results.get("solve_time", 0) if results else 0
            st.write(f"[2/4] Решение модели — **{t2-t1:.1f}с** (солвер {solve_time:.0f}с)")

            if results is None:
                status.update(label="Ошибка оптимизации", state="error")
                st.error("Оптимизация не удалась. Попробуйте другие параметры.")
            else:
                status.update(label="[3/4] Извлечение результатов...")
                tb = results["to_be"]
                flows_df = results.get("flows")
                n_flows = len(flows_df) if isinstance(flows_df, pd.DataFrame) else 0
                t3 = time.time()
                st.write(f"[3/4] Извлечение результатов — **{t3-t2:.1f}с** ({n_flows:,} потоков)")

                status.update(label="[4/4] Сохранение результатов...")
                params_dict["Время решения"] = f"{solve_time:.0f}с"
                params_dict["Невып. спрос"] = f"{tb.get('unmet_demand', 0):,.0f} т"
                experiment = {
                    "name": auto_name,
                    "type": "TO-BE",
                    "data": tb,
                    "params_dict": params_dict,
                    "flows": flows_df,
                    "demand": results.get("demand"),
                }
                st.session_state.experiments.append(experiment)
                st.session_state.selected_exp = len(st.session_state.experiments) - 1
                t4 = time.time()
                st.write(f"[4/4] Сохранение — **{t4-t3:.1f}с**")

                status.update(label=f"TO-BE завершён за {t4-t0:.1f}с (солвер {solve_time:.0f}с)",
                              state="complete", expanded=False)


# ── Просмотр и редактирование данных ──

if st.session_state.data_loaded:
    with st.expander("Загруженные данные", expanded=False):
        tab_ship, tab_wh, tab_cap, tab_inv = st.tabs([
            "Отгрузки", "Склады компании", "Мощность складов", "Остатки"
        ])

        with tab_ship:
            ship_df = st.session_state.df
            c1, c2, c3 = st.columns(3)
            c1.metric("Отгрузок", f"{len(ship_df):,}")
            c2.metric("Объём, тонн", f"{ship_df['Количество'].sum():,.0f}")
            c3.metric("Транспорт, млн руб",
                      f"{ship_df['Затраты на транспортировку (LTL), в рублях'].sum()/1e6:,.0f}")

            with st.expander("Разбивка по видам отправки"):
                st.dataframe(
                    ship_df.groupby("Вид Отправки").agg(
                        Строк=("Количество", "count"),
                        Объём_тонн=("Количество", "sum"),
                        Затраты_руб=("Затраты на транспортировку (LTL), в рублях", "sum"),
                    ).reset_index(),
                    use_container_width=True, hide_index=True,
                )

            PREVIEW_ROWS = 1000
            st.caption(f"Превью: первые {PREVIEW_ROWS:,} строк")
            st.dataframe(ship_df.head(PREVIEW_ROWS), use_container_width=True, height=300)

        with tab_wh:
            edited_wh = st.data_editor(st.session_state.wh_costs_orig,
                                        use_container_width=True, num_rows="dynamic", key="wh_editor")
            if not edited_wh.equals(st.session_state.wh_costs_orig):
                st.session_state.wh_costs_orig = edited_wh

        with tab_cap:
            edited_cap = st.data_editor(st.session_state.wh_cap,
                                         use_container_width=True, num_rows="dynamic", key="cap_editor")
            if not edited_cap.equals(st.session_state.wh_cap):
                st.session_state.wh_cap = edited_cap

        with tab_inv:
            edited_inv = st.data_editor(st.session_state.inv_raw,
                                         use_container_width=True, num_rows="dynamic", key="inv_editor")
            if not edited_inv.equals(st.session_state.inv_raw):
                st.session_state.inv_raw = edited_inv


# ── Результаты экспериментов ──

if st.session_state.experiments:
    st.header("Эксперименты")

    cols_list = st.columns([4, 1])
    with cols_list[1]:
        if st.button("Очистить все", type="secondary"):
            st.session_state.experiments = []
            st.session_state.selected_exp = None
            st.session_state.exp_counter = 0
            st.rerun()

    # Выбор эксперимента через selectbox (мгновенное переключение)
    exp_names = [f"{exp['name']} — {(exp['data']['total'] - exp['data'].get('penalty', 0))/1e6:,.0f} млн руб"
                 for exp in st.session_state.experiments]

    default_idx = st.session_state.selected_exp if st.session_state.selected_exp is not None else len(exp_names) - 1
    if default_idx >= len(exp_names):
        default_idx = len(exp_names) - 1

    selected = st.selectbox("Выберите эксперимент", exp_names, index=default_idx,
                             label_visibility="collapsed")
    sel = exp_names.index(selected)
    st.session_state.selected_exp = sel

    # Детали выбранного
    exp = st.session_state.experiments[sel]
    st.divider()

    # Заголовок + удалить
    col_title, col_del = st.columns([6, 1])
    col_title.subheader(exp["name"])
    with col_del:
        if st.button("Удалить", key="del_selected"):
            st.session_state.experiments.pop(sel)
            st.session_state.selected_exp = max(0, sel - 1) if st.session_state.experiments else None
            st.rerun()

    # Параметры
    params = exp.get("params_dict", {})
    if params:
        param_cols = st.columns(min(len(params), 4))
        for j, (k, v) in enumerate(params.items()):
            param_cols[j % len(param_cols)].metric(k, v)

    st.markdown("---")

    # Сводка затрат
    categories = [
        ("Входящий транспорт (Пост->Склад)", "transport_sw"),
        ("Исходящий транспорт (Склад->Клиент)", "transport_wc"),
        ("Внутр. перемещения (Склад->Склад)", "transport_ww"),
        ("Фикс. транспорт (ММ+перераб.)", "transport_fixed"),
        ("Перем. складские (оптимизируемые)", "var_wh_opt"),
        ("Перем. складские (фиксированные)", "var_wh_fixed"),
        ("Постоянные складские", "fix_wh"),
    ]
    rows = []
    for cat_name, key in categories:
        val = exp["data"].get(key, 0)
        rows.append({"Статья затрат": cat_name, "Сумма, млн руб": f"{val/1e6:,.1f}"})
    pen = exp["data"].get("penalty", 0)
    total = exp["data"]["total"] - pen
    rows.append({"Статья затрат": "ИТОГО", "Сумма, млн руб": f"{total/1e6:,.1f}"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Потоки
    flows_data = exp.get("flows")
    if isinstance(flows_data, pd.DataFrame) and not flows_data.empty:
        with st.expander(f"Потоки ({len(flows_data):,} строк)", expanded=False):
            st.dataframe(flows_data.head(1000), use_container_width=True, height=300)
    else:
        st.warning("Потоки не найдены. Попробуйте перезапустить Streamlit и прогнать эксперимент заново.")

    # Скачивание
    excel_data = experiment_to_excel(exp)
    st.download_button(
        f"Скачать {exp['name']} (Excel)",
        data=excel_data,
        file_name=f"{exp['name']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

elif st.session_state.data_loaded:
    st.info("Выберите тип модели и параметры в боковой панели, затем нажмите 'Запустить эксперимент'")
elif not uploaded:
    st.info("Загрузите файл с данными в боковой панели")
