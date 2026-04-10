"""
Модуль расчёта тарифов авто для новых маршрутов оптимизатора.

Используется после вызова solve() для присвоения тарифов маршрутам,
которых не было в исходных исторических данных.
"""

from __future__ import annotations

import polars as pl
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Построение справочников
# ---------------------------------------------------------------------------

def build_tariff_reference(df_shipments: pl.DataFrame) -> dict:
    """
    Строит справочник тарифов из исходного DataFrame отгрузок.

    Параметры
    ----------
    df_shipments : pl.DataFrame
        DataFrame листа «Отгрузки», содержащий колонки:
        «Город Отправления», «Склад Назначения» / «Город Назначения»,
        «Мода», «Затраты на транспортировку (LTL), в рублях», «Количество»,
        «Группа Номенклатуры».

    Возвращает
    ----------
    dict со следующими ключами:
      'by_pg'          : dict[(orig, dest, group), float]  — тариф по (откуда, куда, группа)
      'by_route'       : dict[(orig, dest), float]         — тариф по (откуда, куда)
      'by_orig_region' : dict[(orig, group), float]        — средний тариф по региону отправления + группе
      'by_dest_region' : dict[(dest, group), float]        — средний тариф по региону назначения + группе
      'by_pg_global'   : dict[str, float]                  — средний тариф по группе номенклатуры
      'global_avg'     : float                             — глобальный средний тариф авто
      'volumes_by_route': dict[(orig, dest), float]        — суммарный объём по маршруту (для взвешивания)
    """
    # Определяем колонку «назначение» — может называться «Склад Назначения» или «Город Назначения»
    dest_col = "Склад Назначения" if "Склад Назначения" in df_shipments.columns else "Город Назначения"

    # Фильтр: только Авто, ненулевое количество
    df = (
        df_shipments
        .filter(
            (pl.col("Мода") == "Авто")
            & (pl.col("Количество") > 0)
        )
        .rename({dest_col: "_dest"})
        .select([
            pl.col("Город Отправления").alias("orig"),
            pl.col("_dest").alias("dest"),
            pl.col("Группа Номенклатуры").alias("group"),
            pl.col("Затраты на транспортировку (LTL), в рублях").alias("cost"),
            pl.col("Количество").alias("qty"),
        ])
    )

    def _weighted_tariff(df_agg: pl.DataFrame, group_cols: list[str]) -> dict:
        """Вспомогательная: агрегация и конвертация в словарь."""
        agg = (
            df_agg
            .group_by(group_cols)
            .agg([
                pl.sum("cost").alias("sum_cost"),
                pl.sum("qty").alias("sum_qty"),
            ])
            .filter(pl.col("sum_qty") > 0)
            .with_columns(
                (pl.col("sum_cost") / pl.col("sum_qty")).alias("tariff")
            )
        )
        result = {}
        for row in agg.iter_rows(named=True):
            key = tuple(row[c] for c in group_cols)
            if len(group_cols) == 1:
                key = key[0]
            result[key] = row["tariff"]
        return result

    by_pg    = _weighted_tariff(df, ["orig", "dest", "group"])
    by_route = _weighted_tariff(df, ["orig", "dest"])

    # Объёмы по маршруту (для взвешенного поиска аналогов)
    volumes_by_route: dict[tuple[str, str], float] = {}
    for (orig, dest), tariff in by_route.items():
        # уже есть в by_route, но нужны объёмы — перестроим
        pass
    vol_agg = (
        df.group_by(["orig", "dest"])
        .agg(pl.sum("qty").alias("sum_qty"))
    )
    for row in vol_agg.iter_rows(named=True):
        volumes_by_route[(row["orig"], row["dest"])] = row["sum_qty"]

    # Средние по региону отправления / назначения (нужны city_to_region — передаются позже)
    # Здесь строим агрегаты по городам, а регион будет подставляться в assign_tariffs
    by_orig_city  = _weighted_tariff(df, ["orig", "group"])
    by_dest_city  = _weighted_tariff(df, ["dest", "group"])
    by_pg_global  = _weighted_tariff(df, ["group"])

    # Глобальный средний тариф
    total_cost = df.select(pl.sum("cost")).item()
    total_qty  = df.select(pl.sum("qty")).item()
    global_avg = total_cost / total_qty if total_qty > 0 else 0.0

    print(f"[Тарифы] Справочник построен: {len(by_pg)} записей (orig+dest+group), "
          f"{len(by_route)} маршрутов, {len(by_pg_global)} групп номенклатуры.")

    return {
        "by_pg":           by_pg,
        "by_route":        by_route,
        "by_orig_city":    by_orig_city,
        "by_dest_city":    by_dest_city,
        "by_pg_global":    by_pg_global,
        "global_avg":      global_avg,
        "volumes_by_route": volumes_by_route,
    }


def load_neighbors(neighbors_path: str) -> dict[str, list[str]]:
    """
    Загружает neighbors.csv и возвращает отсортированный словарь:
    region -> [neighbor1, neighbor2, ...] в порядке возрастания order_num.

    Параметры
    ----------
    neighbors_path : str
        Путь к CSV-файлу с колонками: region, neighbor_region, order_num.
    """
    df = pl.read_csv(neighbors_path)
    result: dict[str, list[str]] = {}
    for row in (
        df
        .sort("order_num")
        .iter_rows(named=True)
    ):
        region = row["region"]
        neighbor = row["neighbor_region"]
        result.setdefault(region, []).append(neighbor)
    return result


def load_city_region(city_region_path: str) -> dict[str, str]:
    """
    Загружает city_region.csv и возвращает словарь city -> region.

    Параметры
    ----------
    city_region_path : str
        Путь к CSV-файлу с колонками: city, region.
    """
    df = pl.read_csv(city_region_path)
    return {row["city"]: row["region"] for row in df.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# Поиск аналогичного маршрута по соседним регионам
# ---------------------------------------------------------------------------

def find_analog_route(
    A: str,
    B: str,
    tariff_routes: set,
    neighbors_sorted: dict[str, list[str]],
) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Ищет аналогичный маршрут для пары регионов (A, B) по справочнику соседних регионов.

    Параметры
    ----------
    A : str
        Регион отправления.
    B : str
        Регион назначения.
    tariff_routes : set
        Множество доступных пар регионов (orig_region, dest_region) в справочнике тарифов.
    neighbors_sorted : dict[str, list[str]]
        Словарь region -> [neighbor1, ...] в порядке близости.

    Возвращает
    ----------
    (match_origin, match_dest, priority_step) или (None, None, None).
    priority_step:
      0 — внутрирегиональный (A == B)
      1 — A -> X (заменён регион назначения)
      2 — Y -> B (заменён регион отправления)
      3 — Y -> X (заменены оба региона)
    """
    # Шаг 0: внутрирегиональный
    if A == B and (A, A) in tariff_routes:
        return A, A, 0

    # Шаг 1: фиксируем A, перебираем соседей B
    for nb in neighbors_sorted.get(B, []):
        if (A, nb) in tariff_routes:
            return A, nb, 1

    # Шаг 2: фиксируем B, перебираем соседей A
    for na in neighbors_sorted.get(A, []):
        if (na, B) in tariff_routes:
            return na, B, 2

    # Шаг 3: перебираем соседей обоих
    for na in neighbors_sorted.get(A, []):
        for nb in neighbors_sorted.get(B, []):
            if (na, nb) in tariff_routes:
                return na, nb, 3

    return None, None, None


# ---------------------------------------------------------------------------
# Основная функция присвоения тарифов
# ---------------------------------------------------------------------------

def assign_tariffs(
    flows_df: pd.DataFrame,
    tariff_ref: dict,
    neighbors_sorted: dict[str, list[str]],
    city_to_region: dict[str, str],
) -> pd.DataFrame:
    """
    Присваивает тарифы новым авто-маршрутам в flows_df.

    Обрабатывает только строки где Мода == "Авто" и Оптимизировано == True.
    Остальные строки получают tariff_source = "not_applicable".

    Параметры
    ----------
    flows_df : pd.DataFrame
        DataFrame результатов оптимизации (лист «Потоки»).
    tariff_ref : dict
        Справочник тарифов, возвращённый build_tariff_reference().
    neighbors_sorted : dict[str, list[str]]
        Словарь соседних регионов (из load_neighbors).
    city_to_region : dict[str, str]
        Словарь город -> регион (из load_city_region).

    Возвращает
    ----------
    pd.DataFrame с добавленными колонками:
      - «Тариф авто, руб/т»  — рассчитанный тариф
      - «tariff_source»      — источник (direct / analog_step_N / fallback_3a..3d / not_found)
    """
    flows_df = flows_df.copy()
    flows_df["Тариф авто, руб/т"] = float("nan")
    flows_df["tariff_source"] = "not_applicable"

    by_pg           = tariff_ref["by_pg"]
    by_route        = tariff_ref["by_route"]
    by_orig_city    = tariff_ref.get("by_orig_city", {})
    by_dest_city    = tariff_ref.get("by_dest_city", {})
    by_pg_global    = tariff_ref["by_pg_global"]
    global_avg      = tariff_ref["global_avg"]

    # Маска целевых строк
    mask = (flows_df["Мода"] == "Авто") & (flows_df["Оптимизировано"] == True)  # noqa: E712
    target_idx = flows_df[mask].index.tolist()

    if not target_idx:
        print("[Тарифы] Нет строк Авто + Оптимизировано для обработки.")
        return flows_df

    # --- Строим множество регионов для find_analog_route ---
    # Ключи by_pg: (orig_city, dest_city, group) — нужно собрать (orig_region, dest_region)
    tariff_region_routes: set[tuple[str, str]] = set()
    for (orig_city, dest_city, _group) in by_pg.keys():
        orig_r = city_to_region.get(orig_city)
        dest_r = city_to_region.get(dest_city)
        if orig_r and dest_r:
            tariff_region_routes.add((orig_r, dest_r))

    # --- Вспомогательная: взвешенный тариф по аналоговому маршруту ---
    def _tariff_for_analog(
        match_orig_region: str,
        match_dest_region: str,
        group: str,
    ) -> Optional[float]:
        """
        Ищет средневзвешенный тариф по всем городам, чьи регионы == match_orig/dest.
        Взвешивание — по объёму.
        """
        total_cost = 0.0
        total_qty  = 0.0
        # Идём по by_pg: ключ (orig_city, dest_city, group)
        for (oc, dc, g), tariff in by_pg.items():
            if (
                g == group
                and city_to_region.get(oc) == match_orig_region
                and city_to_region.get(dc) == match_dest_region
            ):
                vol = tariff_ref["volumes_by_route"].get((oc, dc), 1.0)
                total_cost += tariff * vol
                total_qty  += vol
        if total_qty > 0:
            return total_cost / total_qty
        # Если по группе не нашли — без группы
        for (oc, dc), tariff in by_route.items():
            if (
                city_to_region.get(oc) == match_orig_region
                and city_to_region.get(dc) == match_dest_region
            ):
                vol = tariff_ref["volumes_by_route"].get((oc, dc), 1.0)
                total_cost += tariff * vol
                total_qty  += vol
        return total_cost / total_qty if total_qty > 0 else None

    # --- Счётчики статистики ---
    cnt_direct = 0
    cnt_analog: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    cnt_fallback: dict[str, int] = {"3a": 0, "3b": 0, "3c": 0, "3d": 0}
    cnt_not_found = 0

    for idx in target_idx:
        row   = flows_df.loc[idx]
        orig  = row["Отправитель"]
        dest  = row["Получатель"]
        group = row.get("Группа Номенклатуры", row.get("Группа", ""))

        tariff: Optional[float] = None
        source: str = "not_found"

        # === Шаг 1: прямое совпадение ===
        key_pg    = (orig, dest, group)
        key_route = (orig, dest)

        if key_pg in by_pg:
            tariff = by_pg[key_pg]
            source = "direct"
        elif key_route in by_route:
            tariff = by_route[key_route]
            source = "direct"

        if tariff is not None:
            cnt_direct += 1
            flows_df.at[idx, "Тариф авто, руб/т"] = tariff
            flows_df.at[idx, "tariff_source"]      = source
            continue

        # === Шаг 2: поиск аналога по соседним регионам ===
        orig_region = city_to_region.get(orig)
        dest_region = city_to_region.get(dest)

        if orig_region and dest_region:
            match_orig, match_dest, step = find_analog_route(
                orig_region, dest_region, tariff_region_routes, neighbors_sorted
            )
            if match_orig is not None:
                tariff = _tariff_for_analog(match_orig, match_dest, group)
                if tariff is not None:
                    source = f"analog_step_{step}"
                    cnt_analog[step] = cnt_analog.get(step, 0) + 1
                    flows_df.at[idx, "Тариф авто, руб/т"] = tariff
                    flows_df.at[idx, "tariff_source"]      = source
                    continue

        # === Шаг 3: каскадный fallback ===
        # 3a: средний тариф по региону отправления (взвешенный по городам этого региона)
        if orig_region:
            total_cost_r = 0.0
            total_qty_r  = 0.0
            for (oc, g), t in by_orig_city.items():
                if city_to_region.get(oc) == orig_region:
                    vol = sum(
                        v for (o2, d2), v in tariff_ref["volumes_by_route"].items()
                        if o2 == oc
                    )
                    total_cost_r += t * max(vol, 1.0)
                    total_qty_r  += max(vol, 1.0)
            if total_qty_r > 0:
                tariff = total_cost_r / total_qty_r
                source = "fallback_3a"
                cnt_fallback["3a"] += 1
                flows_df.at[idx, "Тариф авто, руб/т"] = tariff
                flows_df.at[idx, "tariff_source"]      = source
                continue

        # 3b: средний тариф по региону назначения
        if dest_region:
            total_cost_r = 0.0
            total_qty_r  = 0.0
            for (dc, g), t in by_dest_city.items():
                if city_to_region.get(dc) == dest_region:
                    vol = sum(
                        v for (o2, d2), v in tariff_ref["volumes_by_route"].items()
                        if d2 == dc
                    )
                    total_cost_r += t * max(vol, 1.0)
                    total_qty_r  += max(vol, 1.0)
            if total_qty_r > 0:
                tariff = total_cost_r / total_qty_r
                source = "fallback_3b"
                cnt_fallback["3b"] += 1
                flows_df.at[idx, "Тариф авто, руб/т"] = tariff
                flows_df.at[idx, "tariff_source"]      = source
                continue

        # 3c: средний тариф по группе номенклатуры
        if group in by_pg_global:
            tariff = by_pg_global[group]
            source = "fallback_3c"
            cnt_fallback["3c"] += 1
            flows_df.at[idx, "Тариф авто, руб/т"] = tariff
            flows_df.at[idx, "tariff_source"]      = source
            continue

        # 3d: глобальный средний
        if global_avg > 0:
            tariff = global_avg
            source = "fallback_3d"
            cnt_fallback["3d"] += 1
            flows_df.at[idx, "Тариф авто, руб/т"] = tariff
            flows_df.at[idx, "tariff_source"]      = source
            continue

        # Не присвоено
        cnt_not_found += 1
        flows_df.at[idx, "tariff_source"] = "not_found"

    # --- Статистика ---
    analog_total = sum(cnt_analog.values())
    print(
        f"[Тарифы] Шаг 1 (прямое совпадение): {cnt_direct} маршрутов\n"
        f"[Тарифы] Шаг 2 (аналог): {analog_total} маршрутов "
        f"(step0: {cnt_analog[0]}, step1: {cnt_analog[1]}, step2: {cnt_analog[2]}, step3: {cnt_analog[3]})\n"
        f"[Тарифы] Шаг 3 (fallback): "
        f"3a: {cnt_fallback['3a']}, 3b: {cnt_fallback['3b']}, "
        f"3c: {cnt_fallback['3c']}, 3d: {cnt_fallback['3d']}\n"
        f"[Тарифы] Не присвоено: {cnt_not_found}"
    )

    return flows_df
