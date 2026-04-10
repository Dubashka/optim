"""
Проверка консистентности входных данных (Polars-версия).
"""
import polars as pl
import pandas as pd
from collections import defaultdict


def validate_data(shipments, wh_costs, wh_cap, inventory, config) -> list[str]:
    """
    Проверяет данные и возвращает список предупреждений/ошибок.
    Принимает как pandas, так и polars DataFrames.
    """
    # Конвертируем в pandas если нужно (для совместимости с app.py)
    if isinstance(shipments, pl.DataFrame):
        shipments = shipments.to_pandas()
    if isinstance(wh_costs, pl.DataFrame):
        wh_costs = wh_costs.to_pandas()
    if isinstance(wh_cap, pl.DataFrame):
        wh_cap = wh_cap.to_pandas()
    if isinstance(inventory, pl.DataFrame):
        inventory = inventory.to_pandas()

    issues = []
    wh_set = set(wh_costs["Склад"].unique())

    wh_in_shipments = set()
    for col in ["Склад Отправления", "Склад Назначения"]:
        wh_in_shipments.update(shipments[shipments[col].notna()][col].unique())

    missing_costs = wh_in_shipments - wh_set
    missing_real = {w for w in missing_costs if "перевалк" not in w.lower() and "Перевалк" not in w}
    if missing_real:
        issues.append(f"Склады есть в перемещениях, но НЕТ в затратах: {sorted(missing_real)}")

    wh_cap_set = set(wh_cap["Названия строк"].unique())
    missing_cap = wh_set - wh_cap_set
    if missing_cap:
        issues.append(f"Склады без данных о мощности (будут без ограничений): {sorted(missing_cap)}")

    ship_pgs = set(shipments["Группа Номенклатуры"].dropna().unique())
    inv_pgs = set(inventory["Продукт"].unique()) if len(inventory) > 0 else set()
    missing_pg = ship_pgs - inv_pgs
    if missing_pg and len(inv_pgs) > 0:
        issues.append(f"Группы номенклатуры в отгрузках, но не в остатках: {sorted(missing_pg)}")

    sw = shipments[shipments["Вид Отправки"] == "Поставщик - Склад"]
    wc = shipments[shipments["Вид Отправки"] == "Склад - Клиент"]
    supply_total = sw[sw["Склад Назначения"].isin(wh_set)]["Количество"].sum()
    demand_total = wc[wc["Склад Отправления"].isin(wh_set)]["Количество"].sum()
    inv_total = inventory[inventory["Склад"].isin(wh_set)]["Объем"].sum() if len(inventory) > 0 else 0

    if supply_total + inv_total < demand_total * 0.95:
        issues.append(
            f"Дефицит: поставки ({supply_total:,.0f}) + остатки ({inv_total:,.0f}) "
            f"< спрос ({demand_total:,.0f}). Будет невыполненный спрос."
        )

    for col in ["Количество", "Затраты на транспортировку (LTL), в рублях"]:
        n_na = shipments[col].isna().sum()
        if n_na > 0:
            issues.append(f"{n_na} пропущенных значений в столбце '{col}'")

    sup_cities = set(sw["Город Отправления"].unique())
    for ms in config.MANDATORY_SUPPLIERS:
        if ms not in sup_cities:
            issues.append(f"Обязательный поставщик '{ms}' не найден в данных")

    return issues


def print_validation(issues: list[str]):
    if not issues:
        print("[OK] Валидация пройдена.")
    else:
        print(f"Валидация: {len(issues)} замечаний:")
        for i in issues:
            print(f"  {i}")
