"""
Юнит-тесты для модуля tariff_calculator.

Запуск: pytest tests/test_tariff_calculator.py -v
"""

import sys
import os
import pytest
import polars as pl
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logistics_optimizer_polars.tariff_calculator import (
    build_tariff_reference,
    find_analog_route,
    assign_tariffs,
    load_neighbors,
    load_city_region,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_shipments() -> pl.DataFrame:
    """Минимальный DataFrame отгрузок для тестов."""
    return pl.DataFrame({
        "Город Отправления":                         ["Москва", "Москва", "СПб", "Казань"],
        "Склад Назначения":                          ["Новосибирск", "Новосибирск", "Екатеринбург", "Самара"],
        "Мода":                                      ["Авто", "Авто", "Авто", "Авто"],
        "Затраты на транспортировку (LTL), в рублях": [100_000.0, 80_000.0, 60_000.0, 40_000.0],
        "Количество":                                [10.0, 8.0, 6.0, 4.0],
        "Группа Номенклатуры":                       ["Металл", "Металл", "Металл", "Прочее"],
    })


@pytest.fixture
def tariff_ref(sample_shipments) -> dict:
    return build_tariff_reference(sample_shipments)


@pytest.fixture
def neighbors_sorted() -> dict:
    return {
        "Московская область": ["Тверская область", "Тульская область"],
        "Тверская область":   ["Московская область", "Новгородская область"],
        "Новосибирская область": ["Омская область"],
        "Омская область":        ["Новосибирская область"],
    }


@pytest.fixture
def city_to_region() -> dict:
    return {
        "Москва":       "Московская область",
        "Новосибирск":  "Новосибирская область",
        "СПб":          "Ленинградская область",
        "Екатеринбург": "Свердловская область",
        "Казань":       "Татарстан",
        "Самара":       "Самарская область",
        "Тверь":        "Тверская область",
        "Омск":         "Омская область",
    }


# ---------------------------------------------------------------------------
# Тесты build_tariff_reference
# ---------------------------------------------------------------------------

def test_build_tariff_reference_keys(tariff_ref):
    """Проверяем наличие всех ключей в справочнике."""
    expected_keys = {"by_pg", "by_route", "by_orig_city", "by_dest_city",
                     "by_pg_global", "global_avg", "volumes_by_route"}
    assert expected_keys.issubset(set(tariff_ref.keys()))


def test_build_tariff_reference_weighted(tariff_ref):
    """Средневзвешенный тариф Москва -> Новосибирск / Металл = (100k+80k)/(10+8) = 10000."""
    key = ("Москва", "Новосибирск", "Металл")
    assert key in tariff_ref["by_pg"]
    assert abs(tariff_ref["by_pg"][key] - 10_000.0) < 1e-6


def test_build_tariff_reference_global(tariff_ref):
    """Глобальный средний: (100k+80k+60k+40k) / (10+8+6+4) = 280000/28 = 10000."""
    assert abs(tariff_ref["global_avg"] - 10_000.0) < 1e-6


# ---------------------------------------------------------------------------
# Тесты find_analog_route
# ---------------------------------------------------------------------------

def test_find_analog_direct_region(neighbors_sorted):
    """Если маршрут между теми же регионами есть — step=0."""
    routes = {("Московская область", "Московская область")}
    mo, md, step = find_analog_route(
        "Московская область", "Московская область", routes, neighbors_sorted
    )
    assert step == 0
    assert mo == "Московская область"


def test_find_analog_step1(neighbors_sorted):
    """Фиксируем отправление, заменяем назначение на ближайшего соседа."""
    routes = {("Московская область", "Тверская область")}
    mo, md, step = find_analog_route(
        "Московская область", "Новосибирская область", routes, neighbors_sorted
    )
    assert step == 1
    assert mo == "Московская область"
    assert md == "Тверская область"


def test_find_analog_not_found(neighbors_sorted):
    """Нет аналогов — возвращаем (None, None, None)."""
    routes = set()  # пустой справочник
    mo, md, step = find_analog_route(
        "Московская область", "Новосибирская область", routes, neighbors_sorted
    )
    assert mo is None and md is None and step is None


# ---------------------------------------------------------------------------
# Тесты assign_tariffs
# ---------------------------------------------------------------------------

@pytest.fixture
def flows_df_sample() -> pd.DataFrame:
    """Минимальный flows_df после оптимизатора."""
    return pd.DataFrame({
        "Отправитель":         ["Москва",       "Москва",   "Тверь"],
        "Получатель":          ["Новосибирск",  "Самара",   "Омск"],
        "Группа Номенклатуры": ["Металл",       "Прочее",   "Металл"],
        "Мода":                ["Авто",          "Авто",    "Авто"],
        "Оптимизировано":      [True,             True,      True],
    })


def test_assign_direct_match(tariff_ref, neighbors_sorted, city_to_region, flows_df_sample):
    """Строка 0: Москва->Новосибирск/Металл — прямое совпадение."""
    result = assign_tariffs(flows_df_sample, tariff_ref, neighbors_sorted, city_to_region)
    assert result.at[0, "tariff_source"] == "direct"
    assert abs(result.at[0, "Тариф авто, руб/т"] - 10_000.0) < 1e-6


def test_assign_fallback_3d(tariff_ref, neighbors_sorted, city_to_region):
    """Маршрут без аналогов и без региональной статистики — fallback 3d."""
    flows = pd.DataFrame({
        "Отправитель":         ["НеизвестныйГород"],
        "Получатель":          ["ДругойГород"],
        "Группа Номенклатуры": ["НеизвестнаяГруппа"],
        "Мода":                ["Авто"],
        "Оптимизировано":      [True],
    })
    result = assign_tariffs(flows, tariff_ref, neighbors_sorted, city_to_region)
    assert result.at[0, "tariff_source"] == "fallback_3d"
    assert abs(result.at[0, "Тариф авто, руб/т"] - tariff_ref["global_avg"]) < 1e-6


def test_not_applicable_for_non_auto(tariff_ref, neighbors_sorted, city_to_region):
    """Строки с Мода != Авто получают not_applicable."""
    flows = pd.DataFrame({
        "Отправитель":         ["Москва"],
        "Получатель":          ["Новосибирск"],
        "Группа Номенклатуры": ["Металл"],
        "Мода":                ["ЖД"],
        "Оптимизировано":      [True],
    })
    result = assign_tariffs(flows, tariff_ref, neighbors_sorted, city_to_region)
    assert result.at[0, "tariff_source"] == "not_applicable"
