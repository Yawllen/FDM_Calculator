
# -*- coding: utf-8 -*-
"""
CLI-версия 3D калькулятора печати (FDM)
— быстрая серверная утилита без UI, 3MF/STL, точность = GUI-версии (всегда метод tetra)

Примеры:
  python cli_calculator.py model.3mf --material "Enduse PETG" --infill 20 --json
  python cli_calculator.py a.stl b.3mf --material "Proto PLA" --per-object --text --full

Ключевые гарантии:
• Каждая обработка файла начинается с ЧИСТОГО состояния парсера.
• Формулы времени/стоимости перенесены из GUI-версии и эквивалентны.
• Поддержка materials.json и pricing.json (+ точечные override'ы флагом --set).
• Параллель по файлам (--workers N) с детерминированной агрегацией.


=============================
Кратко для backend-разработчика
=============================
Назначение:
  • Утилита командной строки, отдаёт детерминированный расчёт стоимости/времени печати.
  • Вход: пути к .3mf/.stl + параметры (материал, infill, и т.д.).
  • Выход: JSON (по флагу --json) либо текстовый отчёт (по умолчанию/--text).

Интеграция с бэкендом (рекомендация):
  • Вызывать бинарь/скрипт из сервиса (например, FastAPI/Laravel) через subprocess.
  • Передавать файлы с диска (временная папка) и нужные флаги, читать stdout.
  • Коды возврата: 0 — успех; ≠0 — ошибка (валидируйте и логируйте stderr).

Стабильный JSON-контракт (--json):
  ПРИ --per-object:
  {
    "success": true,
    "count": <int>,                # число файлов во входе
    "per_object": [
      {
        "file": "<имя файла>",
        "material": "<строка>",
        "price_rub_per_g": <float>,
        "volume_model_cm3": <float>,
        "volume_print_cm3": <float>,
        "weight_g": <float>,
        "time_h": <float>,
        "costs": {
          "material": <float>,
          "energy": <float>,
          "depreciation": <float>,
          "labor": <float>,
          "reserve": <float>
        },
        "subtotal_rub": <float>,
        "min_order_rub": <float>,
        "min_applied": <bool>,
        "markup_rub": <float>,
        "platform_fee_rub": <float>,
        "vat_rub": <float>,
        "total_rub": <float>,              # до округления и применения min_policy
        "total_rounded_rub": <float>,      # финальная цена после округления
        "rounding_step_rub": <float>
      },
      ...
    ],
    "summary": null,
    "time_s": <float>               # время работы CLI, секунды
  }

  БЕЗ --per-object (сводка по всем файлам как единой сборке):
  {
    "success": true,
    "count": <int>,
    "per_object": null,
    "summary": {
      "material": "<строка>",
      "price_rub_per_g": <float>,
      "volume_model_cm3": <float>,
      "volume_print_cm3": <float>,
      "weight_g": <float>,
      "time_h": <float>,
      "total_rub": <float>,         # финальная цена после округления
      "total_raw_rub": <float>,     # до округления
      "min_applied": <bool>
    },
    "time_s": <float>
  }

Конкурентность и детерминизм:
  • Флаг --workers N включает мультипроцессную обработку ПО ФАЙЛАМ.
  • Результаты сортируются по имени файла перед агрегацией — порядок вывода стабильный.
  • Перед обработкой каждого файла выполняется полный сброс состояния парсера.

Конфиги (materials.json / pricing.json):
  • По умолчанию берутся из cwd (если есть оба файла), иначе рядом со скриптом.
  • Можно указать --config-dir для явной папки.
  • Переопределять отдельные параметры можно флагом --set key=val.

    Пример вызова из Python (FastAPI):
  >>> import subprocess, json
  >>> cmd = [
  ...   "treed-calc", "part.3mf",
  ...   "--json", "--material", "Enduse PETG",
  ...   "--infill", "20",
  ... ]
  >>> p = subprocess.run(cmd, capture_output=True, text=True, check=False)
  >>> if p.returncode == 0:
  ...     data = json.loads(p.stdout)
  ... else:
  ...     # обработать ошибку, логировать p.stderr
  ...     ...

Пример вызова из bash:
  $ treed-calc part.3mf --material "Enduse PETG" --infill 20 --json > result.json

Безопасность и идемпотентность:
  • Никаких глобальных накоплений между файлами/запусками; тест «один и тот же ввод -> один и тот же вывод» проходит.
  • На сервере запускайте с ограничением прав на чтение/запись только во временной папке.
"""
from __future__ import annotations

import os, sys, json, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

# импортируем ядро: геометрия, формулы, утилиты и форматтер отчёта
import core_calc as core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Утилиты ----------
def set_by_dotted_path(d: dict, path: str, value):
    """Устанавливает значение по точечному пути (например, 'printing.a1'). Создаёт вложенные словари при необходимости."""
    keys = path.split('.')
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def parse_kv_override(pairs):
    """Парсит список key=val оверрайдов из CLI (--set). Пытается привести val к bool/int/float, иначе оставляет строкой."""
    out = {}
    for kv in pairs or []:
        if '=' not in kv:
            raise ValueError(f"Неверный формат override '{kv}', нужен key=val")
        k, v = kv.split('=', 1)
        # попытка привести число/логическое
        vv = v
        try:
            if v.lower() in ('true','false'):
                vv = (v.lower() == 'true')
            elif '.' in v:
                vv = float(v)
            else:
                vv = int(v)
        except Exception:
            pass
        set_by_dotted_path(out, k, vv)
    return out

# ---------- Загрузка конфигов ----------
class ConfigError(Exception):
    """Ошибка конфигурации CLI (нет файла, неверный JSON, валидация и т.д.)."""


def resolve_config_paths(config_dir: str | None = None) -> tuple[str, str]:
    """Определяет пути к materials.json и pricing.json по config_dir/cwd/директории скрипта."""
    if config_dir:
        base_dir = os.path.abspath(os.path.expanduser(config_dir))
    else:
        cwd = os.getcwd()
        cwd_materials = os.path.join(cwd, "materials.json")
        cwd_pricing = os.path.join(cwd, "pricing.json")
        if os.path.exists(cwd_materials) and os.path.exists(cwd_pricing):
            base_dir = cwd
        else:
            base_dir = BASE_DIR
    return (
        os.path.join(base_dir, "materials.json"),
        os.path.join(base_dir, "pricing.json"),
    )


def load_configs_via_core(config_dir: str | None, override: dict | None = None) -> tuple[dict, dict, dict, str, str]:
    """Загружает materials/pricing через core_calc как единую точку правды."""
    materials_path, pricing_path = resolve_config_paths(config_dir)
    try:
        density, price_g = core.load_materials_json(materials_path)
    except FileNotFoundError as e:
        raise ConfigError(f"Файл конфигурации не найден: {e.filename}") from None
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"materials.json: ошибка JSON ({e.msg}, строка {e.lineno}, колонка {e.colno})"
        ) from None
    except ValueError as e:
        raise ConfigError(str(e)) from None

    try:
        pricing = core.load_pricing_json(pricing_path, override=override)
    except FileNotFoundError as e:
        raise ConfigError(f"Файл конфигурации не найден: {e.filename}") from None
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"pricing.json: ошибка JSON ({e.msg}, строка {e.lineno}, колонка {e.colno})"
        ) from None
    except ValueError as e:
        raise ConfigError(str(e)) from None
    return density, price_g, pricing, materials_path, pricing_path


def finalize_json_payload(payload: dict, errors: List[dict], count_ok: int) -> dict:
    """Добавляет поля ошибок и итоговые счетчики для JSON-вывода."""
    payload["errors"] = list(errors)
    payload["count_failed"] = len(errors)
    payload["count_ok"] = int(count_ok)
    payload["success"] = len(errors) == 0
    return payload

# ---------- Формулы времени и денег (идентичны GUI) ----------
def _compute_one_file(
    path: str,
    *,
    materials_density: dict,
    price_per_gram: dict,
    pricing: dict,
    material_name: str,
    infill: float,
    setup_min: float,
    post_min: float,
    qty: int,
    volume_mode: str,
    brief: bool,
    diag: bool,
) -> dict:
    """
    Процесс-воркер: считает один файл.
    Важно: состояние 3MF-парсера в core_calc глобальное, поэтому перед каждым файлом делаем сброс.
    """
    import time
    import os

    t0 = time.perf_counter()

    density = float(materials_density.get(material_name, 1.2))
    price_g = float(price_per_gram.get(material_name, 0.0))

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # СБРОС ПЕРЕД МОДЕЛЬЮ
    core._reset_parser_state()
    objs = core.parse_geometry(path)

    vol_factor = float((pricing.get("geometry", {}) or {}).get("volume_factor", 1.0))
    if volume_mode == "stream" and os.path.splitext(path)[1].lower() != ".stl":
        raise ValueError("volume-mode=stream is supported only for binary STL")
    any_3mf = any(((srcinfo or {}).get("type") == "3mf") for _, _, _, _, srcinfo in objs)

    total_V_model_cm3 = 0.0
    total_V_print_cm3 = 0.0
    total_weight_g = 0.0
    total_mat_cost = 0.0

    # Параметры печати — намеренно захардкожены (как в UI)
    wall_count = 2
    wall_width = 0.4
    layer_height_geo = 0.2
    top_bottom_layers = 4

    for _, V, T, vol_fast_cm3, srcinfo in objs:
        # 1) Объём модели (см³)
        # Для 3MF в fast-режиме используем предрасчитанный объём с учётом transforms.
        meta_for_volume = dict(srcinfo or {})
        if (
            meta_for_volume.get("type") == "3mf"
            and vol_fast_cm3
            and vol_fast_cm3 > 0
        ):
            meta_for_volume["precomputed_volume_cm3"] = float(vol_fast_cm3)
        V_model = float(core.compute_volume_cm3(V, T, mode=volume_mode, meta=meta_for_volume))

        # 2) Объём печати (см³): стенки/крышки/заполнение (как в UI)
        V_total = float(
            core.compute_print_volume_cm3(
                V_model_cm3=V_model,
                V_mm=V,
                T=T,
                infill_pct=float(infill),
                fast_only=False,
                wall_count=wall_count,
                wall_width=wall_width,
                layer_height_geo=layer_height_geo,
                top_bottom_layers=top_bottom_layers,
            )
        )

        # 3) Калибровка объёма
        V_total *= vol_factor

        # 4) Вес и материал
        weight_g = V_total * density
        mat_cost = weight_g * price_g

        total_V_model_cm3 += V_model
        total_V_print_cm3 += V_total
        total_weight_g += weight_g
        total_mat_cost += mat_cost

    # Тираж: масштабируем переменные метрики на qty (setup/post/min-order остаются per-order)
    total_V_model_cm3, total_V_print_cm3, total_weight_g, total_mat_cost = core.apply_qty_to_totals(
        volume_model_cm3=total_V_model_cm3,
        volume_print_cm3=total_V_print_cm3,
        weight_g=total_weight_g,
        material_cost=total_mat_cost,
        qty=int(qty),
    )

    bd = core.calc_breakdown(pricing, total_mat_cost, total_V_print_cm3, setup_min, post_min)

    calc_seconds = time.perf_counter() - t0

    # diag только для 3MF (иначе у STL будет мусорный пустой блок "Файл:")
    diag_text = ""
    if diag and any_3mf:
        diag_text = core.status_block_text()

    return {
        "file": os.path.basename(path),
        "qty": int(qty),
        "material": material_name,
        "price_rub_per_g": price_g,
        "volume_model_cm3": float(total_V_model_cm3),
        "volume_print_cm3": float(total_V_print_cm3),
        "weight_g": float(total_weight_g),
        "time_h": float(bd["time_h"]),
        "costs": {
            "material": float(total_mat_cost),
            "energy": float(bd["energy_cost"]),
            "depreciation": float(bd["depreciation_cost"]),
            "labor": float(bd["labor_cost"]),
            "reserve": float(bd["reserve"]),
        },
        "subtotal_rub": float(bd["subtotal"]),
        "min_order_rub": float(pricing.get("min_order_rub", 0.0)),
        "min_applied": bool(bd["min_applied"]),
        "markup_rub": float(bd["markup"]),
        "platform_fee_rub": float(bd["service_fee"]),
        "vat_rub": float(bd["vat"]),
        "total_rub": float(bd["total_with_min"]),
        "total_rounded_rub": float(bd["total"]),
        "rounding_step_rub": float(pricing.get("rounding_to_rub", 1.0)),
        "calc_seconds": float(calc_seconds),
        "diag_text": diag_text,
    }


# ---------- Расчёт набора файлов ----------
def compute_for_files(
    files: List[str],
    *,
    materials_density: dict,
    price_per_gram: dict,
    pricing: dict,
    material_name: str,
    infill: float,
    setup_min: float,
    post_min: float,
    qty: int,
    brief: bool,
    diag: bool,
    per_object: bool,
    as_json: bool,
    workers: int = 1,
    volume_mode: str = "fast",
    errors: List[dict] | None = None,
) -> dict:
    """
    Высокоуровневая функция: считает набор файлов с опциональной параллелью.
    Возвращает либо JSON payload (as_json=True), либо {"text": "..."}.
    Важно: бизнес-логика расчёта (геометрия/объёмы/ценообразование) живёт в core_calc.
    """
    t0 = time.time()

    # Нормализация входных значений (защита от мусора/отрицательных значений)
    infill = float(max(0.0, min(100.0, float(infill))))
    setup_min = float(max(0.0, float(setup_min)))
    post_min = float(max(0.0, float(post_min)))
    qty = int(core.coerce_qty(qty))

    results: List[dict] = []
    errors = errors if errors is not None else []
    grand = {"V_model_cm3": 0.0, "V_print_cm3": 0.0, "weight_g": 0.0, "material_cost": 0.0}

    file_list = list(files)

    # Параллель: по файлам, только если файлов>1 и workers>1
    if workers and workers > 1 and len(file_list) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futs = {
                ex.submit(
                    _compute_one_file,
                    p,
                    materials_density=materials_density,
                    price_per_gram=price_per_gram,
                    pricing=pricing,
                    material_name=material_name,
                    infill=infill,
                    setup_min=setup_min,
                    post_min=post_min,
                    qty=qty,
                    volume_mode=volume_mode,
                    brief=brief,
                    diag=diag,
                ): p
                for p in file_list
            }
            for fut in as_completed(futs):
                path = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    errors.append({"file": os.path.basename(path), "error": str(exc)})

        # стабильный порядок для вывода/тестов
        results.sort(key=lambda r: r["file"])
    else:
        # Последовательная обработка тем же кодом, что и в воркере (один источник правды)
        for p in file_list:
            try:
                results.append(
                    _compute_one_file(
                        p,
                        materials_density=materials_density,
                        price_per_gram=price_per_gram,
                        pricing=pricing,
                        material_name=material_name,
                        infill=infill,
                        setup_min=setup_min,
                        post_min=post_min,
                        qty=qty,
                        volume_mode=volume_mode,
                        brief=brief,
                        diag=diag,
                    )
                )
            except Exception as exc:
                errors.append({"file": os.path.basename(p), "error": str(exc)})

    # Аггрегация общих метрик (без повторного применения setup/post/min-order)
    for r in results:
        grand["V_model_cm3"] += float(r.get("volume_model_cm3", 0.0))
        grand["V_print_cm3"] += float(r.get("volume_print_cm3", 0.0))
        grand["weight_g"] += float(r.get("weight_g", 0.0))
        costs = r.get("costs") or {}
        grand["material_cost"] += float(costs.get("material", 0.0))

    calc_time_s = time.time() - t0

    # ---------- JSON ----------
    if as_json:
        payload = {
            "success": True,
            "qty": int(qty),
            "count": len(results),
            "per_object": results if per_object else None,
            "summary": None,
            "time_s": calc_time_s,
        }
        if not per_object:
            bd = core.calc_breakdown(pricing, grand["material_cost"], grand["V_print_cm3"], setup_min, post_min)
            payload["summary"] = {
                "qty": int(qty),
                "material": material_name,
                "price_rub_per_g": float(price_per_gram.get(material_name, 0.0)),
                "volume_model_cm3": grand["V_model_cm3"],
                "volume_print_cm3": grand["V_print_cm3"],
                "weight_g": grand["weight_g"],
                "time_h": bd["time_h"],
                "total_rub": bd["total"],
                "total_raw_rub": bd["total_with_min"],
                "min_applied": bd["min_applied"],
            }
        return finalize_json_payload(payload, errors, len(results))

    # ---------- TEXT ----------
    lines: List[str] = []
    if per_object:
        for r in results:
            report = core.render_report(
                file_name=r["file"],
                obj_name=r["file"],
                material_name=r["material"],
                price_per_g=r["price_rub_per_g"],
                volume_model_cm3=r["volume_model_cm3"],
                volume_print_cm3=r["volume_print_cm3"],
                weight_g=r["weight_g"],
                time_h=r["time_h"],
                costs=r["costs"],
                subtotal_rub=r["subtotal_rub"],
                min_order_rub=r["min_order_rub"],
                min_applied=r["min_applied"],
                markup_rub=r["markup_rub"],
                platform_fee_rub=r["platform_fee_rub"],
                vat_rub=r["vat_rub"],
                total_rub=r["total_rub"],
                total_rounded_rub=r["total_rounded_rub"],
                rounding_step_rub=r["rounding_step_rub"],
                calc_time_s=calc_time_s,
                qty=int(qty),
                brief=brief,
                show_diag=diag,
                diag_text=r.get("diag_text", "") if diag else "",
            )
            lines.append(report)
            lines.append("\n")
        return {"text": "".join(lines).rstrip()}

    # сводная "Сборка (N объектов)"
    bd = core.calc_breakdown(pricing, grand["material_cost"], grand["V_print_cm3"], setup_min, post_min)
    obj_name = f"Сборка ({len(results)} объектов)"

    # Диагностика: берём из результатов (в workers>1 главный процесс не парсил файлы)
    diag_join = ""
    if diag:
        parts = [str(r.get("diag_text", "")).rstrip() for r in results if r.get("diag_text")]
        if parts:
            diag_join = "\n".join(parts) + "\n"

    report = core.render_report(
        file_name="; ".join([r["file"] for r in results]),
        obj_name=obj_name,
        material_name=material_name,
        price_per_g=float(price_per_gram.get(material_name, 0.0)),
        volume_model_cm3=grand["V_model_cm3"],
        volume_print_cm3=grand["V_print_cm3"],
        weight_g=grand["weight_g"],
        time_h=bd["time_h"],
        costs={
            "material": grand["material_cost"],
            "energy": bd["energy_cost"],
            "depreciation": bd["depreciation_cost"],
            "labor": bd["labor_cost"],
            "reserve": bd["reserve"],
        },
        subtotal_rub=bd["subtotal"],
        min_order_rub=float(pricing.get("min_order_rub", 0.0)),
        min_applied=bd["min_applied"],
        markup_rub=bd["markup"],
        platform_fee_rub=bd["service_fee"],
        vat_rub=bd["vat"],
        total_rub=bd["total_with_min"],
        total_rounded_rub=bd["total"],
        rounding_step_rub=float(pricing.get("rounding_to_rub", 1.0)),
        calc_time_s=calc_time_s,
        qty=int(qty),
        brief=brief,
        show_diag=diag,
        diag_text=diag_join,
    )
    return {"text": report}

# ---------- CLI ----------
def main():
    """Точка входа CLI. Парсит аргументы, загружает конфиги, валидирует материал, вызывает compute_for_files и печатает результат."""
    # Windows/CP1251 safe output: не падаем на спецсимволах
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="CLI 3D калькулятор печати (FDM) — .3mf/.stl, без UI, идентичен GUI-логике")
    ap.add_argument('files', nargs='+', help='Пути к моделям .3mf / .stl')
    ap.add_argument('--materials', default='(ignored)', help='[ignored] используйте --config-dir')
    ap.add_argument('--pricing',   default='(ignored)',      help='[ignored] используйте --config-dir')
    ap.add_argument('--set', dest='overrides', action='append', help='Переопределить параметры pricing (format: key=val, напр. printing.a1=1.02). Можно несколько раз.')
    ap.add_argument('--config-dir', default=None, help='Папка с materials.json и pricing.json (по умолчанию: cwd или рядом со скриптом)')

    ap.add_argument('--material', required=False, help='Название материала из materials.json')
    ap.add_argument('--infill', type=float, default=10.0, help='% заполнения (0-100)')
    ap.add_argument('--setup-min', type=float, default=10.0, help='Подготовка (мин)')
    ap.add_argument('--post-min',  type=float, default=0.0,  help='Постпроцесс (мин)')
    ap.add_argument('--qty', type=int, default=1, help='Количество одинаковых комплектов моделей (тираж). Цена считается за заказ.')
    ap.add_argument(
        '--volume-mode',
        choices=['fast', 'stream', 'bbox'],
        default='fast',
        help='Режим расчёта объёма: fast (tetra), stream (только бинарный STL), bbox (по габаритам).',
    )

    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument('--json', action='store_true', help='Вывод в JSON')
    fmt.add_argument('--text', action='store_true', help='Текстовый отчёт (по умолчанию)')

    ap.add_argument('--full', dest='brief', action='store_false', help='Полный отчёт (иначе краткий)')
    ap.add_argument('--diag', action='store_true', help='Добавить блок диагностики 3MF')
    ap.add_argument('--per-object', action='store_true', help='Считать и выводить каждый файл отдельно (по умолчанию — сводный)')
    ap.add_argument('--workers', type=int, default=1, help='Процессы для параллельной обработки файлов (>1 — включить мультипроцессинг)')

    args = ap.parse_args()

    try:
        args.qty = core.coerce_qty(args.qty)
    except Exception as e:
        print(f"Неверное значение --qty: {e}", file=sys.stderr)
        sys.exit(2)

    # Нормализуем числовые параметры (CLI может дать отрицательные значения)
    args.infill = float(max(0.0, min(100.0, args.infill)))
    args.setup_min = float(max(0.0, args.setup_min))
    args.post_min = float(max(0.0, args.post_min))

    # загрузка конфигов
    try:
        overrides = parse_kv_override(args.overrides)
    except ValueError as e:
        print(f"Неверный формат --set: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        dens, priceg, pricing, materials_path, pricing_path = load_configs_via_core(args.config_dir, overrides)
    except ConfigError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[cli] using materials: {materials_path}", file=sys.stderr)
    print(f"[cli] using pricing  : {pricing_path}", file=sys.stderr)


    # выбор материала
    material = args.material
    if not material:
        material = sorted(dens.keys())[0]
    if material not in dens:
        print(f"Материал '{material}' не найден в {materials_path}", file=sys.stderr)
        sys.exit(2)

    errors: List[dict] = []
    try:
        payload = compute_for_files(
            args.files,
            materials_density=dens, price_per_gram=priceg, pricing=pricing,
            material_name=material, infill=args.infill, setup_min=args.setup_min, post_min=args.post_min,
            qty=int(args.qty),
            brief=bool(args.brief), diag=bool(args.diag),
            per_object=bool(args.per_object), as_json=bool(args.json),
            workers=int(max(1, args.workers)),
            volume_mode=str(args.volume_mode),
            errors=errors,
        )
    except Exception as e:
        print(f"Ошибка расчёта: {e}", file=sys.stderr)
        sys.exit(1)

    if errors and not args.json:
        for err in errors:
            print(f"[cli] файл {err.get('file')}: {err.get('error')}", file=sys.stderr)

    # вывод
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["text"])

    if errors:
        sys.exit(1)

if __name__ == '__main__':
    main()
