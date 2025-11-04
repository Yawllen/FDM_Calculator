
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
  • ВСЕГДА грузятся из папки, где лежит сам скрипт (строгий режим).
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRICT_MATERIALS_PATH = os.path.join(BASE_DIR, 'materials.json')
STRICT_PRICING_PATH   = os.path.join(BASE_DIR, 'pricing.json')


# === Strict config loading: always from script directory ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRICT_MATERIALS_PATH = os.path.join(BASE_DIR, 'materials.json')
STRICT_PRICING_PATH   = os.path.join(BASE_DIR, 'pricing.json')
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

# импортируем ядро: геометрия, формулы, утилиты и форматтер отчёта
sys.path.append('/mnt/data')
import calculator as core  # type: ignore

# ---------- Утилиты ----------
def deep_merge(dst: dict, src: dict) -> dict:
    """Глубокое слияние словарей src в dst (in-place). Возвращает dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

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
def load_materials(path: str):
    """Читает materials.json строго из указанного пути. Возвращает (density_by_material, price_per_gram_by_material)."""
    if not os.path.exists(path):
        raise SystemExit(f"[cli] materials.json не найден: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise SystemExit(f"[cli] materials.json: ошибка чтения/JSON — {e}")
    if not isinstance(data, dict) or not data:
        raise SystemExit("[cli] materials.json: ожидается объект {material: {...}}")
    density, price_g = {}, {}
    for name, row in data.items():
        if not isinstance(row, dict):
            raise SystemExit(f"[cli] materials.json: неверный формат у материала '{name}'")
        try:
            d = float(row['density_g_cm3'])
            pg = float(row['price_rub_per_g'])
        except Exception:
            raise SystemExit(f"[cli] materials.json: у материала '{name}' нет density_g_cm3/price_rub_per_g или неверный тип")
        density[name] = d
        price_g[name] = pg
    return density, price_g

def load_pricing(path: str, override: dict | None = None) -> dict:
    """Читает pricing.json строго из указанного пути. Применяет override (--set)."""
    if not os.path.exists(path):
        raise SystemExit(f"[cli] pricing.json не найден: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        raise SystemExit(f"[cli] pricing.json: ошибка чтения/JSON — {e}")
    if not isinstance(cfg, dict) or not cfg:
        raise SystemExit("[cli] pricing.json: ожидается объект конфигурации")
    if override:
        deep_merge(cfg, override)
    return cfg

# ---------- Формулы времени и денег (идентичны GUI) ----------
def estimate_time_hours_by_volume(pricing: dict, V_total_cm3: float) -> float:
    """Оценивает время печати (ч) по эффективному объёму печати и параметрам производительности из pricing.printing."""
    p = pricing.get("printing", {})
    Q_mm3_s = (core.nz(p.get("print_speed_mm_s"), 100) * core.nz(p.get("line_width_mm"), 0.45) *
               core.nz(p.get("layer_height_mm"), 0.20) * max(0.0, min(1.0, core.nz(p.get("utilization"), 0.85))))
    if Q_mm3_s <= 0:
        return 0.0
    est_h = (core.nz(V_total_cm3) * 1000.0 * core.nz(p.get("travel_factor"), 1.0)) / (Q_mm3_s * 3600.0)
    a0 = core.nz(p.get("a0"), 0.0); a1 = core.nz(p.get("a1"), 1.0)
    return max(0.0, a0 + a1 * est_h)

def calc_breakdown(pricing: dict, total_mat_cost: float, total_V_total_cm3: float, setup_min: float, postproc_min: float) -> dict:
    """Считает разбиение стоимости: материал/энергия/амортизация/труд/резерв + порядок надбавок/НДС/минималка. Возвращает словарь полей для итогов."""
    p = pricing
    t_h = estimate_time_hours_by_volume(pricing, total_V_total_cm3)

    power = p.get("power", {})
    energy_cost = t_h * (core.nz(power.get("avg_power_w"), 0.0) / 1000.0) * core.nz(power.get("tariff_rub_per_kwh"), 0.0)
    depreciation_cost = t_h * core.nz(p.get("depreciation_per_hour_rub"), 0.0)

    labor = p.get("labor", {})
    setup_billable_h = max(0.0, core.nz(setup_min) - core.nz(labor.get("setup_min_included"), 0.0)) / 60.0
    post_h = core.nz(postproc_min) / 60.0
    labor_cost = core.nz(labor.get("hour_rate_rub"), 0.0) * (setup_billable_h + post_h)

    extras = p.get("extras", {})
    consumables = core.nz(extras.get("consumables_rub"), 0.0)
    fixed_overhead = core.nz(extras.get("fixed_overhead_rub"), 0.0)

    non_material = energy_cost + depreciation_cost + labor_cost + consumables + fixed_overhead

    risk_cfg = p.get("risk", {})
    reserve = (non_material * (core.nz(risk_cfg.get("pct"),0.0) / 100.0)) if (risk_cfg.get("apply_to","non_material") == "non_material") else 0.0

    subtotal = total_mat_cost + non_material + reserve

    # Минимумы и наценки
    min_order = core.nz(p.get("min_order_rub"), 0.0)
    min_policy = p.get("min_policy", "final")

    markup_pct = core.nz(p.get("markup_pct"),0.0)/100.0
    fee_pct    = core.nz(p.get("service_fee_pct"),0.0)/100.0
    vat_pct    = core.nz(p.get("vat_pct"),0.0)/100.0

    if min_policy == "subtotal":
        base_for_margin = max(min_order, subtotal)
        min_applied = (min_order > subtotal)
        markup = base_for_margin * markup_pct
        after_markup = base_for_margin + markup
        service_fee = after_markup * fee_pct
        after_service = after_markup + service_fee
        vat = after_service * vat_pct
        chain_total = after_service + vat
        total_raw = chain_total
    elif min_policy == "after_markup":
        markup = subtotal * markup_pct
        after_markup = subtotal + markup
        base2 = max(min_order, after_markup)
        min_applied = (min_order > after_markup)
        service_fee = base2 * fee_pct
        after_service = base2 + service_fee
        vat = after_service * vat_pct
        chain_total = after_service + vat
        total_raw = chain_total
    else:  # "final"
        markup = subtotal * markup_pct
        after_markup = subtotal + markup
        service_fee = after_markup * fee_pct
        after_service = after_markup + service_fee
        vat = after_service * vat_pct
        chain_total = after_service + vat
        min_applied = (min_order > chain_total)
        total_raw = max(min_order, chain_total)

    total = core.round_to_step(total_raw, p.get("rounding_to_rub", 1))
    return {"time_h": t_h,
            "energy_cost": energy_cost, "depreciation_cost": depreciation_cost, "labor_cost": labor_cost,
            "consumables": consumables, "fixed_overhead": fixed_overhead, "non_material": non_material,
            "reserve": reserve, "subtotal": subtotal, "markup": markup, "service_fee": service_fee, "vat": vat,
            "total_raw": chain_total, "total_with_min": total_raw, "min_applied": bool(min_applied), "total": total}

# ---------- Диагностика парсера ----------
def status_block_text() -> str:
    """Возвращает диагностический блок по последнему распарсенному файлу (единицы, статистика по items). Только для человека, не JSON-API."""
    dets = core._last_status.get("det_values") or []
    det_min = f"{min(dets):.3f}" if dets else "—"
    det_max = f"{max(dets):.3f}" if dets else "—"
    units = ", ".join(sorted(core._last_status.get("unit_set") or [])) or "millimeter"
    return (f"Файл: {core._last_status.get('file','')}\n"
            f"Единицы (из моделей): {units}\n"
            f"Items: {core._last_status.get('item_count',0)} | Components: {core._last_status.get('component_count',0)} | p:path внешних: {core._last_status.get('external_p_path',0)}\n"
            f"det по items: min={det_min}, max={det_max}\n"
            "----------------------------------------\n")

# ---------- Параллельная обработка одного файла ----------
def _compute_one_file(path: str, *, materials_density: dict, price_per_gram: dict, pricing: dict,
                      material_name: str, infill: float, setup_min: float, post_min: float,
                      brief: bool, diag: bool) -> dict:
    """Процесс-воркер: считает один файл в отдельном процессе (изоляция состояния). Возвращает готовую запись для per_object массива."""
    density = float(materials_density.get(material_name, 1.2))
    price_g = float(price_per_gram.get(material_name, 0.0))

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # СБРОС ПЕРЕД МОДЕЛЬЮ (в процессе)
    core._reset_parser_state()  # ВАЖНО: чистим любое глобальное/кэшированное состояние перед очередным файлом
    objs = core.parse_geometry(path)

    vol_factor = float(pricing.get("geometry",{}).get("volume_factor", 1.0))

    total_V_model_cm3 = 0.0
    total_V_total_cm3 = 0.0
    total_weight_g    = 0.0
    total_mat_cost    = 0.0

    for _, V, T, vol_fast_cm3, srcinfo in objs:
        # 1) объём модели
        V_model = core.volume_tetra(V, T)

        # 2) стенки/крышки/заполнение
        shell_area = core.surface_area_mesh(V, T)  # см²
        xy_area = core.xy_area_bbox_from_V(V)      # см²
        wall_count = 2; wall_width = 0.4; layer_height_geo = 0.2; top_bottom_layers = 4

        V_shell = shell_area * wall_count * wall_width / 10.0
        V_top_bottom = xy_area * top_bottom_layers * layer_height_geo / 10.0
        shell_total = V_shell + V_top_bottom
        if shell_total > V_model * 0.6:
            scale = (V_model * 0.6) / max(shell_total, 1e-12)
            V_shell *= scale; V_top_bottom *= scale
        V_infill = max(0.0, V_model - V_shell - V_top_bottom) * (infill / 100.0)
        V_total = V_shell + V_top_bottom + V_infill

        # 3) калибровка объёма
        V_total *= vol_factor

        # 4) вес/материал
        weight_g = V_total * density
        mat_cost = weight_g * price_g

        total_V_model_cm3 += V_model
        total_V_total_cm3 += V_total
        total_weight_g    += weight_g
        total_mat_cost    += mat_cost

    bd = calc_breakdown(pricing, total_mat_cost, total_V_total_cm3, setup_min, post_min)

    res = {
        "file": os.path.basename(path),
        "material": material_name,
        "price_rub_per_g": price_g,
        "volume_model_cm3": total_V_model_cm3,
        "volume_print_cm3": total_V_total_cm3,
        "weight_g": total_weight_g,
        "time_h": bd["time_h"],
        "costs": {
            "material": total_mat_cost,
            "energy": bd["energy_cost"],
            "depreciation": bd["depreciation_cost"],
            "labor": bd["labor_cost"],
            "reserve": bd["reserve"],
        },
        "subtotal_rub": bd["subtotal"],
        "min_order_rub": float(pricing.get("min_order_rub", 0.0)),
        "min_applied": bd["min_applied"],
        "markup_rub": bd["markup"],
        "platform_fee_rub": bd["service_fee"],
        "vat_rub": bd["vat"],
        "total_rub": bd["total_with_min"],
        "total_rounded_rub": bd["total"],
        "rounding_step_rub": float(pricing.get("rounding_to_rub", 1.0)),
        "diag_text": status_block_text() if diag else ""
    }
    return res

# ---------- Расчёт набора файлов ----------
def compute_for_files(files: List[str], *, materials_density: dict, price_per_gram: dict, pricing: dict,
                      material_name: str, infill: float, setup_min: float, post_min: float,
                      brief: bool, diag: bool, per_object: bool, as_json: bool, workers: int = 1) -> dict:
    """Высокоуровневая функция: считает набор файлов с опциональной параллелью. Формирует JSON payload или текстовый отчёт."""
    t0 = time.time()
    density = float(materials_density.get(material_name, 1.2))
    price_g = float(price_per_gram.get(material_name, 0.0))

    results = []
    grand = {"V_model_cm3": 0.0, "V_print_cm3": 0.0, "weight_g": 0.0, "material_cost": 0.0}

    # Параллель: по файлам, только если файлов>1 и workers>1
    if workers and workers > 1 and len(files) > 1:
        file_list = list(files)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _compute_one_file, p,
                    materials_density=materials_density, price_per_gram=price_per_gram, pricing=pricing,
                    material_name=material_name, infill=infill, setup_min=setup_min, post_min=post_min,
                    brief=brief, diag=diag
                ) for p in file_list
            ]
            for fut in as_completed(futs):
                results.append(fut.result())
        # стабильный порядок для вывода/тестов
        results.sort(key=lambda r: r["file"])  # Детерминированный вывод и агрегация

        for r in results:
            grand["V_model_cm3"] += r["volume_model_cm3"]
            grand["V_print_cm3"] += r["volume_print_cm3"]
            grand["weight_g"]    += r["weight_g"]
            grand["material_cost"] += r["costs"]["material"]  # материал=сумма по объектам
    else:
        for path in files:
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            # СБРОС ПЕРЕД КАЖДОЙ МОДЕЛЬЮ
            core._reset_parser_state()  # ВАЖНО: чистим любое глобальное/кэшированное состояние перед очередным файлом

            objs = core.parse_geometry(path)  # гарант. сброс и внутри

            vol_factor = float(pricing.get("geometry",{}).get("volume_factor", 1.0))

            total_V_model_cm3 = 0.0
            total_V_total_cm3 = 0.0
            total_weight_g    = 0.0
            total_mat_cost    = 0.0

            for _, V, T, vol_fast_cm3, src in objs:
                # 1) объём модели
                V_model = core.volume_tetra(V, T)

                # 2) разбиение на стенки/крышки/заполнение (как в GUI)
                shell_area = core.surface_area_mesh(V, T)  # см²
                xy_area = core.xy_area_bbox_from_V(V)      # см²
                wall_count = 2; wall_width = 0.4; layer_height_geo = 0.2; top_bottom_layers = 4

                V_shell = shell_area * wall_count * wall_width / 10.0
                V_top_bottom = xy_area * top_bottom_layers * layer_height_geo / 10.0
                shell_total = V_shell + V_top_bottom
                if shell_total > V_model * 0.6:
                    scale = (V_model * 0.6) / max(shell_total, 1e-12)
                    V_shell *= scale; V_top_bottom *= scale
                V_infill = max(0.0, V_model - V_shell - V_top_bottom) * (infill / 100.0)
                V_total = V_shell + V_top_bottom + V_infill

                # 3) калибровка объёма
                V_total *= vol_factor

                # 4) вес/материал
                weight_g = V_total * density
                mat_cost = weight_g * price_g

                total_V_model_cm3 += V_model
                total_V_total_cm3 += V_total
                total_weight_g    += weight_g
                total_mat_cost    += mat_cost

            bd = calc_breakdown(pricing, total_mat_cost, total_V_total_cm3, setup_min, post_min)

            # агрегируем в общий итог
            grand["V_model_cm3"] += total_V_model_cm3
            grand["V_print_cm3"] += total_V_total_cm3
            grand["weight_g"]    += total_weight_g
            grand["material_cost"] += total_mat_cost

            res = {
                "file": os.path.basename(path),
                "material": material_name,
                "price_rub_per_g": price_g,
                "volume_model_cm3": total_V_model_cm3,
                "volume_print_cm3": total_V_total_cm3,
                "weight_g": total_weight_g,
                "time_h": bd["time_h"],
                "costs": {
                    "material": total_mat_cost,
                    "energy": bd["energy_cost"],
                    "depreciation": bd["depreciation_cost"],
                    "labor": bd["labor_cost"],
                    "reserve": bd["reserve"],
                },
                "subtotal_rub": bd["subtotal"],
                "min_order_rub": float(pricing.get("min_order_rub", 0.0)),
                "min_applied": bd["min_applied"],
                "markup_rub": bd["markup"],
                "platform_fee_rub": bd["service_fee"],
                "vat_rub": bd["vat"],
                "total_rub": bd["total_with_min"],
                "total_rounded_rub": bd["total"],
                "rounding_step_rub": float(pricing.get("rounding_to_rub", 1.0)),
            }
            results.append(res)

    calc_time_s = time.time() - t0

    # Формирование вывода
    if as_json:
        payload = {
            "success": True,
            "count": len(results),
            "per_object": results if per_object else None,
            "summary": None,
            "time_s": calc_time_s,
        }
        if not per_object:
            # общий сводный расчёт по всем файлам как единой сборке
            bd = calc_breakdown(pricing, grand["material_cost"], grand["V_print_cm3"], setup_min, post_min)
            payload["summary"] = {
                "material": material_name,
                "price_rub_per_g": price_g,
                "volume_model_cm3": grand["V_model_cm3"],
                "volume_print_cm3": grand["V_print_cm3"],
                "weight_g": grand["weight_g"],
                "time_h": bd["time_h"],
                "total_rub": bd["total"],
                "total_raw_rub": bd["total_with_min"],
                "min_applied": bd["min_applied"],
            }
        return payload

    # Текстовый отчёт
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
                brief=brief,
                show_diag=diag,
                diag_text=status_block_text() if diag else ""
            )
            lines.append(report)
            lines.append("\n")
        return {"text": "".join(lines).rstrip()}

    # сводная "Сборка (N объектов)"
    bd = calc_breakdown(pricing, grand["material_cost"], grand["V_print_cm3"], setup_min, post_min)
    obj_name = f"Сборка ({len(results)} объектов)"
    report = core.render_report(
        file_name="; ".join([r["file"] for r in results]),
        obj_name=obj_name,
        material_name=material_name,
        price_per_g=price_g,
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
        brief=brief,
        show_diag=diag,
        diag_text=status_block_text() if diag else ""
    )
    return {"text": report}

# ---------- CLI ----------
def main():
    """Точка входа CLI. Парсит аргументы, загружает конфиги, валидирует материал, вызывает compute_for_files и печатает результат."""
    ap = argparse.ArgumentParser(description="CLI 3D калькулятор печати (FDM) — .3mf/.stl, без UI, идентичен GUI-логике")
    ap.add_argument('files', nargs='+', help='Пути к моделям .3mf / .stl')
    ap.add_argument('--materials', default='(ignored)', help='[ignored] materials.json всегда берётся из папки скрипта')
    ap.add_argument('--pricing',   default='(ignored)',      help='[ignored] pricing.json всегда берётся из папки скрипта')
    ap.add_argument('--set', dest='overrides', action='append', help='Переопределить параметры pricing (format: key=val, напр. printing.a1=1.02). Можно несколько раз.')

    ap.add_argument('--material', required=False, help='Название материала из materials.json')
    ap.add_argument('--infill', type=float, default=10.0, help='% заполнения (0-100)')
    ap.add_argument('--setup-min', type=float, default=10.0, help='Подготовка (мин)')
    ap.add_argument('--post-min',  type=float, default=0.0,  help='Постпроцесс (мин)')

    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument('--json', action='store_true', help='Вывод в JSON')
    fmt.add_argument('--text', action='store_true', help='Текстовый отчёт (по умолчанию)')

    ap.add_argument('--full', dest='brief', action='store_false', help='Полный отчёт (иначе краткий)')
    ap.add_argument('--diag', action='store_true', help='Добавить блок диагностики 3MF')
    ap.add_argument('--per-object', action='store_true', help='Считать и выводить каждый файл отдельно (по умолчанию — сводный)')
    ap.add_argument('--workers', type=int, default=1, help='Процессы для параллельной обработки файлов (>1 — включить мультипроцессинг)')

    args = ap.parse_args()

    # загрузка конфигов
    print(f"[cli] using materials: {STRICT_MATERIALS_PATH}", file=sys.stderr)
    print(f"[cli] using pricing  : {STRICT_PRICING_PATH}",   file=sys.stderr)
    dens, priceg = load_materials(STRICT_MATERIALS_PATH)
    pricing = load_pricing(STRICT_PRICING_PATH, override=parse_kv_override(args.overrides))


    # выбор материала
    material = args.material
    if not material:
        material = sorted(dens.keys())[0]
    if material not in dens:
        raise SystemExit(f"Материал '{material}' не найден в {args.materials}")

    payload = compute_for_files(
        args.files,
        materials_density=dens, price_per_gram=priceg, pricing=pricing,
        material_name=material, infill=args.infill, setup_min=args.setup_min, post_min=args.post_min,
        brief=bool(args.brief), diag=bool(args.diag),
        per_object=bool(args.per_object), as_json=bool(args.json), workers=int(max(1, args.workers))
    )

    # вывод
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["text"])

if __name__ == '__main__':
    main()
