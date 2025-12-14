# -*- coding: utf-8 -*-
"""
core_calc.py — чистое ядро 3D калькулятора печати (FDM)

Цели:
- Никакого UI / tkinter.
- Один источник правды для: геометрии/парсинга, FDM-разбивки, времени, ценообразования, форматирования отчёта.
- UI и CLI должны быть тонкими оболочками, импортирующими этот модуль.

Совместимость:
- Формат materials.json и pricing.json не меняется.
- Поведение расчётов соответствует текущей GUI/CLI логике (tetra, стенки/крышки/заполнение + volume_factor).
"""
from __future__ import annotations

import os
import struct
import zipfile
import json
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List

import numpy as np


# ---------- Утилиты ----------
def nz(v, d=0.0) -> float:
    try:
        f = float(v)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return d


def round_to_step(value: float, step: float) -> float:
    s = nz(step, 1.0)
    return round(value / s) * s if s > 0 else value


def _hm(hours: float) -> str:
    hours = max(0.0, nz(hours))
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h}ч {m:02d}м"


def _rub(v: float) -> str:
    s = f"{nz(v):,.2f}".replace(",", " ")
    if s.endswith(".00"):
        s = s[:-3]
    return s + " ₽"


def _line(label: str, value: float, width: int = 12) -> str:
    return f"  {label:<28}{_rub(value):>{width}}\n"


def deep_merge(dst: dict, src: dict) -> dict:
    """Глубокое слияние словарей src в dst (in-place). Возвращает dst."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


# ---------- Config loading (единая правда для UI/CLI) ----------
def get_default_config_dir() -> str:
    """
    Папка, относительно которой по умолчанию ищем materials.json / pricing.json.
    Детерминированно: рядом с core_calc.py.
    """
    return os.path.dirname(os.path.abspath(__file__))


def get_default_materials_path(config_dir: str | None = None) -> str:
    base = config_dir or get_default_config_dir()
    return os.path.join(base, "materials.json")


def get_default_pricing_path(config_dir: str | None = None) -> str:
    base = config_dir or get_default_config_dir()
    return os.path.join(base, "pricing.json")


def load_materials_json(path: str) -> tuple[dict, dict]:
    """
    materials.json -> (density_by_material, price_per_g_by_material)
    Формат: { "Material": {"density_g_cm3": 1.2, "price_rub_per_g": 2.3}, ... }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("materials.json: expected object {material: {...}}")

    density, price_g = {}, {}
    for name, row in data.items():
        if not isinstance(row, dict):
            raise ValueError(f"materials.json: invalid row for '{name}'")
        if "density_g_cm3" not in row or "price_rub_per_g" not in row:
            raise ValueError(f"materials.json: '{name}' missing density_g_cm3/price_rub_per_g")
        density[name] = float(row["density_g_cm3"])
        price_g[name] = float(row["price_rub_per_g"])

    return density, price_g


def load_pricing_json(path: str, *, base: dict | None = None, override: dict | None = None) -> dict:
    """
    pricing.json -> pricing dict.
    base: если задан, то в него мерджится файл (удобно для DEFAULT_PRICING).
    override: мердж поверх результата (например, --set в CLI).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("pricing.json: expected object")

    out = json.loads(json.dumps(base)) if isinstance(base, dict) else {}
    deep_merge(out, cfg)
    if override:
        deep_merge(out, override)
    return out



# ---------- Форматирование отчёта (общий для UI/CLI) ----------
def render_report(*,
    file_name: str,
    obj_name: str,
    material_name: str,
    price_per_g: float,
    volume_model_cm3: float,
    volume_print_cm3: float,
    weight_g: float,
    time_h: float,
    costs: dict,
    subtotal_rub: float,
    min_order_rub: float,
    min_applied: bool,
    markup_rub: float,
    platform_fee_rub: float,
    vat_rub: float,
    total_rub: float,
    total_rounded_rub: float,
    rounding_step_rub: float,
    calc_time_s: float,
    brief: bool = True,
    show_diag: bool = False,
    diag_text: str = ""
) -> str:
    head = []
    if show_diag and diag_text:
        head.append(diag_text.rstrip() + "\n")

    head.append(f"Деталь: {obj_name}\n")
    head.append(f"• Объём: модель {volume_model_cm3:.2f} см³ → печать {volume_print_cm3:.2f} см³\n")
    head.append(f"• Вес: {weight_g:.2f} г | Время печати: {_hm(time_h)}\n")
    head.append(f"• Материал: {material_name} ({nz(price_per_g):.2f} ₽/г)\n")
    head.append("-" * 42 + "\n")

    if brief:
        body = []
        body.append(_line("Материал", costs.get("material", 0.0)))
        other = costs.get("energy", 0) + costs.get("depreciation", 0) + costs.get("labor", 0) + costs.get("reserve", 0)
        if other:
            body.append(_line("Прочие (энергия, аморт., труд, резерв)", other))
        body.append(_line("Промежуточная сумма", subtotal_rub))
        body.append(f"  Минимальный чек: {_rub(min_order_rub)} → {'применён' if min_applied else 'не нужен'}\n")
        body.append(_line("Наценка 25%", markup_rub))
        body.append(_line("Комиссия площадки 10%", platform_fee_rub))
        body.append(_line("НДС 22%", vat_rub))
        body.append("-" * 42 + "\n")
        body.append(f"ИТОГО: {_rub(total_rounded_rub)} (округление шагом {int(nz(rounding_step_rub,1))} ₽ из {_rub(total_rub)})\n")
        body.append(f"Время расчёта: {calc_time_s:.4f} с\n")
        return "".join(head + body)

    # Подробная версия
    b = []
    b.append(_line("Материал", costs.get("material", 0.0)))
    b.append(_line("Электроэнергия", costs.get("energy", 0.0)))
    b.append(_line("Амортизация", costs.get("depreciation", 0.0)))
    b.append(_line("Труд", costs.get("labor", 0.0)))
    b.append(_line("Резерв (без материала)", costs.get("reserve", 0.0)))
    b.append(_line("Промежуточная сумма", subtotal_rub))
    b.append("\n")
    b.append(_line("Наценка 25%", markup_rub))
    b.append(_line("Комиссия площадки 10%", platform_fee_rub))
    b.append(_line("НДС 22%", vat_rub))
    b.append(f"  Минимальный чек: {_rub(min_order_rub)} → {'применён' if min_applied else 'не нужен'}\n")
    b.append("-" * 42 + "\n")
    b.append(f"ИТОГО: {_rub(total_rounded_rub)} (округление: {_rub(total_rub)} → шаг {int(nz(rounding_step_rub,1))} ₽)\n")
    b.append(f"Файл: {file_name} | Расчёт: {calc_time_s:.4f} с\n")
    return "".join(head + b)


# ---------- Геометрия ----------
def volume_tetra_units(V: np.ndarray, T: np.ndarray) -> float:
    if V.size == 0 or T.size == 0:
        return 0.0
    v0 = V[T[:, 0]]; v1 = V[T[:, 1]]; v2 = V[T[:, 2]]
    cross = np.cross(v1, v2)
    vol6 = np.einsum('ij,ij->i', v0, cross)
    return abs(vol6.sum()) / 6.0


def volume_tetra(V_mm: np.ndarray, T: np.ndarray) -> float:
    if V_mm.size == 0 or T.size == 0:
        return 0.0
    v0 = V_mm[T[:, 0]]; v1 = V_mm[T[:, 1]]; v2 = V_mm[T[:, 2]]
    cross = np.cross(v1, v2)
    vol6 = np.einsum('ij,ij->i', v0, cross)
    vol_mm3 = abs(vol6.sum()) / 6.0
    return vol_mm3 / 1000.0


def surface_area_mesh(V_mm: np.ndarray, T: np.ndarray) -> float:
    if V_mm.size == 0 or T.size == 0:
        return 0.0
    v0 = V_mm[T[:, 0]]; v1 = V_mm[T[:, 1]]; v2 = V_mm[T[:, 2]]
    area_mm2 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
    return area_mm2 / 100.0


def xy_area_bbox_from_V(V_mm: np.ndarray) -> float:
    if V_mm.size == 0:
        return 0.0
    mins = V_mm.min(axis=0); maxs = V_mm.max(axis=0)
    dx, dy = (maxs[0] - mins[0]), (maxs[1] - mins[1])
    return (dx * dy) / 100.0


def volume_bbox(V_mm: np.ndarray) -> float:
    if V_mm.size == 0:
        return 0.0
    mins = V_mm.min(axis=0); maxs = V_mm.max(axis=0)
    dx, dy, dz = (maxs - mins)
    return (dx * dy * dz) / 1000.0


# ---------- 3MF / STL ----------
NAMESPACE = {'ns': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
NS_PROD   = 'http://schemas.microsoft.com/3dmanufacturing/production/2015/06'


def _unit_to_mm(unit_str: str) -> float:
    unit = (unit_str or 'millimeter').strip().lower()
    return {
        'micron': 0.001, 'millimeter': 1.0, 'centimeter': 10.0, 'meter': 1000.0, 'inch': 25.4, 'foot': 304.8
    }.get(unit, 1.0)


def _parse_transform(s: str | None) -> np.ndarray:
    if not s:
        return np.eye(4, dtype=np.float64)
    vals = [float(x) for x in s.replace(',', ' ').split()]
    if len(vals) != 12:
        return np.eye(4, dtype=np.float64)
    m00, m01, m02, m10, m11, m12, m20, m21, m22, m30, m31, m32 = vals
    return np.array([
        [m00, m01, m02, 0.0],
        [m10, m11, m12, 0.0],
        [m20, m21, m22, 0.0],
        [m30, m31, m32, 1.0],
    ], dtype=np.float64)


def _apply_transform(V_mm: np.ndarray, M: np.ndarray) -> np.ndarray:
    R = M[:3, :3]; t = M[3, :3]
    return V_mm @ R + t


def _detect_and_set_namespace(root: ET.Element) -> None:
    try:
        if root.tag.startswith('{') and '}model' in root.tag:
            ns_uri = root.tag[1:].split('}')[0]
            NAMESPACE['ns'] = ns_uri
    except Exception:
        pass


# --- состояние парсера (для диагностики) ---
_last_status = {"file": "","unit_set": set(),"item_count": 0,"component_count": 0,"external_p_path": 0,"det_values": []}


def _reset_parser_state():
    global NAMESPACE, _last_status
    NAMESPACE = {'ns': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    _last_status = {
        "file": "",
        "unit_set": set(),
        "item_count": 0,
        "component_count": 0,
        "external_p_path": 0,
        "det_values": [],
    }


def status_block_text() -> str:
    """Текстовый диагностический блок по последнему распарсенному файлу (3MF units/items/components)."""
    dets = _last_status.get("det_values") or []
    det_min = f"{min(dets):.3f}" if dets else "—"
    det_max = f"{max(dets):.3f}" if dets else "—"
    units = ", ".join(sorted(_last_status.get("unit_set") or [])) or "millimeter"
    return (f"Файл: {_last_status.get('file','')}\n"
            f"Единицы (из моделей): {units}\n"
            f"Items: {_last_status.get('item_count',0)} | Components: {_last_status.get('component_count',0)} | p:path внешних: {_last_status.get('external_p_path',0)}\n"
            f"det по items: min={det_min}, max={det_max}\n"
            "----------------------------------------\n")


def _gather_model_mm(root: ET.Element, unit_scale_mm: float, model_path: str):
    meshes_mm: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    comps_map: Dict[str, List[Tuple[str, str, np.ndarray]]] = {}
    base_vol_mm3: Dict[str, float] = {}

    for obj in root.findall('.//ns:object', NAMESPACE):
        oid = obj.get('id')
        mesh = obj.find('ns:mesh', NAMESPACE)
        if mesh is not None:
            vs = mesh.find('ns:vertices', NAMESPACE)
            verts = []
            if vs is not None:
                for v in vs.findall('ns:vertex', NAMESPACE):
                    verts.append((float(v.get('x', '0')) * unit_scale_mm,
                                  float(v.get('y', '0')) * unit_scale_mm,
                                  float(v.get('z', '0')) * unit_scale_mm))
            V_mm = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3), dtype=np.float64)

            ts = mesh.find('ns:triangles', NAMESPACE)
            tris = []
            if ts is not None:
                for t in ts.findall('ns:triangle', NAMESPACE):
                    tris.append((int(t.get('v1', '0')),
                                 int(t.get('v2', '0')),
                                 int(t.get('v3', '0'))))
            T = np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)
            meshes_mm[oid] = (V_mm, T)
            base_vol_mm3[oid] = volume_tetra_units(V_mm / max(unit_scale_mm, 1e-12), T) * (unit_scale_mm ** 3)
        else:
            comp_list: List[Tuple[str, str, np.ndarray]] = []
            comps_node = obj.find('ns:components', NAMESPACE)
            if comps_node is not None:
                for c in comps_node.findall('ns:component', NAMESPACE):
                    ref = c.get('objectid')
                    M = _parse_transform(c.get('transform'))
                    p_path = c.get(f'{{{NS_PROD}}}path') or c.get('path')
                    child_model = p_path.lstrip('/') if p_path else model_path
                    comp_list.append((child_model, ref, M))
            comps_map[oid] = comp_list
    return meshes_mm, comps_map, base_vol_mm3


def _build_model_cache(zf: zipfile.ZipFile):
    cache = {}
    model_files = [f for f in zf.namelist() if f.startswith('3D/') and f.endswith('.model')]

    _last_status["unit_set"].clear()
    _last_status["item_count"] = 0
    _last_status["component_count"] = 0
    _last_status["external_p_path"] = 0
    _last_status["det_values"].clear()

    for mf in model_files:
        root = ET.fromstring(zf.read(mf))
        _detect_and_set_namespace(root)
        unit_scale = _unit_to_mm(root.get('unit'))
        _last_status["unit_set"].add(root.get('unit') or 'millimeter')
        meshes_mm, comps_map, base_vol_mm3 = _gather_model_mm(root, unit_scale, mf)
        _last_status["component_count"] += sum(len(v) for v in comps_map.values())
        for lst in comps_map.values():
            for child_model, _, _ in lst:
                if child_model != mf:
                    _last_status["external_p_path"] += 1
        cache[mf] = {'unit_scale_mm': unit_scale,'meshes_mm': meshes_mm,'comps': comps_map,'base_vol_mm3': base_vol_mm3,'root': root}
    return cache


def _flatten_object_cached(cache: dict, model_file: str, oid: str, cum_M: np.ndarray):
    entry = cache.get(model_file)
    if entry is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32), 0.0

    meshes_mm = entry['meshes_mm']; comps = entry['comps']; base_vol = entry['base_vol_mm3']

    if oid in meshes_mm:
        V_mm, T = meshes_mm[oid]
        if V_mm.size == 0 or T.size == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32), 0.0
        Vt = _apply_transform(V_mm, cum_M)
        det_full = abs(np.linalg.det(cum_M[:3, :3]))
        vol_mm3_fast = base_vol.get(oid, 0.0) * det_full
        return Vt, T.copy(), vol_mm3_fast

    out_V, out_T, offset, vol_mm3_fast = [], [], 0, 0.0
    for child_model, child_oid, Mchild in comps.get(oid, []):
        cum_next = cum_M @ Mchild
        Vc, Tc, vc = _flatten_object_cached(cache, child_model, child_oid, cum_next)
        if Vc.size == 0 or Tc.size == 0:
            vol_mm3_fast += vc
            continue
        out_V.append(Vc); out_T.append(Tc + offset)
        offset += Vc.shape[0]; vol_mm3_fast += vc

    if out_V:
        V = np.vstack(out_V); T = np.vstack(out_T)
        return V, T, vol_mm3_fast

    return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32), vol_mm3_fast


def parse_3mf(path: str):
    _reset_parser_state()
    data = []
    _last_status["file"] = os.path.basename(path)
    with zipfile.ZipFile(path) as z:
        cache = _build_model_cache(z)
        model_files = list(cache.keys())
        item_models, items_per_model = [], {}
        for mf in model_files:
            root = cache[mf]['root']
            _detect_and_set_namespace(root)
            build = root.find('ns:build', NAMESPACE)
            items = [] if build is None else build.findall('ns:item', NAMESPACE)
            items_per_model[mf] = items
            if items:
                item_models.append(mf)
                _last_status["item_count"] += len(items)

        selected = item_models if item_models else model_files
        for mf in selected:
            items = items_per_model.get(mf, [])
            if not items:
                all_ids = set(cache[mf]['meshes_mm'].keys()) | set(cache[mf]['comps'].keys())
                for oid in all_ids:
                    V_mm, T, vol_mm3_fast = _flatten_object_cached(cache, mf, oid, np.eye(4))
                    if V_mm.size == 0 and vol_mm3_fast == 0.0:
                        continue
                    name = f"{os.path.basename(mf)}:object_{oid}"
                    data.append((name, V_mm, T, vol_mm3_fast / 1000.0, {'type': '3mf', 'path': path}))
                continue

            for idx, item in enumerate(items, 1):
                oid = item.get('objectid')
                Mitem = _parse_transform(item.get('transform'))
                _last_status["det_values"].append(float(np.linalg.det(Mitem[:3, :3])))
                V_mm, T, vol_mm3_fast = _flatten_object_cached(cache, mf, oid, Mitem)
                if V_mm.size == 0 and vol_mm3_fast == 0.0:
                    continue
                name = f"{os.path.basename(mf)}:item_{idx}"
                data.append((name, V_mm, T, vol_mm3_fast / 1000.0, {'type': '3mf', 'path': path}))
    return data


def stl_stream_volume_cm3(path: str) -> float:
    with open(path, 'rb') as f:
        f.seek(80); count = struct.unpack('<I', f.read(4))[0]
        total6 = 0.0
        for _ in range(count):
            f.read(12)
            v0 = struct.unpack('<fff', f.read(12))
            v1 = struct.unpack('<fff', f.read(12))
            v2 = struct.unpack('<fff', f.read(12))
            cx = (v1[1]*v2[2] - v1[2]*v2[1])
            cy = (v1[2]*v2[0] - v1[0]*v2[2])
            cz = (v1[0]*v2[1] - v1[1]*v2[0])
            total6 += v0[0]*cx + v0[1]*cy + v0[2]*cz
            f.read(2)
    vol_mm3 = abs(total6) / 6.0
    return vol_mm3 / 1000.0


def parse_stl(path: str):
    verts, tris, idx_map = [], [], {}
    with open(path, 'rb') as f:
        f.seek(80); count = struct.unpack('<I', f.read(4))[0]
        for _ in range(count):
            f.read(12); face = []
            for _ in range(3):
                xyz = struct.unpack('<fff', f.read(12))
                if xyz not in idx_map:
                    idx_map[xyz] = len(verts); verts.append(xyz)
                face.append(idx_map[xyz])
            tris.append(tuple(face)); f.read(2)
    V = np.array(verts, dtype=np.float64); T = np.array(tris, dtype=np.int32)
    vol_fast_cm3 = volume_tetra(V, T)
    return [("STL model", V, T, vol_fast_cm3, {'type': 'stl', 'path': path})]


def parse_geometry(path: str):
    _reset_parser_state()
    ext = os.path.splitext(path)[1].lower()
    if ext == '.3mf':
        return parse_3mf(path)
    if ext == '.stl':
        return parse_stl(path)
    raise ValueError('Only .3mf and .stl supported')


# ---------- Defaults (как раньше в UI) ----------
DEFAULT_MATERIALS_DENSITY = {"Enduse PETG": 1.27, "Proto PLA": 1.24, "Enduse ABS": 1.04, "Другой материал": 1.00}
DEFAULT_PRICE_PER_GRAM    = {"Enduse PETG": 2.33, "Proto PLA": 3.07, "Enduse ABS": 2.40, "Другой материал": 2.00}

DEFAULT_PRICING = {
    "currency": "RUB",
    "rounding_to_rub": 10,
    "min_order_rub": 380,
    "min_policy": "final",
    "vat_pct": 22,
    "markup_pct": 25,
    "service_fee_pct": 10,
    "risk": {"pct": 10, "apply_to": "non_material"},
    "power": {"tariff_rub_per_kwh": 4.7, "avg_power_w": 350},
    "depreciation_per_hour_rub": 30,
    "labor": {"hour_rate_rub": 500, "setup_min_included": 10},
    "printing": {
        "print_speed_mm_s": 100,
        "line_width_mm": 0.45,
        "layer_height_mm": 0.20,
        "utilization": 0.85,
        "travel_factor": 1.00,
        "a0": 0.0,
        "a1": 1.0
    },
    "geometry": {"volume_factor": 1.0},
    "extras": {"fixed_overhead_rub": 0.0, "consumables_rub": 0.0}
}


# ---------- Формулы времени и денег ----------
def estimate_time_hours_by_volume(pricing: dict, V_total_cm3: float) -> float:
    p = (pricing or {}).get("printing", {})
    Q_mm3_s = (nz(p.get("print_speed_mm_s"), 100) * nz(p.get("line_width_mm"), 0.45) *
               nz(p.get("layer_height_mm"), 0.20) * max(0.0, min(1.0, nz(p.get("utilization"), 0.85))))
    if Q_mm3_s <= 0:
        return 0.0
    est_h = (nz(V_total_cm3) * 1000.0 * nz(p.get("travel_factor"), 1.0)) / (Q_mm3_s * 3600.0)
    a0 = nz(p.get("a0"), 0.0); a1 = nz(p.get("a1"), 1.0)
    return max(0.0, a0 + a1 * est_h)


def calc_breakdown(pricing: dict, total_mat_cost: float, total_V_total_cm3: float, setup_min: float, postproc_min: float) -> dict:
    p = pricing or {}
    t_h = estimate_time_hours_by_volume(p, total_V_total_cm3)

    power = p.get("power", {})
    energy_cost = t_h * (nz(power.get("avg_power_w"), 0.0) / 1000.0) * nz(power.get("tariff_rub_per_kwh"), 0.0)
    depreciation_cost = t_h * nz(p.get("depreciation_per_hour_rub"), 0.0)

    labor = p.get("labor", {})
    setup_billable_h = max(0.0, nz(setup_min) - nz(labor.get("setup_min_included"), 0.0)) / 60.0
    post_h = nz(postproc_min) / 60.0
    labor_cost = nz(labor.get("hour_rate_rub"), 0.0) * (setup_billable_h + post_h)

    extras = p.get("extras", {})
    consumables = nz(extras.get("consumables_rub"), 0.0)
    fixed_overhead = nz(extras.get("fixed_overhead_rub"), 0.0)

    non_material = energy_cost + depreciation_cost + labor_cost + consumables + fixed_overhead

    risk_cfg = p.get("risk", {})
    reserve = (non_material * (nz(risk_cfg.get("pct"), 0.0) / 100.0)) if (risk_cfg.get("apply_to", "non_material") == "non_material") else 0.0

    subtotal = total_mat_cost + non_material + reserve

    min_order = nz(p.get("min_order_rub"), 0.0)
    min_policy = p.get("min_policy", "final")

    markup_pct = nz(p.get("markup_pct"), 0.0) / 100.0
    fee_pct    = nz(p.get("service_fee_pct"), 0.0) / 100.0
    vat_pct    = nz(p.get("vat_pct"), 0.0) / 100.0

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

    total = round_to_step(total_raw, p.get("rounding_to_rub", 1))
    return {
        "time_h": t_h,
        "energy_cost": energy_cost,
        "depreciation_cost": depreciation_cost,
        "labor_cost": labor_cost,
        "consumables": consumables,
        "fixed_overhead": fixed_overhead,
        "non_material": non_material,
        "reserve": reserve,
        "subtotal": subtotal,
        "markup": markup,
        "service_fee": service_fee,
        "vat": vat,
        "total_raw": chain_total,
        "total_with_min": total_raw,
        "min_applied": bool(min_applied),
        "total": total,
    }


# ---------- FDM-разбивка объёма (как в текущем UI/CLI) ----------
def compute_print_volume_cm3(
    V_model_cm3: float,
    V_mm: np.ndarray,
    T: np.ndarray,
    infill_pct: float,
    *,
    fast_only: bool,
    wall_count: int = 2,
    wall_width: float = 0.4,
    layer_height_geo: float = 0.2,
    top_bottom_layers: int = 4
) -> float:
    """Возвращает V_total (см³) до применения geometry.volume_factor."""
    infill = nz(infill_pct, 0.0)
    if fast_only:
        return max(0.0, nz(V_model_cm3)) * (infill / 100.0)

    shell_area = surface_area_mesh(V_mm, T)  # см²
    xy_area = xy_area_bbox_from_V(V_mm)      # см²

    V_shell = shell_area * wall_count * wall_width / 10.0
    V_top_bottom = xy_area * top_bottom_layers * layer_height_geo / 10.0
    shell_total = V_shell + V_top_bottom

    V_model = max(0.0, nz(V_model_cm3))
    if shell_total > V_model * 0.6:
        scale = (V_model * 0.6) / max(shell_total, 1e-12)
        V_shell *= scale
        V_top_bottom *= scale

    V_infill = max(0.0, V_model - V_shell - V_top_bottom) * (infill / 100.0)
    return V_shell + V_top_bottom + V_infill
