# -*- coding: utf-8 -*-
"""
3D калькулятор печати (FDM) — аккуратные 3MF-трансформы + калибровка объёма +
читабельный отчёт (краткий/подробный) + опция диагностики.

• Поддержка .3mf (build/items, production p:path) и .stl (потоковый объём).
• FDM-разбивка: стенки/крышки/заполнение, затем корректировка geometry.volume_factor.
• Время печати: throughput + travel_factor + a0/a1 (калибруется в pricing.json).
• Ценообразование: материал + энергия + амортизация + труд + резерв + наценка + комиссия + НДС.
• UI: загрузка materials.json и pricing.json, «Краткий вывод», «Диагностика 3MF».

Где править параметры:
- materials.json   → плотность (density_g_cm3) и цена (price_rub_per_g) по материалам
- pricing.json     → travel_factor, a0, a1, depreciation_per_hour_rub, volume_factor и пр.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os, struct, zipfile, json, xml.etree.ElementTree as ET
import numpy as np
import time
from typing import Dict, Tuple, List

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

# --- НОВОЕ: полный сброс состояния парсера между открытиями файлов ---
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
    _reset_parser_state()  # важный сброс
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
                    if V_mm.size == 0 and vol_mm3_fast == 0.0: continue
                    name = f"{os.path.basename(mf)}:object_{oid}"
                    data.append((name, V_mm, T, vol_mm3_fast / 1000.0, {'type': '3mf', 'path': path}))
                continue

            for idx, item in enumerate(items, 1):
                oid = item.get('objectid')
                Mitem = _parse_transform(item.get('transform'))
                _last_status["det_values"].append(float(np.linalg.det(Mitem[:3, :3])))
                V_mm, T, vol_mm3_fast = _flatten_object_cached(cache, mf, oid, Mitem)
                if V_mm.size == 0 and vol_mm3_fast == 0.0: continue
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
    _reset_parser_state()  # сброс и для STL
    ext = os.path.splitext(path)[1].lower()
    if ext == '.3mf': return parse_3mf(path)
    if ext == '.stl': return parse_stl(path)
    raise ValueError('Only .3mf and .stl supported')

# ---------- Конфиги ----------
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

# ---------- Приложение ----------
class CalculatorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('3D калькулятор (FDM) — красивый отчёт')
        self.root.geometry('740x900')
        self.loaded = []
        self.materials_density = dict(DEFAULT_MATERIALS_DENSITY)
        self.price_per_gram    = dict(DEFAULT_PRICE_PER_GRAM)
        self.pricing           = json.loads(json.dumps(DEFAULT_PRICING))

        # UI-переменные
        self.mode = tk.StringVar(value='tetra')
        self.fast_volume_var = tk.IntVar(value=0)
        self.stream_stl_var = tk.IntVar(value=0)
        self.selected_material = tk.StringVar(value=next(iter(self.materials_density.keys())))
        self.brief_var = tk.IntVar(value=1)
        self.diag_var  = tk.IntVar(value=0)

        self.entry_infill = None; self.entry_setup = None; self.entry_post = None; self.output = None

        self.try_autoload_json()
        self.build_ui()

    # ---- JSON ----
    def try_autoload_json(self) -> None:
        try:
            if os.path.exists("materials.json"):
                data = json.load(open("materials.json", "r", encoding="utf-8"))
                self.materials_density.clear(); self.price_per_gram.clear()
                for k, v in data.items():
                    self.materials_density[k] = float(v.get('density_g_cm3', 1.2))
                    self.price_per_gram[k]    = float(v.get('price_rub_per_g', 0.0))
        except Exception as e:
            print("[WARN] materials.json:", e)
        try:
            if os.path.exists("pricing.json"):
                data = json.load(open("pricing.json", "r", encoding="utf-8"))
                self._deep_merge(self.pricing, data)
        except Exception as e:
            print("[WARN] pricing.json:", e)

    def _deep_merge(self, dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                self._deep_merge(dst[k], v)
            else:
                dst[k] = v

    def load_materials_json(self) -> None:
        path = filedialog.askopenfilename(filetypes=[('JSON', '*.json')], title='Открыть JSON материалов')
        if not path: return
        try:
            data = json.load(open(path, 'r', encoding='utf-8'))
            self.materials_density.clear(); self.price_per_gram.clear()
            for k, v in data.items():
                self.materials_density[k] = float(v.get('density_g_cm3', 1.2))
                self.price_per_gram[k]    = float(v.get('price_rub_per_g', 0.0))
            self.update_material_menu()
            messagebox.showinfo('OK', f'Материалы загружены: {len(self.materials_density)} позиций')
            self.recalc()
        except Exception as e:
            messagebox.showerror('Ошибка JSON материалов', str(e))

    def load_pricing_json(self) -> None:
        path = filedialog.askopenfilename(filetypes=[('JSON', '*.json')], title='Открыть JSON конфига')
        if not path: return
        try:
            data = json.load(open(path, 'r', encoding='utf-8'))
            self._deep_merge(self.pricing, data)
            messagebox.showinfo('OK', 'Конфиг ценообразования загружен')
            self.recalc()
        except Exception as e:
            messagebox.showerror('Ошибка JSON конфига', str(e))

    # ---- UI ----
    def build_ui(self) -> None:
        frame = tk.Frame(self.root, bg='#f9f9f9', padx=20, pady=20); frame.pack(fill='both', expand=True)
        tk.Label(frame, text='3D Калькулятор (.3mf / .stl)', font=('Arial', 20, 'bold'), bg='#f9f9f9').pack(pady=(0, 10))

        json_bar = tk.Frame(frame, bg='#f9f9f9'); json_bar.pack(fill='x', pady=(0, 6))
        tk.Button(json_bar, text='Загрузить materials.json', command=self.load_materials_json, bg='#dbeafe').pack(side='left', padx=(0, 6))
        tk.Button(json_bar, text='Загрузить pricing.json',   command=self.load_pricing_json,   bg='#e9d5ff').pack(side='left')

        tk.Radiobutton(frame, text='Ограничивающий параллелепипед',
                       variable=self.mode, value='bbox', font=('Arial', 12),
                       bg='#f9f9f9', fg='#7A6EB0', command=self.recalc).pack(anchor='w')
        tk.Radiobutton(frame, text='Тетраэдры',
                       variable=self.mode, value='tetra', font=('Arial', 12),
                       bg='#f9f9f9', fg='#7A6EB0', command=self.recalc).pack(anchor='w')

        fast_frame = tk.Frame(frame, bg='#f9f9f9'); fast_frame.pack(pady=(6, 6), fill='x')
        tk.Checkbutton(fast_frame, text='Быстрый объём (без стенок/крышек)',
                       variable=self.fast_volume_var, bg='#f9f9f9', command=self.recalc).pack(anchor='w')
        tk.Checkbutton(fast_frame, text='Потоковый STL (объём напрямую из файла)',
                       variable=self.stream_stl_var, bg='#f9f9f9', command=self.recalc).pack(anchor='w')

        material_frame = tk.Frame(frame, bg='#f9f9f9'); material_frame.pack(pady=(5, 5), fill='x')
        tk.Label(material_frame, text='Материал:', font=('Arial', 12), bg='#f9f9f9').pack(side='left')
        self.material_menu = tk.OptionMenu(material_frame, self.selected_material, *self.materials_density.keys())
        self.material_menu.config(font=('Arial', 12), bg='white'); self.material_menu.pack(side='left', padx=5)
        self.selected_material.trace_add('write', lambda *_: self.recalc())

        infill_frame = tk.Frame(frame, bg='#f9f9f9'); infill_frame.pack(pady=(5, 6), fill='x')
        tk.Label(infill_frame, text='Заполнение (%):', font=('Arial', 12), bg='#f9f9f9').pack(side='left')
        self.entry_infill = tk.Entry(infill_frame, width=6, font=('Arial', 12)); self.entry_infill.insert(0, '10')
        self.entry_infill.pack(side='left', padx=5); self.entry_infill.bind('<KeyRelease>', lambda _e: self.recalc())

        work_frame = tk.Frame(frame, bg='#f9f9f9'); work_frame.pack(pady=(5, 6), fill='x')
        tk.Label(work_frame, text='Подготовка (мин):', font=('Arial', 12), bg='#f9f9f9').pack(side='left')
        self.entry_setup = tk.Entry(work_frame, width=6, font=('Arial', 12)); self.entry_setup.insert(0, '10')
        self.entry_setup.pack(side='left', padx=5)
        tk.Label(work_frame, text='Постпроцесс (мин):', font=('Arial', 12), bg='#f9f9f9').pack(side='left')
        self.entry_post = tk.Entry(work_frame, width=6, font=('Arial', 12)); self.entry_post.insert(0, '0')
        self.entry_post.pack(side='left', padx=5)
        self.entry_setup.bind('<KeyRelease>', lambda _e: self.recalc()); self.entry_post.bind('<KeyRelease>', lambda _e: self.recalc())

        switches = tk.Frame(frame, bg='#f9f9f9'); switches.pack(pady=(4, 8), fill='x')
        tk.Checkbutton(switches, text='Краткий вывод', variable=self.brief_var, bg='#f9f9f9', command=self.recalc).pack(side='left')
        tk.Checkbutton(switches, text='Диагностика 3MF', variable=self.diag_var, bg='#f9f9f9', command=self.recalc).pack(side='left', padx=(12,0))

        tk.Button(frame, text='Загрузить 3D файл', font=('Arial', 12, 'bold'),
                  bg='#7A6EB0', fg='white', command=self.open_file).pack(pady=10, fill='x')

        self.output = scrolledtext.ScrolledText(frame, font=('Consolas', 12), state='disabled', height=24)
        self.output.pack(fill='both', expand=True, pady=5)

    def update_material_menu(self) -> None:
        menu = self.material_menu['menu']; menu.delete(0, 'end')
        for n in self.materials_density.keys():
            menu.add_command(label=n, command=lambda v=n: self.selected_material.set(v))
        if self.selected_material.get() not in self.materials_density:
            first = next(iter(self.materials_density.keys())); self.selected_material.set(first)

    # ---- Файлы ----
    def open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[('3D Files', '*.3mf *.stl')])
        if not path: return
        try:
            objs = parse_geometry(path)
        except Exception as e:
            messagebox.showerror('Ошибка', str(e)); return
        # не накапливаем — заменяем целиком
        self.loaded = list(objs)
        self.recalc()

    # ---- Вспомогательные блоки ----
    def _status_block_text(self) -> str:
        dets = _last_status.get("det_values") or []
        det_min = f"{min(dets):.3f}" if dets else "—"
        det_max = f"{max(dets):.3f}" if dets else "—"
        units = ", ".join(sorted(_last_status.get("unit_set") or [])) or "millimeter"
        return (f"Файл: {_last_status.get('file','')}\n"
                f"Единицы (из моделей): {units}\n"
                f"Items: {_last_status.get('item_count',0)} | Components: {_last_status.get('component_count',0)} | p:path внешних: {_last_status.get('external_p_path',0)}\n"
                f"det по items: min={det_min}, max={det_max}\n"
                "----------------------------------------\n")

    def estimate_time_hours_by_volume(self, V_total_cm3: float) -> float:
        p = self.pricing.get("printing", {})
        Q_mm3_s = (nz(p.get("print_speed_mm_s"), 100) * nz(p.get("line_width_mm"), 0.45) *
                   nz(p.get("layer_height_mm"), 0.20) * max(0.0, min(1.0, nz(p.get("utilization"), 0.85))))
        if Q_mm3_s <= 0: return 0.0
        est_h = (nz(V_total_cm3) * 1000.0 * nz(p.get("travel_factor"), 1.0)) / (Q_mm3_s * 3600.0)
        a0 = nz(p.get("a0"), 0.0); a1 = nz(p.get("a1"), 1.0)
        return max(0.0, a0 + a1 * est_h)

    def calc_breakdown(self, total_mat_cost: float, total_V_total_cm3: float, setup_min: float, postproc_min: float) -> dict:
        p = self.pricing
        t_h = self.estimate_time_hours_by_volume(total_V_total_cm3)

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
        reserve = (non_material * (nz(risk_cfg.get("pct"),0.0) / 100.0)) if (risk_cfg.get("apply_to","non_material") == "non_material") else 0.0

        subtotal = total_mat_cost + non_material + reserve

        # Минимумы и наценки
        min_order = nz(p.get("min_order_rub"), 0.0)
        min_policy = p.get("min_policy", "final")

        markup_pct = nz(p.get("markup_pct"),0.0)/100.0
        fee_pct    = nz(p.get("service_fee_pct"),0.0)/100.0
        vat_pct    = nz(p.get("vat_pct"),0.0)/100.0

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

        return {"time_h": t_h,
                "energy_cost": energy_cost, "depreciation_cost": depreciation_cost, "labor_cost": labor_cost,
                "consumables": consumables, "fixed_overhead": fixed_overhead, "non_material": non_material,
                "reserve": reserve, "subtotal": subtotal, "markup": markup, "service_fee": service_fee, "vat": vat,
                "total_raw": chain_total, "total_with_min": total_raw, "min_applied": bool(min_applied), "total": total}

    # ---- Основной перерасчёт ----
    def recalc(self) -> None:
        t0 = time.time()
        self.output.config(state='normal'); self.output.delete('1.0', tk.END)
        if not self.loaded:
            self.output.insert(tk.END, 'Сначала загрузите модель.'); self.output.config(state='disabled'); return

        try:
            infill = float(self.entry_infill.get())
        except ValueError:
            messagebox.showerror('Ошибка', 'Введите корректный % заполнения.'); self.output.config(state='disabled'); return
        try:
            setup_min = float(self.entry_setup.get()); post_min = float(self.entry_post.get())
        except ValueError:
            messagebox.showerror('Ошибка', 'Подготовка/постпроцесс — числа.'); self.output.config(state='disabled'); return

        material = self.selected_material.get()
        density = nz(self.materials_density.get(material), 1.2)
        price_g = nz(self.price_per_gram.get(material), 0.0)

        mode_val = self.mode.get()
        fast_only = self.fast_volume_var.get() == 1
        stream_stl = self.stream_stl_var.get() == 1

        diag_text = self._status_block_text() if (self.diag_var.get() == 1) else ""

        wall_count = 2; wall_width = 0.4; layer_height_geo = 0.2; top_bottom_layers = 4
        vol_factor = nz(self.pricing.get("geometry", {}).get("volume_factor"), 1.0)

        total_weight_g = 0.0
        total_material_cost = 0.0
        total_V_total_cm3 = 0.0
        total_V_model_cm3 = 0.0

        for _, V, T, vol_fast_cm3, src in self.loaded:
            if src.get('type') == '3mf' and vol_fast_cm3 > 0:
                V_model = vol_fast_cm3
            else:
                if fast_only and src.get('type') == 'stl' and stream_stl and src.get('path'):
                    try:
                        V_model = stl_stream_volume_cm3(src['path'])
                    except Exception:
                        V_model = volume_bbox(V) if mode_val == 'bbox' else volume_tetra(V, T)
                else:
                    V_model = volume_bbox(V) if mode_val == 'bbox' else volume_tetra(V, T)

            if fast_only:
                V_total = V_model * (infill / 100.0)
            else:
                shell_area = surface_area_mesh(V, T)  # см²
                xy_area = xy_area_bbox_from_V(V)      # см²
                V_shell = shell_area * wall_count * wall_width / 10.0
                V_top_bottom = xy_area * top_bottom_layers * layer_height_geo / 10.0
                shell_total = V_shell + V_top_bottom
                if shell_total > V_model * 0.6:
                    scale = (V_model * 0.6) / max(shell_total, 1e-12)
                    V_shell *= scale; V_top_bottom *= scale
                V_infill = max(0.0, V_model - V_shell - V_top_bottom) * (infill / 100.0)
                V_total = V_shell + V_top_bottom + V_infill

            V_total *= vol_factor  # калибровка

            weight_g = V_total * density
            mat_cost = weight_g * price_g

            total_V_model_cm3 += V_model
            total_V_total_cm3 += V_total
            total_weight_g += weight_g
            total_material_cost += mat_cost

        bd = self.calc_breakdown(total_material_cost, total_V_total_cm3, setup_min, post_min)

        if len(self.loaded) == 1:
            obj_name = os.path.basename(self.loaded[0][0])
        else:
            obj_name = f"Сборка ({len(self.loaded)} объектов)"

        report = render_report(
            file_name=_last_status.get("file",""),
            obj_name=obj_name,
            material_name=material,
            price_per_g=price_g,
            volume_model_cm3=total_V_model_cm3,
            volume_print_cm3=total_V_total_cm3,
            weight_g=total_weight_g,
            time_h=bd["time_h"],
            costs={
                "material": total_material_cost,
                "energy": bd["energy_cost"],
                "depreciation": bd["depreciation_cost"],
                "labor": bd["labor_cost"],
                "reserve": bd["reserve"],
            },
            subtotal_rub=bd["subtotal"],
            min_order_rub=nz(self.pricing.get("min_order_rub"), 0.0),
            min_applied=bd["min_applied"],
            markup_rub=bd["markup"],
            platform_fee_rub=bd["service_fee"],
            vat_rub=bd["vat"],
            total_rub=bd["total_with_min"],
            total_rounded_rub=bd["total"],
            rounding_step_rub=nz(self.pricing.get("rounding_to_rub"), 1.0),
            calc_time_s=time.time() - t0,
            brief=bool(self.brief_var.get() == 1),
            show_diag=bool(self.diag_var.get() == 1),
            diag_text=diag_text
        )

        self.output.insert(tk.END, report)
        if price_g <= 0:
            self.output.insert(tk.END, '\n⚠ ВНИМАНИЕ: у выбранного материала цена/г = 0. Итог занижен.\n')
        self.output.config(state='disabled')

    # ---- run ----
    def run(self) -> None:
        self.root.mainloop()

if __name__ == "__main__":
    app = CalculatorApp()
    app.run()
