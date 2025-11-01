import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os, struct, zipfile, xml.etree.ElementTree as ET
import numpy as np
import time

"""
АККУРАТНАЯ ВЕРСИЯ БЕЗ 3D-ОТРИСОВКИ. Правильные трансформы 3MF, поддержка Production p:path,
быстрый потоковый STL, статус‑плашка и итоговая сумма.

Ключевые моменты (простыми словами):
- Матрица трансформации 3MF: 12 чисел в одну строку (row‑major). Перенос — последняя строка.
- Применение матрицы: v' = v @ R + t (R = M[:3,:3], t = M[3,:3]).
- Композиция в сборках: накапливаем справа (row‑major): cum_next = cum_M @ Mchild.
- Production `p:path`: компонент может ссылаться на объект в ДРУГОМ .model внутри архива — обрабатываем.
- Быстрый объём 3MF: базовый объём (мм³) × |det(ПОЛНОЙ матрицы)| ⇒ см³ (гарантирует ×8 при 200%).
- STL: при желании считаем объём «потоково» прямо из файла (без построения меша).
"""

# =========================
# Глобалы и материалы (НЕ МЕНЯЛ)
# =========================
loaded: list[tuple[str, np.ndarray, np.ndarray, float, dict]] = []  # [(name, V_mm, T, vol_fast_cm3, src), ...]
NAMESPACE = {'ns': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
NS_PROD = 'http://schemas.microsoft.com/3dmanufacturing/production/2015/06'

# Для статус‑плашки последнего парсинга
last_status = {
    "file": "",
    "unit_set": set(),
    "item_count": 0,
    "component_count": 0,
    "external_p_path": 0,
    "det_values": [],  # по item'ам
}

# Материалы (плотность, г/см³)
MATERIALS = {
    "Sealant TPU93": 1.20,
    "Fiberpart ABS G4": 1.04,
    "Fiberpart TPU C5": 1.20,
    "Fiberpart ABSPA G8": 1.10,
    "Fiberpart TPU G30": 1.20,
    "Enduse SBS": 1.04,
    "Enduse ABS": 1.04,
    "Enduse PETG": 1.27,
    "Enduse PP": 0.90,
    "Proto PLA": 1.24,
    "ContiFiber CPA": 1.15,
    "Sealant SEBS": 1.04,
    "Sealant TPU": 1.20,
    "Proto PVA": 1.19,
    "Enduse-PA": 1.15,
    "Enduse-TPU D70": 1.20,
    "Fiberpart ABS G13": 1.04,
    "Fiberpart PP G": 0.90,
    "Fiberpart PP G30": 0.90,
    "Enduse PC": 1.20,
    "Fiberpart PA12 G12": 1.01,
    "Metalcast-316L": 8.00,
    "Fiberpart PC G20": 1.20,
    "Fiberpart PA G30": 1.15,
    "Enduse TPU D60": 1.20,
    "Sealant TPU A90": 1.20,
    "Sealant TPU A70": 1.20,
    "Fiberpart PA CF30": 1.15,
    "Другой материал": 1.00,
}

# Цена (₽/г)
PRICE_PER_GRAM = {
    "Sealant TPU93": 3.75,
    "Fiberpart ABS G4": 3.07,
    "Fiberpart TPU C5": 5.6,
    "Fiberpart ABSPA G8": 4.0,
    "Fiberpart TPU G30": 4.0,
    "Enduse SBS": 2.4,
    "Enduse ABS": 2.4,
    "Enduse PETG": 2.33,
    "Enduse PP": 8.08,
    "Proto PLA": 3.07,
    "ContiFiber CPA": 200.0,
    "Sealant SEBS": 5.2,
    "Sealant TPU": 5.98,
    "Proto PVA": 9.5,
    "Enduse-PA": 3.0,
    "Enduse-TPU D70": 3.99,
    "Fiberpart ABS G13": 6.65,
    "Fiberpart PP G": 3.3,
    "Fiberpart PP G30": 3.31,
    "Enduse PC": 0.0,
    "Fiberpart PA12 G12": 7.24,
    "Metalcast-316L": 16.0,
    "Fiberpart PC G20": 2.6,
    "Fiberpart PA G30": 7.9,
    "Enduse TPU D60": 1.9,
    "Sealant TPU A90": 2.6,
    "Sealant TPU A70": 6.4,
    "Fiberpart PA CF30": 10.4,
    "Другой материал": 2.0,
}

# Технологические параметры FDM (как было)
wall_count = 2
wall_width = 0.4
layer_height = 0.2
top_bottom_layers = 4

# =========================
# Геометрия: объёмы и площади (векторизация)
# =========================

def volume_tetra_units(V: np.ndarray, T: np.ndarray) -> float:
    """Объём в куб. МОДЕЛЬНЫХ единицах. Используется для базового объёма сетки."""
    if V.size == 0 or T.size == 0:
        return 0.0
    v0 = V[T[:, 0]]; v1 = V[T[:, 1]]; v2 = V[T[:, 2]]
    cross = np.cross(v1, v2)
    vol6 = np.einsum('ij,ij->i', v0, cross)
    return abs(vol6.sum()) / 6.0

def volume_tetra(V_mm: np.ndarray, T: np.ndarray) -> float:
    """Объём в см³. V в мм."""
    if V_mm.size == 0 or T.size == 0:
        return 0.0
    v0 = V_mm[T[:, 0]]; v1 = V_mm[T[:, 1]]; v2 = V_mm[T[:, 2]]
    cross = np.cross(v1, v2)
    vol6 = np.einsum('ij,ij->i', v0, cross)
    vol_mm3 = abs(vol6.sum()) / 6.0
    return vol_mm3 / 1000.0

def surface_area_mesh(V_mm: np.ndarray, T: np.ndarray) -> float:
    """Площадь поверхности в см². V в мм."""
    if V_mm.size == 0 or T.size == 0:
        return 0.0
    v0 = V_mm[T[:, 0]]; v1 = V_mm[T[:, 1]]; v2 = V_mm[T[:, 2]]
    area_mm2 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
    return area_mm2 / 100.0

def xy_area_bbox_from_V(V_mm: np.ndarray) -> float:
    """Площадь проекции bbox на XY в см². V в мм."""
    if V_mm.size == 0:
        return 0.0
    mins = V_mm.min(axis=0); maxs = V_mm.max(axis=0)
    dx, dy = (maxs[0] - mins[0]), (maxs[1] - mins[1])
    return (dx * dy) / 100.0

def volume_bbox(V_mm: np.ndarray) -> float:
    """Объём bbox в см³. V в мм."""
    if V_mm.size == 0:
        return 0.0
    mins = V_mm.min(axis=0); maxs = V_mm.max(axis=0)
    dx, dy, dz = (maxs - mins)
    return (dx * dy * dz) / 1000.0

# =========================
# 3MF: единицы, матрицы и кэш моделей
# =========================

def _unit_to_mm(unit_str: str) -> float:
    """Перевод единиц 3MF в миллиметры (мм на 1 модельную единицу)."""
    unit = (unit_str or 'millimeter').strip().lower()
    return {
        'micron': 0.001,
        'millimeter': 1.0,
        'centimeter': 10.0,
        'meter': 1000.0,
        'inch': 25.4,
        'foot': 304.8,
    }.get(unit, 1.0)

def _parse_transform(s: str | None) -> np.ndarray:
    """
    3MF transform из 12 чисел, ПО СТРОКАМ (row‑major):
      m00 m01 m02   m10 m11 m12   m20 m21 m22   m30 m31 m32
    Перенос — ПОСЛЕДНЯЯ СТРОКА (m30 m31 m32). Возвращаем 4×4 row‑major.
    """
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
    """Применение 4×4 (row‑major): v' = v @ R + t, где R = M[:3,:3], t = M[3,:3]."""
    R = M[:3, :3]
    t = M[3, :3]
    return V_mm @ R + t

def _detect_and_set_namespace(root: ET.Element):
    """Определить namespace из корня <model> и записать в глобальный NAMESPACE для корректного XPath."""
    try:
        if root.tag.startswith('{') and '}model' in root.tag:
            ns_uri = root.tag[1:].split('}')[0]
            NAMESPACE['ns'] = ns_uri
    except Exception:
        pass

def _gather_model_mm(root: ET.Element, unit_scale_mm: float, model_path: str):
    """
    Считать один .model:
      - вершины → ММ, треугольники T
      - базовый объём сетки в мм³ (без внешних трансформаций)
      - компоненты с p:path: (child_model_path, child_objectid, Mchild)
    """
    meshes_mm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    comps_map: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    base_vol_mm3: dict[str, float] = {}

    for obj in root.findall('.//ns:object', NAMESPACE):
        oid = obj.get('id')
        mesh = obj.find('ns:mesh', NAMESPACE)
        if mesh is not None:
            # Вершины в ММ: единая система для удобства p:path между моделями
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

            # Базовый объём (мм³): считаем в модельных единицах и переводим в мм³
            base_vol_mm3[oid] = volume_tetra_units(V_mm / max(unit_scale_mm, 1e-12), T) * (unit_scale_mm ** 3)
        else:
            comp_list: list[tuple[str, str, np.ndarray]] = []
            comps_node = obj.find('ns:components', NAMESPACE)
            if comps_node is not None:
                for c in comps_node.findall('ns:component', NAMESPACE):
                    ref = c.get('objectid')
                    M = _parse_transform(c.get('transform'))
                    # Production extension: компонент может ссылаться на другой .model
                    p_path = c.get(f'{{{NS_PROD}}}path') or c.get('path')
                    child_model = p_path.lstrip('/') if p_path else model_path
                    comp_list.append((child_model, ref, M))
            comps_map[oid] = comp_list

    return meshes_mm, comps_map, base_vol_mm3

def _build_model_cache(zf: zipfile.ZipFile):
    """Собрать кэш по всем 3D/*.model: вершины (мм), компоненты и базовые объёмы. Обновляет last_status."""
    cache = {}
    model_files = [f for f in zf.namelist() if f.startswith('3D/') and f.endswith('.model')]
    # Сброс статус‑плашки
    last_status.update({"unit_set": set(), "item_count": 0, "component_count": 0, "external_p_path": 0, "det_values": []})
    for mf in model_files:
        root = ET.fromstring(zf.read(mf))
        _detect_and_set_namespace(root)
        unit_scale = _unit_to_mm(root.get('unit'))
        last_status["unit_set"].add(root.get('unit') or 'millimeter')
        meshes_mm, comps_map, base_vol_mm3 = _gather_model_mm(root, unit_scale, mf)
        # Статистика
        last_status["component_count"] += sum(len(v) for v in comps_map.values())
        for lst in comps_map.values():
            for child_model, _, _ in lst:
                if child_model != mf:
                    last_status["external_p_path"] += 1
        cache[mf] = {
            'unit_scale_mm': unit_scale,
            'meshes_mm': meshes_mm,
            'comps': comps_map,
            'base_vol_mm3': base_vol_mm3,
            'root': root,
        }
    return cache

def _flatten_object_cached(cache: dict, model_file: str, oid: str, cum_M: np.ndarray):
    """
    Разворачивает объект oid из model_file, применяя ПОЛНУЮ матрицу cum_M к листовым сеткам.
    Соглашение row‑major (v' = v @ M): накапливаем справа ⇒ cum_next = cum_M @ Mchild.
    Быстрый объём: базовый объём мм³ × |det(cum_M[:3,:3])|.
    Возврат: (V_mm, T, vol_mm3_fast)
    """
    entry = cache.get(model_file)
    if entry is None:
        print(f"[3MF] WARN: model '{model_file}' not in cache")
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32), 0.0

    meshes_mm = entry['meshes_mm']
    comps     = entry['comps']
    base_vol  = entry['base_vol_mm3']

    # Лист: непосредственная сетка
    if oid in meshes_mm:
        V_mm, T = meshes_mm[oid]
        if V_mm.size == 0 or T.size == 0:
            return (np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.int32),
                    0.0)
        Vt = _apply_transform(V_mm, cum_M)
        det_full = abs(np.linalg.det(cum_M[:3, :3]))
        vol_mm3_fast = base_vol.get(oid, 0.0) * det_full
        return Vt, T.copy(), vol_mm3_fast

    # Составной объект: аккумулируем детей
    out_V = []
    out_T = []
    offset = 0
    vol_mm3_fast = 0.0
    for child_model, child_oid, Mchild in comps.get(oid, []):
        cum_next = cum_M @ Mchild  # item → component → …
        Vc, Tc, vc = _flatten_object_cached(cache, child_model, child_oid, cum_next)
        if Vc.size == 0 or Tc.size == 0:
            vol_mm3_fast += vc
            continue
        out_V.append(Vc)
        out_T.append(Tc + offset)
        offset += Vc.shape[0]
        vol_mm3_fast += vc

    if out_V:
        V = np.vstack(out_V)
        T = np.vstack(out_T)
        return V, T, vol_mm3_fast

    # Пусто
    return (np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
            vol_mm3_fast)

# =========================
# Парсеры форматов
# =========================

def parse_3mf(path: str):
    """
    Чтение .3mf c приоритетом сборок <build><item>.
    - Если найдены item'ы — используем их (масштаб/позиции как в слайсере).
    - Если item'ов нет — считаем каждый object как отдельную деталь.
    """
    data = []
    last_status["file"] = os.path.basename(path)
    with zipfile.ZipFile(path) as z:
        cache = _build_model_cache(z)

        # Ищем модели с build/items
        model_files = list(cache.keys())
        item_models = []
        items_per_model = {}
        for mf in model_files:
            root = cache[mf]['root']
            _detect_and_set_namespace(root)
            build = root.find('ns:build', NAMESPACE)
            items = [] if build is None else build.findall('ns:item', NAMESPACE)
            items_per_model[mf] = items
            if items:
                item_models.append(mf)
                last_status["item_count"] += len(items)

        selected = item_models if item_models else model_files

        for mf in selected:
            items = items_per_model.get(mf, [])

            if not items:
                # Нет сборки — считаем каждый object «как есть»
                all_ids = set(cache[mf]['meshes_mm'].keys()) | set(cache[mf]['comps'].keys())
                for oid in all_ids:
                    V_mm, T, vol_mm3_fast = _flatten_object_cached(cache, mf, oid, np.eye(4))
                    if V_mm.size == 0 and vol_mm3_fast == 0.0:
                        continue
                    name = f"{os.path.basename(mf)}:object_{oid}"
                    data.append((name, V_mm, T, vol_mm3_fast / 1000.0, {'type': '3mf', 'path': path}))
                continue

            # Есть сборка — применяем трансформации item'ов
            for idx, item in enumerate(items, 1):
                oid = item.get('objectid')
                Mitem = _parse_transform(item.get('transform'))
                det_dbg = float(np.linalg.det(Mitem[:3, :3]))
                last_status["det_values"].append(det_dbg)

                V_mm, T, vol_mm3_fast = _flatten_object_cached(cache, mf, oid, Mitem)
                if V_mm.size == 0 and vol_mm3_fast == 0.0:
                    continue
                name = f"{os.path.basename(mf)}:item_{idx}"
                data.append((name, V_mm, T, vol_mm3_fast / 1000.0, {'type': '3mf', 'path': path}))

    return data

def stl_stream_volume_cm3(path: str) -> float:
    """Потоковый объём STL в см³, читая бинарный STL напрямую (быстро, без построения меша)."""
    with open(path, 'rb') as f:
        f.seek(80)
        count = struct.unpack('<I', f.read(4))[0]
        total6 = 0.0
        for _ in range(count):
            f.read(12)  # нормаль (игнорируем)
            v0 = struct.unpack('<fff', f.read(12))
            v1 = struct.unpack('<fff', f.read(12))
            v2 = struct.unpack('<fff', f.read(12))
            # вклад в 6*V
            cx = (v1[1]*v2[2] - v1[2]*v2[1])
            cy = (v1[2]*v2[0] - v1[0]*v2[2])
            cz = (v1[0]*v2[1] - v1[1]*v2[0])
            total6 += v0[0]*cx + v0[1]*cy + v0[2]*cz
            f.read(2)  # атрибуты
    vol_mm3 = abs(total6) / 6.0
    return vol_mm3 / 1000.0

def parse_stl(path: str):
    """Минимальный парсер бинарного STL: построение топологии и быстрый объём по тетраэдрам."""
    verts, tris, idx_map = [], [], {}
    with open(path, 'rb') as f:
        f.seek(80)
        count = struct.unpack('<I', f.read(4))[0]
        for _ in range(count):
            f.read(12)  # нормаль
            face = []
            for _ in range(3):
                xyz = struct.unpack('<fff', f.read(12))
                if xyz not in idx_map:
                    idx_map[xyz] = len(verts)
                    verts.append(xyz)
                face.append(idx_map[xyz])
            tris.append(tuple(face))
            f.read(2)  # атрибуты
    V = np.array(verts, dtype=np.float64)
    T = np.array(tris, dtype=np.int32)
    vol_fast_cm3 = volume_tetra(V, T)
    return [("STL model", V, T, vol_fast_cm3, {'type': 'stl', 'path': path})]

def parse_geometry(path: str):
    """Диспетчер форматов: .3mf → parse_3mf, .stl → parse_stl."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.3mf':
        return parse_3mf(path)
    if ext == '.stl':
        return parse_stl(path)
    raise ValueError('Only .3mf and .stl supported')

# =========================
# UI и расчёт стоимости
# =========================

def _status_block_text() -> str:
    """Формирует текст статус‑плашки последнего парсинга (для доверия и быстрой диагностики)."""
    dets = last_status.get("det_values") or []
    det_min = f"{min(dets):.3f}" if dets else "—"
    det_max = f"{max(dets):.3f}" if dets else "—"
    units = ", ".join(sorted(last_status.get("unit_set") or [])) or "millimeter"
    return (
        f"Файл: {last_status.get('file','')}\n"
        f"Единицы (из моделей): {units}\n"
        f"Items: {last_status.get('item_count',0)} | Components: {last_status.get('component_count',0)} | p:path внешних: {last_status.get('external_p_path',0)}\n"
        f"det по items: min={det_min}, max={det_max}\n"
        "----------------------------------------\n"
    )

def recalc(*args):
    """
    Главная функция расчёта.
    - Для 3MF объём берём из vol_fast_cm3 (он уже учёл все масштабы/позиции).
    - Для STL при желании используем потоковый объём из файла.
    - Поверх базового объёма считаем вклад стенок/крышек/заполнения (FDM‑логика).
    """
    t0 = time.time()
    output.config(state='normal')
    output.delete('1.0', tk.END)
    if not loaded:
        output.insert(tk.END, 'Сначала загрузите модель.')
        output.config(state='disabled')
        return

    try:
        infill = float(entry_infill.get())
    except ValueError:
        messagebox.showerror('Ошибка', 'Введите корректный % заполнения.')
        output.config(state='disabled')
        return

    material = selected_material.get()
    density = MATERIALS[material]            # г/см³
    price = PRICE_PER_GRAM[material]         # ₽/г

    mode_val = mode.get()                    # 'bbox' или 'tetra'
    fast_only = fast_volume_var.get() == 1   # без стенок/крышек — только заполнение
    stream_stl = stream_stl_var.get() == 1   # потоковый STL

    # Статус‑плашка
    output.insert(tk.END, _status_block_text())

    total_weight = 0.0
    total_cost = 0.0
    lines = []
    for idx, (name, V, T, vol_fast_cm3, src) in enumerate(loaded, start=1):
        # 1) Базовый объём детали
        if src.get('type') == '3mf' and vol_fast_cm3 > 0:
            base_volume_cm3 = vol_fast_cm3
        else:
            if fast_only and src.get('type') == 'stl' and stream_stl and src.get('path'):
                try:
                    base_volume_cm3 = stl_stream_volume_cm3(src['path'])
                except Exception:
                    base_volume_cm3 = volume_bbox(V) if mode_val == 'bbox' else volume_tetra(V, T)
            else:
                base_volume_cm3 = volume_bbox(V) if mode_val == 'bbox' else volume_tetra(V, T)

        # 2) FDM‑разбор поверх базового объёма
        if fast_only:
            V_model = base_volume_cm3
            V_total = V_model * (infill / 100.0)
        else:
            V_model = base_volume_cm3
            shell_area = surface_area_mesh(V, T)  # см²
            xy_area = xy_area_bbox_from_V(V)      # см²
            # Корка стенок (см³): площадь * ширина * слоёв (мм → см через /10)
            V_shell = shell_area * wall_count * wall_width / 10.0
            # Крышки/донышки (см³): площадь XY * кол-во слоёв (мм → см через /10)
            V_top_bottom = xy_area * top_bottom_layers * layer_height / 10.0
            shell_total = V_shell + V_top_bottom
            # Ограничение: корка не более 60% объёма модели (практическая эвристика)
            if shell_total > V_model * 0.6:
                scale = (V_model * 0.6) / max(shell_total, 1e-12)
                V_shell *= scale
                V_top_bottom *= scale
            V_infill = max(0.0, V_model - V_shell - V_top_bottom) * (infill / 100.0)
            V_total = V_shell + V_top_bottom + V_infill

        # 3) Масса и стоимость
        weight = V_total * density      # граммы
        cost = weight * price           # рубли

        total_weight += weight
        total_cost += cost

        lines.append(f'Объект {idx}: {os.path.basename(name)}\n')
        lines.append(f'  Объём модели: {V_model:.2f} см³' + ('  [FAST]\n' if fast_only else '\n'))
        lines.append(f'  Вес: {weight:.2f} г\n')
        lines.append(f'  Стоимость: {cost:.2f} руб.\n')
        lines.append('')

    dt = time.time() - t0
    lines.append('----------------------------------------\n')
    lines.append(f'ИТОГО: вес {total_weight:.2f} г | стоимость {total_cost:.2f} руб.\n')
    lines.append(f'Время расчёта: {dt:.4f} с')

    output.insert(tk.END, ''.join(lines))
    output.config(state='disabled')

def open_file():
    """Диалог выбора файла и парсинг геометрии."""
    path = filedialog.askopenfilename(filetypes=[('3D Files', '*.3mf *.stl')])
    if not path:
        return
    try:
        objs = parse_geometry(path)
    except Exception as e:
        messagebox.showerror('Ошибка', str(e))
        return
    loaded.clear(); loaded.extend(objs)
    recalc()

# =========================
# GUI (без 3D‑рендера)
# =========================
root = tk.Tk()
root.title('3D калькулятор (PETG, FDM)')
root.geometry('540x640')
frame = tk.Frame(root, bg='#f9f9f9', padx=20, pady=20)
frame.pack(fill='both', expand=True)

tk.Label(frame, text='3D Калькулятор (.3mf / .stl)', font=('Arial', 20, 'bold'),
         bg='#f9f9f9').pack(pady=(0, 10))

# Метод базового объёма (для STL/фолбэка)
mode = tk.StringVar(value='tetra')
tk.Radiobutton(frame, text='Ограничивающий параллелепипед',
               variable=mode, value='bbox', font=('Arial', 14),
               bg='#f9f9f9', fg='#7A6EB0',
               command=lambda: recalc()).pack(anchor='w')
tk.Radiobutton(frame, text='Тетраэдры',
               variable=mode, value='tetra', font=('Arial', 14),
               bg='#f9f9f9', fg='#7A6EB0',
               command=lambda: recalc()).pack(anchor='w')

# Быстрые режимы
fast_frame = tk.Frame(frame, bg='#f9f9f9'); fast_frame.pack(pady=(6, 6), fill='x')
fast_volume_var = tk.IntVar(value=0)
stream_stl_var = tk.IntVar(value=0)
tk.Checkbutton(fast_frame, text='Быстрый объём (без стенок/крышек)',
               variable=fast_volume_var, bg='#f9f9f9',
               command=lambda: recalc()).pack(anchor='w')
tk.Checkbutton(fast_frame, text='Потоковый STL (объём напрямую из файла)',
               variable=stream_stl_var, bg='#f9f9f9',
               command=lambda: recalc()).pack(anchor='w')

# Материал и заполнение
material_frame = tk.Frame(frame, bg='#f9f9f9'); material_frame.pack(pady=(5, 5), fill='x')
tk.Label(material_frame, text='Материал:', font=('Arial', 12), bg='#f9f9f9').pack(side='left')
selected_material = tk.StringVar(value='Enduse PETG')
material_menu = tk.OptionMenu(material_frame, selected_material, *MATERIALS.keys())
material_menu.config(font=('Arial', 12), bg='white')
material_menu.pack(side='left', padx=5)
selected_material.trace_add('write', recalc)

infill_frame = tk.Frame(frame, bg='#f9f9f9'); infill_frame.pack(pady=(5, 10), fill='x')
tk.Label(infill_frame, text='Заполнение (%):', font=('Arial', 12), bg='#f9f9f9').pack(side='left')
entry_infill = tk.Entry(infill_frame, width=6, font=('Arial', 12)); entry_infill.insert(0, '10')
entry_infill.pack(side='left', padx=5)
entry_infill.bind('<KeyRelease>', lambda e: recalc())

tk.Button(frame, text='Загрузить 3D файл', font=('Arial', 12, 'bold'),
          bg='#7A6EB0', fg='white', command=open_file).pack(pady=10, fill='x')

output = scrolledtext.ScrolledText(frame, font=('Consolas', 12), state='disabled', height=18)
output.pack(fill='both', expand=True, pady=5)

root.mainloop()
