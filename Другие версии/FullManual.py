# manual.py — продвинутый ручной калькулятор (FDM)
# Логика согласована с 3.py: словари MATERIALS и PRICE_PER_GRAM совпадают.

import tkinter as tk
from tkinter import messagebox

# ---------- Материалы и цены (взято из 3.py для консистентности) ----------
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

# ---------- Модель FDM (ручной режим, без 3D-мешей) ----------
def safe_float(entry, default=0.0):
    try:
        return float(entry.get().replace(',', '.'))
    except Exception:
        return float(default)

def safe_int(entry, default=0):
    try:
        return int(float(entry.get().replace(',', '.')))
    except Exception:
        return int(default)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def recalc(*_):
    # Ввод: габариты, масштаб, коэффициент формы
    dx = safe_float(entry_dx)      # мм
    dy = safe_float(entry_dy)      # мм
    dz = safe_float(entry_dz)      # мм
    scale = safe_float(entry_scale, 100.0) / 100.0  # %
    k_shape = clamp01(safe_float(entry_kshape, 1.0))  # 0..1

    dx *= scale; dy *= scale; dz *= scale

    if dx <= 0 or dy <= 0 or dz <= 0:
        result_var.set("Введите положительные размеры.")
        return

    # Базовый объём/площадь/периметр по BBox с поправкой формы:
    v_bbox_mm3 = dx * dy * dz * k_shape
    area_xy_mm2 = dx * dy * k_shape
    # Периметр масштабирем ~ sqrt(k) (эвристика для «подобной» формы)
    perim_xy_mm = 2.0 * (dx + dy) * (k_shape ** 0.5)

    # Параметры FDM
    infill = clamp01(safe_float(entry_infill, 10.0) / 100.0)      # доля
    walls = max(0, safe_int(entry_walls, 2))
    w = max(0.01, safe_float(entry_w, 0.4))                        # мм
    h = max(0.01, safe_float(entry_h, 0.2))                        # мм
    tb = max(0, safe_int(entry_tb, 4))                             # слоёв

    # Объём оболочки:
    # Стенки: на слой объем ~ периметр * w * h; слоёв ~ dz/h; учёт числа стенок → walls
    v_shell_walls_mm3 = walls * perim_xy_mm * w * dz
    # Крышки/донышки: площадь слоя * h * tb
    v_topbottom_mm3 = area_xy_mm2 * h * tb
    shell_total_mm3 = v_shell_walls_mm3 + v_topbottom_mm3

    # Ограничиваем долей от модельного объёма (эвристика из основного калькулятора)
    shell_cap = 0.60 * v_bbox_mm3
    if shell_total_mm3 > shell_cap:
        scale_shell = shell_cap / max(shell_total_mm3, 1e-12)
        v_shell_walls_mm3 *= scale_shell
        v_topbottom_mm3 *= scale_shell
        shell_total_mm3 = v_shell_walls_mm3 + v_topbottom_mm3

    # Заполнение
    v_infill_mm3 = max(0.0, v_bbox_mm3 - shell_total_mm3) * infill

    # Поддержки (как % от модельного объёма, задаётся пользователем)
    supports = clamp01(safe_float(entry_supports, 0.0) / 100.0)
    v_supports_mm3 = v_bbox_mm3 * supports

    # Итого объём экструдата
    v_total_mm3 = shell_total_mm3 + v_infill_mm3 + v_supports_mm3
    v_total_cm3 = v_total_mm3 / 1000.0

    # Материал → плотность/цена
    material = mat_var.get()
    density = MATERIALS.get(material, 1.0)          # г/см³
    ppg = PRICE_PER_GRAM.get(material, 2.0)         # ₽/г

    # Ручной тариф руб/см³ (если включен режим «ручная цена»)
    use_manual_rate = rate_mode.get() == 1
    if use_manual_rate:
        rate_cm3 = max(0.0, safe_float(entry_rate_cm3, 3.5))
        cost_material = v_total_cm3 * rate_cm3
        weight_g = v_total_cm3 * density
    else:
        weight_g = v_total_cm3 * density
        cost_material = weight_g * ppg

    # Время печати: по объёмному расходу (мм³/с)
    speed = max(1.0, safe_float(entry_speed, 100.0))       # мм/с (эффективная средняя)
    eff = clamp01(safe_float(entry_eff, 0.45))             # коэффициент эффективности траекторий
    flow_mm3_s = w * h * speed * eff
    time_s = v_total_mm3 / max(flow_mm3_s, 1e-9)

    # Себестоимость времени и электроэнергии (необязательные)
    rate_machine = max(0.0, safe_float(entry_rate_h, 0.0))     # ₽/ч
    power_w = max(0.0, safe_float(entry_power, 0.0))           # Вт
    tariff = max(0.0, safe_float(entry_tariff, 0.0))           # ₽/кВт·ч
    time_h = time_s / 3600.0
    cost_machine = rate_machine * time_h
    cost_electric = tariff * (power_w * time_h / 1000.0)

    # Коммерческие настройки
    markup = max(0.0, safe_float(entry_markup, 0.0)) / 100.0
    fixed_fee = max(0.0, safe_float(entry_fee, 0.0))
    min_price = max(0.0, safe_float(entry_min, 0.0))
    qty = max(1, safe_int(entry_qty, 1))

    # Итоговая стоимость за единицу и за партию
    base_cost_one = cost_material + cost_machine + cost_electric + fixed_fee
    final_cost_one = max(min_price, base_cost_one * (1.0 + markup))
    final_cost_total = final_cost_one * qty

    # Вывод
    lines = []
    lines.append(f"Габариты (мм): {dx:.2f} × {dy:.2f} × {dz:.2f} | k_формы={k_shape:.2f}")
    lines.append(f"Объём экструдата: {v_total_cm3:.2f} см³  (оболочка {shell_total_mm3/1000.0:.2f} см³, заполнение {(v_infill_mm3/1000.0):.2f} см³, поддержка {(v_supports_mm3/1000.0):.2f} см³)")
    lines.append(f"Материал: {material} | Плотность={density:.2f} г/см³ | Цена/г={ppg:.2f} ₽")
    if use_manual_rate:
        lines.append(f"Ручная цена: {rate_cm3:.2f} ₽/см³")
    lines.append(f"Вес: {weight_g:.2f} г")
    lines.append(f"Время печати: {int(time_h)} ч {int(time_s//60)%60} мин (скорость={speed:.0f} мм/с, w={w:.2f} мм, h={h:.2f} мм, eff={eff:.2f})")
    if rate_machine > 0 or power_w > 0:
        lines.append(f"Станок: {cost_machine:.2f} ₽ | Электроэнергия: {cost_electric:.2f} ₽")
    if fixed_fee > 0 or markup > 0 or min_price > 0:
        lines.append(f"Надбавка фикс: {fixed_fee:.2f} ₽ | Наценка: {markup*100:.0f}% | Мин. цена: {min_price:.2f} ₽")

    lines.append(f"ИТОГО за 1 шт: {final_cost_one:.2f} ₽")
    if qty > 1:
        lines.append(f"Количество: {qty} шт → ИТОГО: {final_cost_total:.2f} ₽")

    result_var.set("\n".join(lines))

# ---------- UI ----------
root = tk.Tk()
root.title("Ручной расчёт 3D (FDM)")
root.geometry("680x720")
root.configure(bg="#f9f9f9")

wrap = tk.Frame(root, padx=16, pady=12, bg="#f9f9f9")
wrap.pack(fill="both", expand=True)

title = tk.Label(wrap, text="Ручной калькулятор FDM", font=("Arial", 18, "bold"), bg="#f9f9f9")
title.pack(anchor="w", pady=(0, 8))

# Габариты и форма
box = tk.LabelFrame(wrap, text="Геометрия (мм)", bg="#f9f9f9", font=("Arial", 11, "bold"))
box.pack(fill="x", pady=6)

row1 = tk.Frame(box, bg="#f9f9f9"); row1.pack(fill="x", pady=2)
tk.Label(row1, text="DX", bg="#f9f9f9").pack(side="left"); entry_dx = tk.Entry(row1, width=8); entry_dx.pack(side="left", padx=6)
tk.Label(row1, text="DY", bg="#f9f9f9").pack(side="left"); entry_dy = tk.Entry(row1, width=8); entry_dy.pack(side="left", padx=6)
tk.Label(row1, text="DZ", bg="#f9f9f9").pack(side="left"); entry_dz = tk.Entry(row1, width=8); entry_dz.pack(side="left", padx=6)
tk.Label(row1, text="Масштаб, %", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_scale = tk.Entry(row1, width=6); entry_scale.insert(0, "100"); entry_scale.pack(side="left", padx=6)
tk.Label(row1, text="k формы (0..1)", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_kshape = tk.Entry(row1, width=6); entry_kshape.insert(0, "1.0"); entry_kshape.pack(side="left", padx=6)

# Профиль печати
prof = tk.LabelFrame(wrap, text="Профиль FDM", bg="#f9f9f9", font=("Arial", 11, "bold"))
prof.pack(fill="x", pady=6)

row2 = tk.Frame(prof, bg="#f9f9f9"); row2.pack(fill="x", pady=2)
tk.Label(row2, text="Заполнение, %", bg="#f9f9f9").pack(side="left"); entry_infill = tk.Entry(row2, width=6); entry_infill.insert(0, "10"); entry_infill.pack(side="left", padx=6)
tk.Label(row2, text="Стенок", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_walls = tk.Entry(row2, width=6); entry_walls.insert(0, "2"); entry_walls.pack(side="left", padx=6)
tk.Label(row2, text="Ширина линии, мм", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_w = tk.Entry(row2, width=6); entry_w.insert(0, "0.4"); entry_w.pack(side="left", padx=6)
tk.Label(row2, text="Высота слоя, мм", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_h = tk.Entry(row2, width=6); entry_h.insert(0, "0.2"); entry_h.pack(side="left", padx=6)
tk.Label(row2, text="Крышки/донышки, слоёв", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_tb = tk.Entry(row2, width=6); entry_tb.insert(0, "4"); entry_tb.pack(side="left", padx=6)

row2b = tk.Frame(prof, bg="#f9f9f9"); row2b.pack(fill="x", pady=2)
tk.Label(row2b, text="Поддержки, % от объёма модели", bg="#f9f9f9").pack(side="left"); entry_supports = tk.Entry(row2b, width=6); entry_supports.insert(0, "0"); entry_supports.pack(side="left", padx=6)

# Материал и режим тарифа
mat = tk.LabelFrame(wrap, text="Материал и тариф", bg="#f9f9f9", font=("Arial", 11, "bold"))
mat.pack(fill="x", pady=6)

row3 = tk.Frame(mat, bg="#f9f9f9"); row3.pack(fill="x", pady=2)
tk.Label(row3, text="Материал:", bg="#f9f9f9").pack(side="left")
mat_var = tk.StringVar(value="Enduse PETG")
opt = tk.OptionMenu(row3, mat_var, *MATERIALS.keys()); opt.config(bg="white", width=18)
opt.pack(side="left", padx=6)

rate_mode = tk.IntVar(value=0)  # 0 = цена/г из материала, 1 = ручная цена/см3
tk.Radiobutton(row3, text="Цена по материалу (₽/г)", variable=rate_mode, value=0, bg="#f9f9f9", command=recalc).pack(side="left", padx=(12,0))
tk.Radiobutton(row3, text="Ручная цена (₽/см³)", variable=rate_mode, value=1, bg="#f9f9f9", command=recalc).pack(side="left", padx=6)
entry_rate_cm3 = tk.Entry(row3, width=8); entry_rate_cm3.insert(0, "3.5"); entry_rate_cm3.pack(side="left", padx=6)

# Время/энергия/ставки
costs = tk.LabelFrame(wrap, text="Время и затраты", bg="#f9f9f9", font=("Arial", 11, "bold"))
costs.pack(fill="x", pady=6)

row4 = tk.Frame(costs, bg="#f9f9f9"); row4.pack(fill="x", pady=2)
tk.Label(row4, text="Скорость, мм/с", bg="#f9f9f9").pack(side="left"); entry_speed = tk.Entry(row4, width=6); entry_speed.insert(0, "100"); entry_speed.pack(side="left", padx=6)
tk.Label(row4, text="Эффективность (0..1)", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_eff = tk.Entry(row4, width=6); entry_eff.insert(0, "0.45"); entry_eff.pack(side="left", padx=6)
tk.Label(row4, text="Ставка машины, ₽/ч", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_rate_h = tk.Entry(row4, width=6); entry_rate_h.insert(0, "0"); entry_rate_h.pack(side="left", padx=6)

row4b = tk.Frame(costs, bg="#f9f9f9"); row4b.pack(fill="x", pady=2)
tk.Label(row4b, text="Мощность, Вт", bg="#f9f9f9").pack(side="left"); entry_power = tk.Entry(row4b, width=6); entry_power.insert(0, "0"); entry_power.pack(side="left", padx=6)
tk.Label(row4b, text="Тариф, ₽/кВт·ч", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_tariff = tk.Entry(row4b, width=6); entry_tariff.insert(0, "0"); entry_tariff.pack(side="left", padx=6)

# Коммерция
comm = tk.LabelFrame(wrap, text="Коммерческие настройки", bg="#f9f9f9", font=("Arial", 11, "bold"))
comm.pack(fill="x", pady=6)

row5 = tk.Frame(comm, bg="#f9f9f9"); row5.pack(fill="x", pady=2)
tk.Label(row5, text="Наценка, %", bg="#f9f9f9").pack(side="left"); entry_markup = tk.Entry(row5, width=6); entry_markup.insert(0, "0"); entry_markup.pack(side="left", padx=6)
tk.Label(row5, text="Фикс. надбавка, ₽", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_fee = tk.Entry(row5, width=8); entry_fee.insert(0, "0"); entry_fee.pack(side="left", padx=6)
tk.Label(row5, text="Мин. цена/шт, ₽", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_min = tk.Entry(row5, width=8); entry_min.insert(0, "0"); entry_min.pack(side="left", padx=6)
tk.Label(row5, text="Количество, шт", bg="#f9f9f9").pack(side="left", padx=(12,0)); entry_qty = tk.Entry(row5, width=6); entry_qty.insert(0, "1"); entry_qty.pack(side="left", padx=6)

# Кнопки/результат
btn = tk.Button(wrap, text="Рассчитать", font=("Arial", 12, "bold"), bg="#7A6EB0", fg="white", command=recalc)
btn.pack(fill="x", pady=8)

result_var = tk.StringVar(value="Заполните поля и нажмите «Рассчитать».")
res = tk.Label(wrap, textvariable=result_var, justify="left", anchor="w", font=("Consolas", 11), bg="#f9f9f9")
res.pack(fill="both", expand=True, pady=(4,0))

# Привязки для «живого» пересчёта
for e in []:
    e.bind("<KeyRelease>", recalc)

# Умолчания для быстрого старта
entry_dx.insert(0, "100")
entry_dy.insert(0, "50")
entry_dz.insert(0, "20")

# Автопересчёт при смене материала/режима
mat_var.trace_add("write", recalc)

root.mainloop()
