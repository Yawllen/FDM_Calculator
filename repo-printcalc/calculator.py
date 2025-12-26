# -*- coding: utf-8 -*-
"""
calculator.py — UI оболочка (tkinter) для 3D калькулятора (FDM)

Вся бизнес-логика вынесена в core_calc.py.
Этот файл содержит только UI и привязку введённых параметров к вызовам ядра.
"""
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, scrolledtext
import os, json, time
from io import BytesIO

# Превью (thumbnail) использует Pillow + thumbnail_core (CPU-only). Если пакеты не установлены,
# UI продолжит работать без превью.
try:
    from PIL import Image, ImageTk  # type: ignore
    #import thumbnail_core as tcore
    _THUMB_OK = True
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore
    tcore = None  # type: ignore
    _THUMB_OK = False

import core_calc as core

class CalculatorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('3D калькулятор (FDM) — красивый отчёт')
        self.root.geometry('740x900')

        # --- Scrollable container (vertical) ---
        # Важно: это делает весь UI прокручиваемым, если окно меньше контента.
        container = tk.Frame(self.root, bg='#f9f9f9')
        container.pack(fill='both', expand=True)

        self._canvas = tk.Canvas(container, bg='#f9f9f9', highlightthickness=0)
        vbar = ttk.Scrollbar(container, orient='vertical', command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)

        self._content = tk.Frame(self._canvas, bg='#f9f9f9')
        self._content_id = self._canvas.create_window((0, 0), window=self._content, anchor='nw')

        def _on_frame_configure(_evt=None):
            self._canvas.configure(scrollregion=self._canvas.bbox('all'))

        def _on_canvas_configure(evt):
            # растягиваем внутренний фрейм по ширине canvas
            self._canvas.itemconfigure(self._content_id, width=evt.width)

        self._content.bind('<Configure>', _on_frame_configure)
        self._canvas.bind('<Configure>', _on_canvas_configure)

        # Скролл колесом мыши (Windows)
        def _on_mousewheel(evt):
            # evt.delta кратен 120
            self._canvas.yview_scroll(int(-1 * (evt.delta / 120)), 'units')

        # Привязываем к canvas и корню, чтобы работало почти везде
        self._canvas.bind_all('<MouseWheel>', _on_mousewheel)

        self.loaded = []
        self.last_model_path = None  # последний загруженный файл (для превью/повтора)
        self.materials_density = dict(core.DEFAULT_MATERIALS_DENSITY)
        self.price_per_gram    = dict(core.DEFAULT_PRICE_PER_GRAM)
        self.pricing           = json.loads(json.dumps(core.DEFAULT_PRICING))

        # UI-переменные
        self.mode = tk.StringVar(value='tetra')
        self.fast_volume_var = tk.IntVar(value=0)
        self.stream_stl_var = tk.IntVar(value=0)
        self.selected_material = tk.StringVar(value=next(iter(self.materials_density.keys())))
        # Количество одинаковых экземпляров (тираж). Цена считается за заказ.
        self.qty_var = tk.IntVar(value=1)
        self.brief_var = tk.IntVar(value=1)
        self.diag_var  = tk.IntVar(value=0)

        self.entry_infill = None; self.entry_setup = None; self.entry_post = None; self.output = None
        # Превью (thumbnail) — ссылка на PhotoImage должна храниться, иначе Tk её сборкой удалит.
        self.preview_label = None
        self.preview_photo = None

        self.try_autoload_json()
        self.build_ui(self._content)

    # ---- JSON ----
    
    def try_autoload_json(self) -> None:
        """
        Автозагрузка материалов/конфига.
        Логика:
          1) Если рядом с текущим рабочим каталогом есть materials.json/pricing.json — используем их (удобно для отладки).
          2) Иначе пробуем рядом с core_calc.py (детерминированно для деплоя/пакета).
        Ошибки не фатальны — UI продолжает работать с DEFAULT_*.
        """
        # materials.json
        try:
            candidates = [
                os.path.join(os.getcwd(), "materials.json"),
                core.get_default_materials_path(),
            ]
            for path in candidates:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.materials_density.clear()
                    self.price_per_gram.clear()
                    for k, v in (data or {}).items():
                        self.materials_density[k] = float(v.get("density_g_cm3", 1.2))
                        self.price_per_gram[k] = float(v.get("price_rub_per_g", 0.0))
                    break
        except Exception as e:
            print("[WARN] materials.json:", e)

        # pricing.json
        try:
            candidates = [
                os.path.join(os.getcwd(), "pricing.json"),
                core.get_default_pricing_path(),
            ]
            for path in candidates:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    core.deep_merge(self.pricing, data)
                    break
        except Exception as e:
            print("[WARN] pricing.json:", e)


    def _deep_merge(self, dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                core.deep_merge(dst[k], v)
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
            core.deep_merge(self.pricing, data)
            messagebox.showinfo('OK', 'Конфиг ценообразования загружен')
            self.recalc()
        except Exception as e:
            messagebox.showerror('Ошибка JSON конфига', str(e))

    # ---- UI ----
    def build_ui(self, parent=None) -> None:
        parent = parent or self.root
        frame = tk.Frame(parent, bg='#f9f9f9', padx=20, pady=20); frame.pack(fill='both', expand=True)
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

        tk.Label(infill_frame, text='Количество (шт):', font=('Arial', 12), bg='#f9f9f9').pack(side='left', padx=(16, 0))
        qty_spin = tk.Spinbox(
            infill_frame,
            from_=1,
            to=100000,
            increment=1,
            textvariable=self.qty_var,
            width=7,
            font=('Arial', 12),
        )
        qty_spin.pack(side='left', padx=5)
        # Спинбокс даёт ввод цифр, но пользователь может напечатать руками — поэтому всё равно валидируем в recalc.
        qty_spin.bind('<KeyRelease>', lambda _e: self.recalc())
        qty_spin.bind('<<Increment>>', lambda _e: self.recalc())
        qty_spin.bind('<<Decrement>>', lambda _e: self.recalc())

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

        # --- Превью модели (CPU thumbnail) ---
        preview_wrap = tk.Frame(frame, bg='#f9f9f9')
        preview_wrap.pack(fill='x', pady=(0, 8))
        tk.Label(preview_wrap, text='Превью модели:', font=('Arial', 12), bg='#f9f9f9').pack(anchor='w')
        self.preview_label = tk.Label(preview_wrap, bg='#f9f9f9')
        self.preview_label.pack(anchor='center', pady=(4, 0))


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
            objs = core.parse_geometry(path)
        except Exception as e:
            messagebox.showerror('Ошибка', str(e)); return
        # не накапливаем — заменяем целиком
        self.loaded = list(objs)

        # запомним путь и обновим UI: превью + расчёт
        self.last_model_path = path
        #self.update_preview(path)
        self.recalc()

    def update_preview(self, path: str) -> None:
        """Генерирует и отображает превью для STL/3MF. Не влияет на расчёт, только UI."""
        if not self.preview_label:
            return
        if not _THUMB_OK:
            self.preview_label.configure(image='', text='(превью: нет зависимостей)', fg='#aa0000')
            return
        try:
            res = tcore.generate_thumbnail_bytes(path)
            img = Image.open(BytesIO(res.png_bytes))
            # На всякий случай приводим к заданному размеру (модуль уже отдаёт size_px×size_px).
            img = img.convert("RGBA")
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_photo)
            # Подсказка статусом (можно убрать, но полезно в отладке)
            self.preview_label.configure(text=f"{res.status} ({res.meta.get('mode')})", compound='bottom', fg='#555555', font=('Arial', 9))
        except Exception as e:
            # Не падаем из-за превью — просто очищаем.
            self.preview_photo = None
            self.preview_label.configure(image='', text='(превью недоступно)', fg='#aa0000')
        self.recalc()

    # ---- Вспомогательные блоки ----
    def _status_block_text(self) -> str:
        """Короткий блок диагностики (без падений, даже если статуса нет)."""
        st = getattr(core, "_last_status", {}) or {}
        lines = []
        f = st.get("file")
        if f:
            lines.append(f"[Файл] {f}")
        if st.get("type"):
            lines.append(f"[Тип] {st.get('type')}")
        if st.get("note"):
            lines.append(f"[Заметка] {st.get('note')}")
        warn = st.get("warnings")
        if warn:
            if isinstance(warn, (list, tuple)):
                for w in warn:
                    lines.append(f"[WARN] {w}")
            else:
                lines.append(f"[WARN] {warn}")
        # Для отладки геометрии (если есть)
        if st.get("verts") is not None:
            lines.append(f"[Verts] {st.get('verts')}")
        if st.get("tris") is not None:
            lines.append(f"[Tris] {st.get('tris')}")
        return ("\n".join(lines) + "\n") if lines else ""

    def recalc(self) -> None:
        t0 = time.time()
        self.output.config(state='normal'); self.output.delete('1.0', tk.END)
        if not self.loaded:
            self.output.insert(tk.END, 'Сначала загрузите модель.'); self.output.config(state='disabled'); return

        # --- Валидация ввода (мягко, без всплывающих окон при каждом символе) ---
        try:
            infill = float(str(self.entry_infill.get()).strip())
        except Exception:
            self.output.insert(tk.END, "Ошибка ввода: Заполнение (%) должно быть числом.\n")
            self.output.config(state='disabled')
            return
        infill = max(0.0, min(100.0, infill))

        try:
            setup_min = float(str(self.entry_setup.get()).strip())
            post_min = float(str(self.entry_post.get()).strip())
        except Exception:
            self.output.insert(tk.END, "Ошибка ввода: Подготовка/постпроцесс должны быть числами.\n")
            self.output.config(state='disabled')
            return
        setup_min = max(0.0, setup_min)
        post_min = max(0.0, post_min)

        # Тираж (qty) — всегда целое >= 1 (единая правда: core.coerce_qty)
        try:
            qty = core.coerce_qty(self.qty_var.get())
        except Exception as e:
            self.output.insert(tk.END, f"Ошибка ввода: количество (шт) некорректно: {e}\n")
            self.output.config(state='disabled')
            return

        material = self.selected_material.get()
        density = core.nz(self.materials_density.get(material), 1.2)
        price_g = core.nz(self.price_per_gram.get(material), 0.0)

        mode_val = self.mode.get()
        fast_only = self.fast_volume_var.get() == 1
        stream_stl = self.stream_stl_var.get() == 1
        if stream_stl and self.last_model_path and not core.is_stream_supported(self.last_model_path):
            self.stream_stl_var.set(0)
            messagebox.showerror(
                'Ошибка',
                'volume-mode=stream is supported only for STL files (binary STL will be validated from file)',
            )
            stream_stl = False

        diag_text = core.status_block_text() if (self.diag_var.get() == 1) else ""


        wall_count = 2; wall_width = 0.4; layer_height_geo = 0.2; top_bottom_layers = 4
        vol_factor = core.nz(self.pricing.get("geometry", {}).get("volume_factor"), 1.0)

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
                        V_model = core.stl_stream_volume_cm3(src['path'])
                    except Exception:
                        V_model = core.volume_bbox(V) if mode_val == 'bbox' else core.volume_tetra(V, T)
                else:
                    V_model = core.volume_bbox(V) if mode_val == 'bbox' else core.volume_tetra(V, T)

            V_total = core.compute_print_volume_cm3(
                V_model_cm3=V_model,
                V_mm=V,
                T=T,
                infill_pct=infill,
                fast_only=bool(fast_only),
                wall_count=wall_count,
                wall_width=wall_width,
                layer_height_geo=layer_height_geo,
                top_bottom_layers=top_bottom_layers,
            )
            V_total *= vol_factor  # калибровка

            weight_g = V_total * density
            mat_cost = weight_g * price_g

            total_V_model_cm3 += V_model
            total_V_total_cm3 += V_total
            total_weight_g += weight_g
            total_material_cost += mat_cost

        # Применяем тираж к переменным частям (материал/объём/вес/время).
        # setup_min и post_min — на заказ, поэтому НЕ умножаем.
        if qty != 1:
            total_V_model_cm3 *= qty
            total_V_total_cm3 *= qty
            total_weight_g *= qty
            total_material_cost *= qty

        bd = core.calc_breakdown(self.pricing, total_material_cost, total_V_total_cm3, setup_min, post_min)

        if len(self.loaded) == 1:
            obj_name = os.path.basename(self.loaded[0][0])
        else:
            obj_name = f"Сборка ({len(self.loaded)} объектов)"

        if qty != 1:
            obj_name = f"{obj_name} ×{qty}"

        report = core.render_report(
            file_name=(os.path.basename(self.last_model_path) if self.last_model_path else core._last_status.get("file","")),
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
            min_order_rub=core.nz(self.pricing.get("min_order_rub"), 0.0),
            min_applied=bd["min_applied"],
            markup_rub=bd["markup"],
            platform_fee_rub=bd["service_fee"],
            vat_rub=bd["vat"],
            total_rub=bd["total_with_min"],
            total_rounded_rub=bd["total"],
            rounding_step_rub=core.nz(self.pricing.get("rounding_to_rub"), 1.0),
            calc_time_s=time.time() - t0,
            brief=bool(self.brief_var.get() == 1),
            show_diag=bool(self.diag_var.get() == 1),
            diag_text=diag_text
        )

        # Доп. строка (цена всё равно за заказ) — полезно даже без "цены за штуку".
        if qty != 1:
            report = f"Количество: {qty} шт\n" + report

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
