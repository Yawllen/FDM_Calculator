# manual.py — ручной расчёт 3D стоимости (оптимизированный, стиль сохранён)
import tkinter as tk
from tkinter import messagebox
from concurrent.futures import ThreadPoolExecutor
import re

# ---------- Пул для неблокирующих расчётов ----------
EXEC = ThreadPoolExecutor(max_workers=1)

# ---------- Быстрые утилиты ----------
_DECIMAL_RE = re.compile(r"^[0-9]*[.,]?[0-9]*$")  # простая валидация "числа"

def parse_float(s: str) -> float:
    """Быстрый и терпимый парсер: поддерживает запятую, пустые -> ошибка сверху."""
    return float(s.replace(",", ".").strip())

def valid_numeric(P: str) -> bool:
    """Tk vcmd: пропускаем пустое (для ввода) и простые числа вида 123,45."""
    return P == "" or bool(_DECIMAL_RE.match(P))

def compute(dx: float, dy: float, dz: float, rate: float):
    """Чистая функция расчёта (можно гонять в пуле)."""
    volume_cm3 = (dx * dy * dz) * 0.001  # мм³ → см³
    cost = volume_cm3 * rate
    return volume_cm3, cost

# ---------- Логика UI ----------
def on_calculate():
    """Чтение ввода + запуск расчёта в фоне."""
    try:
        dx = parse_float(entry_dx.get())
        dy = parse_float(entry_dy.get())
        dz = parse_float(entry_dz.get())
        rate = parse_float(entry_rate.get())
        if dx <= 0 or dy <= 0 or dz <= 0 or rate < 0:
            raise ValueError
    except Exception:
        messagebox.showerror("Ошибка", "Введите корректные положительные значения.")
        return

    # Блокируем кнопку на время расчёта (UI остаётся отзывчивым)
    btn.config(state="disabled", text="Считаю…")
    future = EXEC.submit(compute, dx, dy, dz, rate)
    future.add_done_callback(lambda f: root.after(0, lambda: on_done(f)))

def on_done(future):
    """Безопасно обновляем виджеты из главного потока."""
    try:
        volume_cm3, cost = future.result()
        result_label.config(
            text=f"Объём: {volume_cm3:.2f} см³\nСтоимость: {cost:.2f} руб."
        )
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось выполнить расчёт: {e}")
    finally:
        btn.config(state="normal", text="Рассчитать")

def on_close():
    EXEC.shutdown(wait=False, cancel_futures=True)
    root.destroy()

# ---------- UI (оставлен как в исходнике) ----------
root = tk.Tk()
root.title("Ручной расчёт 3D стоимости")
root.geometry("400x400")
root.configure(bg='#f9f9f9')
root.protocol("WM_DELETE_WINDOW", on_close)

frame = tk.Frame(root, padx=20, pady=20, bg='#f9f9f9')
frame.pack(expand=True, fill='both')

tk.Label(frame, text="Введите габариты в мм:", font=('Arial', 14), bg='#f9f9f9').pack(pady=(0, 10))

# Валидатор для числовых полей (ускоряет и снижает ошибки)
vcmd = (root.register(valid_numeric), "%P")

tk.Label(frame, text="Длина (DX)", font=('Arial', 12), bg='#f9f9f9').pack(anchor='w')
entry_dx = tk.Entry(frame, font=('Arial', 12), validate="key", validatecommand=vcmd)
entry_dx.pack(fill='x', pady=2)

tk.Label(frame, text="Ширина (DY)", font=('Arial', 12), bg='#f9f9f9').pack(anchor='w')
entry_dy = tk.Entry(frame, font=('Arial', 12), validate="key", validatecommand=vcmd)
entry_dy.pack(fill='x', pady=2)

tk.Label(frame, text="Высота (DZ)", font=('Arial', 12), bg='#f9f9f9').pack(anchor='w')
entry_dz = tk.Entry(frame, font=('Arial', 12), validate="key", validatecommand=vcmd)
entry_dz.pack(fill='x', pady=2)

tk.Label(frame, text="Цена (руб/см³):", font=('Arial', 12), bg='#f9f9f9').pack(anchor='w', pady=(10, 0))
entry_rate = tk.Entry(frame, font=('Arial', 12), validate="key", validatecommand=vcmd)
entry_rate.pack(fill='x', pady=5)
entry_rate.insert(0, "3.5")

btn = tk.Button(frame, text="Рассчитать", font=('Arial', 12, 'bold'), bg='#7A6EB0', fg='white', command=on_calculate)
btn.pack(pady=10, fill='x')

result_label = tk.Label(frame, text="Результат появится здесь", font=('Arial', 12), bg='#f9f9f9', justify="left")
result_label.pack(pady=(10, 0))

# Удобства: Enter запускает расчёт, базовые значения для быстрого теста
root.bind("<Return>", lambda e: on_calculate())
entry_dx.insert(0, "100")
entry_dy.insert(0, "50")
entry_dz.insert(0, "20")

root.mainloop()
