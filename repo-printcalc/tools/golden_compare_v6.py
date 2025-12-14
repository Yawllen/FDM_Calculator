# -*- coding: utf-8 -*-
"""
golden_compare_v6.py — расширенный golden test для CLI калькулятора (old vs refactored)
с максимально устойчивым вводом/выводом на Windows.

Что делает:
- Прогоняет сценарии (base/per-object/diag/workers + text smoke)
- Сравнивает числа из JSON:
  * base_json: payload["summary"] (dict)
  * per_object_json: payload["per_object"] (list), summary обычно null
- Текстовые режимы не сравнивает по строкам (smoke): "не упало и вывод не пустой"

Почему так:
- Windows консоль/родительский процесс может быть cp1251 → subprocess(text=True) часто ломается.
  Поэтому тут всегда subprocess(text=False) и ручной decode('utf-8','replace').
- CLI печатает логи до JSON → JSON извлекается по первой '{' и последней '}'.

Запуск:
  python golden_compare_v6.py
или
  python golden_compare_v6.py "file.3mf" "file.stl"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PY = sys.executable
OLD = "cli_calculator.py"
NEW = "cli_calculator.refactored.py"

TOL = 1e-6

# Ключи для сравнения summary (base_json)
SUMMARY_KEYS = [
    "material",
    "price_rub_per_g",
    "volume_model_cm3",
    "volume_print_cm3",
    "weight_g",
    "time_h",
    "total_rub",
    "total_raw_rub",
    "min_applied",
]

# Ключи для сравнения per_object элементов (per_object_json)
PER_OBJECT_KEYS = [
    "file",
    "material",
    "price_rub_per_g",
    "volume_model_cm3",
    "volume_print_cm3",
    "weight_g",
    "time_h",
    "subtotal_rub",
    "min_order_rub",
    "min_applied",
    "markup_rub",
    "platform_fee_rub",
    "vat_rub",
    "total_rub",
    "total_rounded_rub",
    "rounding_step_rub",
]

SCENARIOS: List[Tuple[str, List[str]]] = [
    ("base_json", ["--json"]),
    ("per_object_json", ["--per-object", "--json"]),
    ("diag_json", ["--diag", "--json"]),
    ("workers_json", ["--workers", "4", "--json"]),
    ("per_object_workers_json", ["--per-object", "--workers", "4", "--json"]),
    ("per_object_text", ["--per-object", "--text"]),
    ("diag_text", ["--diag", "--text"]),
]


# ------------------ helpers: IO / diagnostics ------------------

def _decode(b: Optional[bytes]) -> str:
    return (b or b"").decode("utf-8", "replace")


def _clip(s: str, head: int = 2000, tail: int = 2000) -> str:
    s = s or ""
    if len(s) <= head + tail + 100:
        return s
    return s[:head] + "\n...\n" + s[-tail:]


def _raise_proc_error(script: str, cmd: List[str], p: subprocess.CompletedProcess[bytes]) -> None:
    out = _decode(p.stdout)
    err = _decode(p.stderr)
    msg = (
        f"[{script}] exit={p.returncode}\n"
        f"CMD: {' '.join(cmd)}\n"
        f"--- STDOUT (clipped) ---\n{_clip(out)}\n"
        f"--- STDERR (clipped) ---\n{_clip(err)}\n"
    )
    raise RuntimeError(msg)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    CLI может печатать логи перед JSON. Достаём подстроку от первой '{' до последней '}'.
    """
    i = text.find("{")
    j = text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError(f"stdout не содержит JSON-блока.\nSTDOUT:\n{_clip(text)}")
    return json.loads(text[i:j+1])


def run_cli(script: str, model_path: str, material: str, infill: float, extra_args: List[str]) -> Dict[str, Any]:
    """
    Возвращает dict:
      - для --json: распарсенный JSON payload
      - для --text: {"success": True, "_text": "..."} (smoke)
    Никогда не возвращает None.
    """
    cmd = [PY, script, model_path, "--material", material, "--infill", str(infill), *extra_args]

    env = dict(os.environ)
    # на всякий случай: пусть дочерний python пишет в utf-8
    env["PYTHONIOENCODING"] = "utf-8"

    # Всегда bytes, чтобы не словить UnicodeDecodeError в родительском процессе
    p = subprocess.run(cmd, capture_output=True, text=False, env=env)

    if p.returncode != 0:
        _raise_proc_error(script, cmd, p)

    out = _decode(p.stdout)
    # err = _decode(p.stderr)  # можно подключить при отладке

    if "--json" in extra_args:
        try:
            payload = extract_json_from_text(out)
        except Exception as e:
            # максимально полезная диагностика
            raise RuntimeError(
                f"[{script}] JSON parse error: {e}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"--- STDOUT (clipped) ---\n{_clip(out)}\n"
            )
        if not isinstance(payload, dict):
            raise RuntimeError(f"[{script}] JSON payload is not a dict. type={type(payload)}")
        return payload

    # --text smoke
    return {"success": True, "_text": out}


# ------------------ helpers: compare ------------------

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))


def _eq(a: Any, b: Any) -> bool:
    if _is_num(a) and _is_num(b):
        return abs(float(a) - float(b)) <= TOL
    return a == b


def find_summary(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    s = payload.get("summary")
    return s if isinstance(s, dict) else None


def find_per_object(payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    po = payload.get("per_object")
    if isinstance(po, list):
        return [x for x in po if isinstance(x, dict)]
    return None


def compare_dict(a: Dict[str, Any], b: Dict[str, Any], keys: List[str], prefix: str) -> List[str]:
    diffs: List[str] = []
    for k in keys:
        if k not in a or k not in b:
            diffs.append(f"MISSING {prefix}.{k}: old_has={k in a} new_has={k in b}")
            continue
        if not _eq(a[k], b[k]):
            diffs.append(f"DIFF {prefix}.{k}: old={a[k]} new={b[k]}")
    return diffs


def compare_per_object(a_list: List[Dict[str, Any]], b_list: List[Dict[str, Any]]) -> List[str]:
    diffs: List[str] = []

    def oid(x: Dict[str, Any]) -> str:
        # у тебя есть "file" — используем его как стабильный id
        v = x.get("file")
        if isinstance(v, str) and v.strip():
            return v.strip()
        # fallback
        for k in ("obj_name", "name", "id"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "|".join(sorted(x.keys()))

    a_map = {oid(x): x for x in a_list}
    b_map = {oid(x): x for x in b_list}

    a_ids = set(a_map.keys())
    b_ids = set(b_map.keys())

    if a_ids != b_ids:
        diffs.append(
            "OBJECT SET DIFF: "
            f"missing_in_new={sorted(a_ids - b_ids)[:20]} "
            f"missing_in_old={sorted(b_ids - a_ids)[:20]}"
        )

    for id_ in sorted(a_ids & b_ids):
        diffs += compare_dict(a_map[id_], b_map[id_], PER_OBJECT_KEYS, f"per_object[{id_}]")

    return diffs


def smoke_text_ok(old_txt: Any, new_txt: Any) -> bool:
    return bool(str(old_txt or "").strip()) and bool(str(new_txt or "").strip())


# ------------------ main ------------------

def main() -> int:
    models = sys.argv[1:]
    if not models:
        here = Path(__file__).resolve().parent
        models = [str(p) for p in here.glob("*.3mf")] + [str(p) for p in here.glob("*.stl")]

    if not models:
        print("Не найдено моделей (*.3mf/*.stl) и не переданы аргументы.")
        return 2

    material = "Enduse PETG"
    infill = 10.0

    all_ok = True

    for model in models:
        mp = str(Path(model).resolve())
        print(f"\n=== MODEL: {Path(mp).name} ===")

        for scen_name, extra in SCENARIOS:
            print(f"  -> scenario: {scen_name}")
            old = run_cli(OLD, mp, material, infill, extra)
            new = run_cli(NEW, mp, material, infill, extra)

            # text smoke
            if "--json" not in extra:
                ok = smoke_text_ok(old.get("_text"), new.get("_text"))
                if ok:
                    print("     OK (text smoke)")
                else:
                    all_ok = False
                    print("     MISMATCH (text smoke): empty output")
                continue

            diffs: List[str] = []

            # summary сравнение (только если dict в обеих)
            osum = find_summary(old)
            nsum = find_summary(new)
            if osum is not None or nsum is not None:
                if osum is None or nsum is None:
                    diffs.append(f"SUMMARY presence differs: old={osum is not None} new={nsum is not None}")
                else:
                    diffs += compare_dict(osum, nsum, SUMMARY_KEYS, "summary")

            # per_object сравнение (если присутствует)
            opo = find_per_object(old)
            npo = find_per_object(new)
            if opo is not None or npo is not None:
                if opo is None or npo is None:
                    diffs.append(f"PER_OBJECT presence differs: old={opo is not None} new={npo is not None}")
                else:
                    diffs += compare_per_object(opo, npo)

            if diffs:
                all_ok = False
                print("     MISMATCH:")
                for d in diffs[:80]:
                    print("       -", d)
            else:
                print("     OK")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
