# FDM Calculator CLI

## Overview
The CLI computes print time and pricing for STL/3MF models using the exact same formulas as the UI.
It is a thin wrapper over `core_calc.py`, which owns geometry parsing, pricing rules, and report formatting.
Both the CLI and the UI load the same configuration files and call the shared core.
This means CLI output is always aligned with the UI’s calculation logic.

## Установка (Windows PowerShell)
```powershell
python -m pip install -r requirements-dev.txt
```

## CLI quickstart
```bash
python cli_calculator.py model.stl --material "PLA"
```

## Тесты (Windows PowerShell)
```powershell
pytest -q
```

## Usage

Single file (text output by default):
```bash
python cli_calculator.py model.stl --material "PLA" --infill 15
```

Batch (multiple files, JSON output):
```bash
python cli_calculator.py part_a.stl part_b.3mf --material "PLA" --json
```

With an explicit config directory:
```bash
python cli_calculator.py model.3mf --material "PLA" --config-dir ./config --json
```

## Configuration loading

Materials and pricing are loaded through `core_calc.py` from `materials.json` and `pricing.json`.

Priority order:
1. `--config-dir` (must contain both `materials.json` and `pricing.json`).
2. Defaults from `core_calc`: if no `--config-dir` is provided, the CLI first checks the current working directory **only when both files are present**; otherwise it falls back to the default files next to the CLI/core module.

Defaults are used whenever `--config-dir` is omitted **and** the current working directory does not contain both `materials.json` and `pricing.json`.

If you try to use `--materials` or `--pricing`, the CLI ignores them (they are present but not configurable).

## JSON output contract

Top-level fields that are always present when `--json` is used:

- `success` (boolean): `true` only when every file is processed without errors.
- `count_ok` (integer): number of files successfully processed.
- `count_failed` (integer): number of files that failed.
- `errors` (array): list of error objects with `file` and `error` strings.
- `results`: the calculation payload returned by the CLI (currently emitted at the top level, not nested under a `results` key). It includes the per-file list (`per_object`) or a summary (`summary`), along with counters like `count` and timing (`time_s`).

When `success=false`, at least one file failed; the CLI still returns results for any files that completed.

Example JSON output (single file, summary mode):
```json
{
  "success": true,
  "qty": 1,
  "count": 1,
  "per_object": null,
  "summary": {
    "qty": 1,
    "material": "PLA",
    "price_rub_per_g": 0.0,
    "volume_model_cm3": 12.34,
    "volume_print_cm3": 15.67,
    "weight_g": 18.8,
    "time_h": 0.42,
    "total_rub": 123.0,
    "total_raw_rub": 123.0,
    "min_applied": false
  },
  "time_s": 0.03,
  "errors": [],
  "count_failed": 0,
  "count_ok": 1
}
```

## Exit codes

- `0`: all files processed successfully.
- `1`: partial failures in batch or a processing error after parsing arguments.
- `2`: fatal error (invalid arguments, invalid `--set`, missing/invalid config, or unknown material).

## Error handling

- Broken or unreadable STL/3MF files are reported per file; the CLI captures the exception and continues processing other files.
- In batch mode, successful files still return results even if others fail.
- In JSON mode, errors are listed in the `errors` array and `success=false` when any file fails.
- In text mode, errors are printed to stderr (prefixed with `[cli] файл …`) while successful results are printed to stdout.
