import json

import cli_calculator as cli
import core_calc as core


def test_cli_and_core_config_loading_match(tmp_path):
    materials = {
        "Proto PLA": {
            "density_g_cm3": 1.24,
            "price_rub_per_g": 2.5,
        }
    }
    pricing = {
        "min_order_rub": 250,
        "rounding_to_rub": 1.0,
        "printing": {"a1": 1.0},
    }

    materials_path = tmp_path / "materials.json"
    pricing_path = tmp_path / "pricing.json"
    materials_path.write_text(json.dumps(materials, ensure_ascii=False), encoding="utf-8")
    pricing_path.write_text(json.dumps(pricing, ensure_ascii=False), encoding="utf-8")

    core_density, core_price = core.load_materials_json(str(materials_path))
    core_pricing = core.load_pricing_json(str(pricing_path))

    cli_density, cli_price, cli_pricing, _, _ = cli.load_configs_via_core(str(tmp_path))

    assert cli_density == core_density
    assert cli_price == core_price
    assert cli_pricing == core_pricing


def test_finalize_json_payload_with_errors():
    payload = {
        "success": True,
        "count": 2,
        "per_object": [{"file": "ok.stl"}],
        "summary": None,
    }
    errors = [{"file": "bad.stl", "error": "boom"}]

    out = cli.finalize_json_payload(payload, errors, count_ok=1)

    assert out["success"] is False
    assert out["count_failed"] == len(errors)
    assert out["count_ok"] == 1
    assert out["errors"][0]["file"] == "bad.stl"
    assert out["errors"][0]["error"] == "boom"
