"""Checks of the PI-CAI Colab merge (scripts/merge_picai_colab.py)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import merge_picai_colab as m  # noqa: E402

SHA = "29021394e1c61b3f50420aa8315c3d43eca4e66f8a57398537c0a439361e50bd"
LOCAL = {"case": "10000_1000000", "archive": "picai_public_images_fold0.zip",
         "member": "10000/10000_1000000_t2w.mha", "shape": "[640, 640, 31]",
         "spacing": "[0.2812, 0.2812, 3.3]", "axcodes": "LPS",
         "corner_max_diff_mm": "7.786710582990963e-06", "status": "ok"}


def colab(**kw):
    r = {"case": "10000_1000000", "archive": "picai_public_images_fold0.zip",
         "member": "10000/10000_1000000_t2w.mha", "shape": [640, 640, 31],
         "spacing": [0.2812, 0.2812, 3.3], "axcodes": "LPS",
         "corner_max_diff_mm": 7.786710582990963e-06, "status": "ok"}
    r.update(kw)
    return r


def test_same_conversion_accepts_the_identical_record():
    assert m.same_conversion(colab(), LOCAL)


def test_same_conversion_catches_each_difference():
    for kw in ({"shape": [640, 640, 30]}, {"spacing": [0.2813, 0.2812, 3.3]}, {"axcodes": "RAS"},
               {"member": "10000/10000_1000000_adc.mha"}, {"corner_max_diff_mm": 1e-5}):
        assert not m.same_conversion(colab(**kw), LOCAL), kw


def test_fold_of():
    assert m.fold_of("picai_public_images_fold3.zip") == 3


def test_same_result_compares_the_volumes():
    a = {"shape": "640x640x31", "cg_ml": 17.0, "pz_ml": 11.0, "wg_ml": 28.0, "wg_ml_R1": 27.6,
         "empty": False, "geometry_ok": True, "seconds": 6.1}
    assert m.same_result(a, {**a, "seconds": 9.9})
    assert not m.same_result(a, {**a, "pz_ml": 11.001})


def test_every_fold_is_assigned_to_exactly_one_run():
    assert sorted(k for r in m.RUNS for k in r["folds"]) == [0, 1, 2, 3, 4]


def _nb(tmp_path, out_lines, sha=SHA):
    nb = {"cells": [
        {"source": [f"BUNDLE_NAME, BUNDLE_SHA256 = 'picai_psad_bundle.zip', '{sha}'\n"],
         "outputs": [{"text": out_lines}]}]}
    p = tmp_path / "run.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    return str(p)


def test_notebook_facts_reads_gpu_bundle_and_torch(tmp_path):
    f = m.notebook_facts(_nb(tmp_path, [
        "GPU 0: Tesla T4 (UUID: GPU-x)\n", "bundle OK: 29021394e1c61b3f\n",
        "labels ce4a472 | SimpleITK 2.4.1 | nibabel 5.4.2 | torch 2.14.1+cu130 | eligible 1440\n",
        "fold 2: done\n", "fold 3: done\n"]))
    assert f["gpu"] == "Tesla T4" and f["torch"] == "2.14.1+cu130"
    assert f["bundle_ok_printed"] and f["bundle_sha256"] == SHA
    assert f["folds_done_printed"] == [2, 3]


def test_notebook_facts_rejects_a_bundle_that_was_not_confirmed(tmp_path):
    f = m.notebook_facts(_nb(tmp_path, ["GPU 0: Tesla T4 (UUID: GPU-x)\n", "bundle OK: ffffffffffffffff\n"]))
    assert not f["bundle_ok_printed"]
