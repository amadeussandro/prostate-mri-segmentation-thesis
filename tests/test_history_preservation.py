"""Regression tests for training-history preservation across a resume.

THE BUG THESE EXIST FOR
-----------------------
`run_training` used to initialize `epoch_history = []` unconditionally. A run
interrupted at epoch 56 and resumed therefore rewrote `metrics.csv` and
`training_history.json` with epochs 57..100 ONLY -- epochs 1..56 silently
disappeared, even though the model/optimizer resume itself was correct. On a
Colab T4 that reliably loses a session or two per 100-epoch run, this destroys
the training curve the thesis has to report.

The fix is `load_existing_history` + `upsert_epoch_record` in src/train.py:
history is read back from disk before the epoch loop, and every epoch is
written under its epoch number as a unique key.

These tests exercise that logic directly (no GPU, no dataset), plus the
whole-file `metrics.csv` rewrite that consumes it.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.train import (  # noqa: E402
    _atomic_json_dump,
    _history_from_metrics_csv,
    _write_metrics_csv,
    load_existing_history,
    upsert_epoch_record,
)


def make_record(epoch: int, val_dice: float = 0.5):
    return {
        "epoch": epoch,
        "train_loss": 1.0 / epoch,
        "train_loss_ce": 1.0 / epoch,
        "train_loss_dice": 0.0,
        "val_loss": 0.9 / epoch,
        "val_loss_ce": 0.9 / epoch,
        "val_loss_dice": 0.0,
        "val_dice": val_dice,
        "val_iou": val_dice - 0.1,
        "val_per_class_dice": {"1": val_dice + 0.05, "2": val_dice - 0.05},
        "learning_rate": 0.001,
        "epoch_seconds": 42.0,
        "amp": False,
    }


def write_history_json(path, records, **extra):
    payload = {"amp": False, "device": "cpu", "num_epochs": 100, "epochs": records}
    payload.update(extra)
    _atomic_json_dump(payload, path)


class TestUpsertEpochRecord(unittest.TestCase):
    def test_appends_new_epochs_in_order(self):
        history = []
        for epoch in (1, 2, 3):
            upsert_epoch_record(history, make_record(epoch))
        self.assertEqual([r["epoch"] for r in history], [1, 2, 3])

    def test_epoch_is_the_unique_key(self):
        history = [make_record(1), make_record(2)]
        upsert_epoch_record(history, make_record(2, val_dice=0.99))
        self.assertEqual(len(history), 2, "a duplicate epoch must not be appended twice")
        self.assertEqual(history[1]["val_dice"], 0.99, "the epoch record must be updated in place")

    def test_out_of_order_insert_is_sorted(self):
        history = [make_record(1), make_record(5)]
        upsert_epoch_record(history, make_record(3))
        self.assertEqual([r["epoch"] for r in history], [1, 3, 5])

    def test_existing_epochs_are_never_dropped(self):
        history = [make_record(e) for e in range(1, 57)]
        for epoch in range(57, 101):
            upsert_epoch_record(history, make_record(epoch))
        self.assertEqual([r["epoch"] for r in history], list(range(1, 101)))


class TestLoadExistingHistory(unittest.TestCase):
    def test_returns_empty_for_a_fresh_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                load_existing_history(
                    os.path.join(tmp, "training_history.json"),
                    os.path.join(tmp, "metrics.csv"),
                ),
                [],
            )

    def test_returns_empty_when_given_no_paths(self):
        self.assertEqual(load_existing_history(None, None), [])

    def test_reads_the_json_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "training_history.json")
            write_history_json(path, [make_record(e) for e in range(1, 6)])
            history = load_existing_history(path, os.path.join(tmp, "metrics.csv"))
            self.assertEqual([r["epoch"] for r in history], [1, 2, 3, 4, 5])
            self.assertEqual(history[0]["val_per_class_dice"], {"1": 0.55, "2": 0.45})

    def test_falls_back_to_metrics_csv_when_json_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "metrics.csv")
            _write_metrics_csv(csv_path, [make_record(e) for e in range(1, 8)])
            history = load_existing_history(os.path.join(tmp, "missing.json"), csv_path)
            self.assertEqual([r["epoch"] for r in history], list(range(1, 8)))
            self.assertAlmostEqual(history[0]["val_dice"], 0.5)
            self.assertAlmostEqual(history[0]["val_per_class_dice"]["1"], 0.55)

    def test_merges_both_sources_keyed_by_epoch(self):
        """A CSV that ran ahead of the JSON (or vice versa) must contribute its
        extra epochs rather than being ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "training_history.json")
            csv_path = os.path.join(tmp, "metrics.csv")
            write_history_json(json_path, [make_record(e) for e in range(1, 4)])
            _write_metrics_csv(csv_path, [make_record(e) for e in range(1, 6)])

            history = load_existing_history(json_path, csv_path)
            self.assertEqual([r["epoch"] for r in history], [1, 2, 3, 4, 5])
            # JSON wins on conflict: it carries the richer val-loss breakdown.
            self.assertIsNotNone(history[0]["val_loss_ce"])
            self.assertIsNone(history[4]["val_loss_ce"])

    def test_a_corrupt_json_does_not_abort_and_falls_back_to_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "training_history.json")
            csv_path = os.path.join(tmp, "metrics.csv")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write('{"epochs": [{"epoch": 1,')  # truncated mid-write
            _write_metrics_csv(csv_path, [make_record(e) for e in range(1, 4)])

            history = load_existing_history(json_path, csv_path)
            self.assertEqual([r["epoch"] for r in history], [1, 2, 3])

    def test_a_corrupt_csv_does_not_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "metrics.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss_total\nnot-an-int,0.5\n3,0.2\n")
            history = _history_from_metrics_csv(csv_path)
            self.assertEqual([r["epoch"] for r in history], [3])

    def test_accepts_a_bare_list_history_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "training_history.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([make_record(1), make_record(2)], f)
            self.assertEqual([r["epoch"] for r in load_existing_history(path, None)], [1, 2])


class TestResumeRoundTrip(unittest.TestCase):
    """End-to-end simulation of the exact failure the user reported: a run that
    stops at epoch 56 and continues to 100 must end with epochs 1..100."""

    def test_interrupted_run_keeps_all_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "training_history.json")
            csv_path = os.path.join(tmp, "metrics.csv")

            # --- Session 1: fresh run, dies after epoch 56.
            history = load_existing_history(json_path, csv_path)
            self.assertEqual(history, [])
            for epoch in range(1, 57):
                upsert_epoch_record(history, make_record(epoch, val_dice=0.5 + epoch / 1000))
                _write_metrics_csv(csv_path, history)
                write_history_json(json_path, history)

            # --- Session 2: resume at epoch 57 and finish.
            resumed = load_existing_history(json_path, csv_path)
            self.assertEqual(len(resumed), 56, "history must be restored before training continues")
            for epoch in range(57, 101):
                upsert_epoch_record(resumed, make_record(epoch, val_dice=0.5 + epoch / 1000))
                _write_metrics_csv(csv_path, resumed)
                write_history_json(json_path, resumed)

            # --- Final artifacts must contain the WHOLE run.
            final = load_existing_history(json_path, csv_path)
            self.assertEqual([r["epoch"] for r in final], list(range(1, 101)))

            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(r["epoch"]) for r in rows], list(range(1, 101)))
            self.assertAlmostEqual(float(rows[0]["val_dice_proxy"]), 0.501)

            with open(json_path, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(len(payload["epochs"]), 100)
            self.assertEqual(payload["epochs"][0]["epoch"], 1)

    def test_two_interruptions_still_keep_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "training_history.json")
            csv_path = os.path.join(tmp, "metrics.csv")
            for start, stop in ((1, 31), (31, 64), (64, 101)):
                history = load_existing_history(json_path, csv_path)
                for epoch in range(start, stop):
                    upsert_epoch_record(history, make_record(epoch))
                    _write_metrics_csv(csv_path, history)
                    write_history_json(json_path, history)
            final = load_existing_history(json_path, csv_path)
            self.assertEqual([r["epoch"] for r in final], list(range(1, 101)))

    def test_replaying_an_already_recorded_epoch_does_not_duplicate_it(self):
        """A resume that re-runs an epoch (e.g. the checkpoint was written
        before the history flush) must correct that row, not append a second
        one."""
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "training_history.json")
            csv_path = os.path.join(tmp, "metrics.csv")
            history = [make_record(e) for e in range(1, 21)]
            _write_metrics_csv(csv_path, history)
            write_history_json(json_path, history)

            resumed = load_existing_history(json_path, csv_path)
            for epoch in range(18, 26):  # overlaps epochs 18, 19, 20
                upsert_epoch_record(resumed, make_record(epoch, val_dice=0.77))
            _write_metrics_csv(csv_path, resumed)

            self.assertEqual([r["epoch"] for r in resumed], list(range(1, 26)))
            self.assertAlmostEqual(resumed[17]["val_dice"], 0.77)
            self.assertAlmostEqual(resumed[0]["val_dice"], 0.5)


class TestAtomicWrites(unittest.TestCase):
    def test_json_dump_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "training_history.json")
            _atomic_json_dump({"epochs": [make_record(1)]}, path)
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_metrics_csv_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.csv")
            _write_metrics_csv(path, [make_record(1)])
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_json_dump_preserves_the_old_file_on_failure(self):
        """An unserializable payload must not destroy the previous history."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "training_history.json")
            write_history_json(path, [make_record(1)])
            with self.assertRaises(TypeError):
                _atomic_json_dump({"epochs": [{"epoch": 1, "bad": object()}]}, path)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)["epochs"]), 1)


if __name__ == "__main__":
    unittest.main()
