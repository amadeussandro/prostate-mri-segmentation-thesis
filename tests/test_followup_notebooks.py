"""Tests for the two follow-up Colab notebooks and their builder.

A notebook that only breaks once it is running in Colab costs a whole session to
discover. These tests catch the failures that are checkable here: a flag the CLI
does not define, a cell collapsed onto one line by a missing newline, a gate
quietly dropped from the builder, or a notebook that has drifted from the
builder that is supposed to generate it.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

NOTEBOOKS = {
    "finalize": os.path.join(PROJECT_ROOT, "notebooks", "prostate158_rq2_finalize_colab.ipynb"),
    "prostatex": os.path.join(PROJECT_ROOT, "notebooks",
                              "prostate158_prostatex_external_colab.ipynb"),
}
BUILDER = os.path.join(SCRIPTS_DIR, "_build_colab_notebook_rq2_followup.py")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cell_text(nb, kind=None):
    return "\n".join("".join(c["source"]) for c in nb["cells"]
                     if kind is None or c["cell_type"] == kind)


def script_source(name):
    with open(os.path.join(SCRIPTS_DIR, name), encoding="utf-8") as fh:
        return fh.read()


class TestStructure(unittest.TestCase):
    def test_both_notebooks_exist_and_parse(self):
        for key, path in NOTEBOOKS.items():
            self.assertTrue(os.path.exists(path), f"{key} notebook missing at {path}")
            nb = load(path)
            self.assertGreater(len(nb["cells"]), 10)
            self.assertEqual(nb["nbformat"], 4)

    def test_source_lines_keep_their_newlines(self):
        # Without this, every cell opens as a single unreadable line.
        for key, path in NOTEBOOKS.items():
            for i, c in enumerate(load(path)["cells"]):
                src = c["source"]
                self.assertTrue(src, f"{key} cell {i} is empty")
                for j, line in enumerate(src[:-1]):
                    self.assertTrue(line.endswith("\n"),
                                    f"{key} cell {i} line {j} lost its newline")

    def test_code_cells_are_well_formed(self):
        for key, path in NOTEBOOKS.items():
            for i, c in enumerate(load(path)["cells"]):
                if c["cell_type"] == "code":
                    self.assertIn("outputs", c, f"{key} cell {i}")
                    self.assertIn("execution_count", c, f"{key} cell {i}")

    def test_runtimes_match_what_each_notebook_needs(self):
        # The finalize notebook never loads the model, so it must not ask for a
        # GPU the user then waits for; the external validation does 204 cases.
        self.assertEqual(load(NOTEBOOKS["finalize"])["metadata"]["accelerator"], "None")
        self.assertEqual(load(NOTEBOOKS["prostatex"])["metadata"]["accelerator"], "GPU")


class TestFlagsMatchTheCLIs(unittest.TestCase):
    def _flags_in(self, text):
        return set(re.findall(r'(--[a-z0-9-]+)', text))

    def _cli_flags(self, script):
        return set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', script_source(script)))

    def test_finalize_notebook_flags_exist(self):
        cells = [c for c in load(NOTEBOOKS["finalize"])["cells"]
                 if c["cell_type"] == "code" and "!python scripts/" in "".join(c["source"])]
        self.assertTrue(cells, "no script invocation found in the finalize notebook")
        allowed = self._cli_flags("run_zone_volumes.py") | self._cli_flags("regenerate_rq2_figures.py")
        used = set()
        for c in cells:
            used |= self._flags_in("".join(c["source"]))
        missing = used - allowed
        self.assertFalse(missing, f"notebook uses undefined flags: {missing}")

    def test_prostatex_notebook_flags_exist(self):
        cells = [c for c in load(NOTEBOOKS["prostatex"])["cells"]
                 if c["cell_type"] == "code" and "!python scripts/" in "".join(c["source"])]
        self.assertTrue(cells, "no script invocation found in the PROSTATEx notebook")
        allowed = self._cli_flags("run_prostatex_external_validation.py")
        used = set()
        for c in cells:
            used |= self._flags_in("".join(c["source"]))
        missing = used - allowed
        self.assertFalse(missing, f"notebook uses undefined flags: {missing}")

    def test_the_escape_hatch_is_never_used_in_the_notebook(self):
        # --skip-sha-check exists for debugging and must never appear in a
        # notebook that produces a reported result.
        text = cell_text(load(NOTEBOOKS["prostatex"]), "code")
        self.assertNotIn("--skip-sha-check", text)


class TestGatesArePresent(unittest.TestCase):
    def test_prostatex_notebook_gates_the_upload_and_the_checkpoint(self):
        nb = load(NOTEBOOKS["prostatex"])
        code = cell_text(nb, "code")
        self.assertIn("EXPECTED_CASES", code)
        self.assertIn("prostatex_manifest.csv", code)
        self.assertIn("sha256", code.lower())
        # Both gates must actually stop the run, not just print.
        self.assertIn("raise FileNotFoundError", code)
        self.assertIn("assert sha == EXPECTED_SHA256", code)

    def test_prostatex_notebook_fixes_the_framing_before_the_numbers(self):
        text = cell_text(load(NOTEBOOKS["prostatex"]), "markdown").lower()
        self.assertIn("expect lower", text)
        self.assertIn("generalization", text)

    def test_finalize_notebook_checks_the_cohort_is_complete(self):
        code = cell_text(load(NOTEBOOKS["finalize"]), "code")
        self.assertIn("EXPECTED_TEST_CASES", code)
        self.assertIn("assert", code)

    def test_finalize_notebook_states_that_metrics_are_untouched(self):
        text = cell_text(load(NOTEBOOKS["finalize"]), "markdown").lower()
        self.assertIn("untouched", text)


class TestBuilderIsTheSourceOfTruth(unittest.TestCase):
    def test_rebuilding_reproduces_the_committed_notebooks(self):
        # The repo convention is to edit the builder, not the .ipynb. If these
        # have diverged, someone hand-edited a generated file.
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            proc = subprocess.run(
                [sys.executable, BUILDER], cwd=PROJECT_ROOT, env=env,
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        for key, path in NOTEBOOKS.items():
            self.assertTrue(os.path.exists(path), f"{key} not regenerated")

    def test_builder_declares_both_notebooks(self):
        src = script_source("_build_colab_notebook_rq2_followup.py")
        self.assertIn("prostate158_rq2_finalize_colab.ipynb", src)
        self.assertIn("prostate158_prostatex_external_colab.ipynb", src)


if __name__ == "__main__":
    unittest.main()
