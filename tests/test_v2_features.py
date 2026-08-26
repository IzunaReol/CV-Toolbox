from __future__ import annotations

import importlib.util
import shutil
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from web.artifact_browser import (
    DIRECT_DOWNLOAD_MAX_BYTES,
    delete_files,
    list_directories,
    list_files,
    list_task_roots,
    resolve_directory,
    resolve_task_root,
    should_bundle,
    toggle_selection,
)
from web.helpers import deferred_files_zip


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFER_SCRIPT = PROJECT_ROOT / "scripts" / "2_model_infer.py"
TEST_ROOT = PROJECT_ROOT / "outputs" / "_v2_test_workspace"


def load_infer_module():
    spec = importlib.util.spec_from_file_location("v2_test_model_infer", INFER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ClassFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_infer_module()

    def test_parse_classes_accepts_ids_names_and_deduplicates(self):
        names = {0: "person", 1: "car", 2: "dog"}
        self.assertEqual(self.module.parse_classes("person,2,person", names), [0, 2])
        self.assertEqual(self.module.parse_classes("0；car", names), [0, 1])
        self.assertIsNone(self.module.parse_classes(None, names))

    def test_parse_classes_rejects_unknown_or_empty_selection(self):
        names = {0: "person"}
        with self.assertRaises(ValueError):
            self.module.parse_classes("car", names)
        with self.assertRaises(ValueError):
            self.module.parse_classes(",,", names)

    def test_infer_passes_selected_classes_to_ultralytics(self):
        input_dir = TEST_ROOT / "infer_input"
        output_dir = TEST_ROOT / "infer_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "frame.jpg").write_bytes(b"test")
        calls = []

        class FakeModel:
            names = {0: "person", 1: "car"}

            def to(self, _device):
                return self

            def __call__(self, image, **kwargs):
                calls.append((image, kwargs))
                result = type("Result", (), {"boxes": None, "names": self.names})()
                return [result]

        with (
            patch.object(self.module, "YOLO", return_value=FakeModel()),
            patch.object(self.module, "imread_unicode", return_value=np.zeros((8, 8, 3), dtype=np.uint8)),
            patch.object(self.module.cv2, "imwrite", return_value=True),
        ):
            self.module.infer(
                "model.pt", str(input_dir), str(output_dir), device="cpu", classes=[1]
            )
        self.assertEqual(calls[0][1]["classes"], [1])

    def test_infer_rejects_empty_class_list(self):
        with self.assertRaises(ValueError):
            self.module.infer("model.pt", ".", classes=[])


class ArtifactBrowserTests(unittest.TestCase):
    def setUp(self):
        self.outputs = TEST_ROOT / "outputs"
        self.task = self.outputs / "task_a"
        self.current = self.task / "frames"
        self.current.mkdir(parents=True, exist_ok=True)
        (self.outputs / "_models").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if TEST_ROOT.exists():
            shutil.rmtree(TEST_ROOT)

    def test_internal_roots_are_hidden_and_task_files_are_listed(self):
        image = self.current / "a.jpg"
        image.write_bytes(b"image")
        empty_task = self.outputs / "empty_task"
        (empty_task / "nested").mkdir(parents=True)
        (self.task / "empty_parent" / "empty_child").mkdir(parents=True)
        self.assertEqual([path.name for path in list_task_roots(self.outputs)], ["task_a"])
        root = resolve_task_root(self.outputs, "task_a")
        self.assertEqual(list_directories(root), [self.current.resolve()])
        directory = resolve_directory(root, "frames")
        self.assertEqual(list_files(directory, root), [image.resolve()])
        with self.assertRaises(ValueError):
            resolve_task_root(self.outputs, "_models")

    def test_task_root_is_shown_as_directory_when_it_contains_a_file(self):
        root_file = self.task / "result.zip"
        nested_file = self.current / "a.jpg"
        root_file.write_bytes(b"zip")
        nested_file.write_bytes(b"image")
        self.assertEqual(list_directories(self.task), [self.task.resolve(), self.current.resolve()])

    def test_delete_is_confined_to_task_root(self):
        inside = self.current / "inside.jpg"
        outside = self.outputs / "outside.jpg"
        inside.write_bytes(b"inside")
        outside.write_bytes(b"outside")
        self.assertEqual(delete_files([inside], self.task), 1)
        self.assertFalse(inside.exists())
        with self.assertRaises(ValueError):
            delete_files([outside], self.task)
        self.assertTrue(outside.exists())

    def test_download_thresholds(self):
        files = []
        for index in range(6):
            path = self.current / f"{index}.txt"
            path.write_bytes(b"x")
            files.append(path)
        self.assertFalse(should_bundle(files[:5]))
        self.assertTrue(should_bundle(files))
        exact = self.current / "exact.bin"
        with exact.open("wb") as stream:
            stream.truncate(DIRECT_DOWNLOAD_MAX_BYTES)
        self.assertFalse(should_bundle([exact]))
        large = self.current / "large.bin"
        with large.open("wb") as stream:
            stream.truncate(DIRECT_DOWNLOAD_MAX_BYTES + 1)
        self.assertTrue(should_bundle([large]))

    def test_image_selection_toggle(self):
        self.assertEqual(toggle_selection([], "a.jpg"), ["a.jpg"])
        self.assertEqual(toggle_selection(["a.jpg", "b.jpg"], "a.jpg"), ["b.jpg"])

    def test_deferred_zip_preserves_relative_names_and_invalidates(self):
        first = self.current / "first.txt"
        first.write_bytes(b"one")
        initial = deferred_files_zip([first], self.task, "v2_test_bundle")()
        with zipfile.ZipFile(BytesIO(initial)) as archive:
            self.assertEqual(archive.namelist(), ["frames/first.txt"])
            self.assertEqual(archive.read("frames/first.txt"), b"one")
        first.write_bytes(b"updated")
        updated = deferred_files_zip([first], self.task, "v2_test_bundle")()
        self.assertNotEqual(initial, updated)


if __name__ == "__main__":
    unittest.main()
