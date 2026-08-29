from __future__ import annotations

import importlib.util
import shutil
import time
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from web import job_manager
from web.app import _format_started_at, _should_show_session_zip
from web.artifact_browser import (
    DIRECT_DOWNLOAD_MAX_BYTES,
    _thumbnail_data_uri,
    cleanup_empty_directories,
    delete_files,
    list_directories,
    list_files,
    list_task_roots,
    next_preview_after_delete,
    resolve_directory,
    resolve_task_root,
    scan_artifact_snapshot,
    should_bundle,
    toggle_selection,
)
from web.helpers import deferred_file_bytes, deferred_files_zip, deferred_frames_zip
from web.pipeline import PipelineEvent, VideoResult, run_pipeline
from web.task_store import job_file, read_task, update_task, write_inference_stats, write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFER_SCRIPT = PROJECT_ROOT / "scripts" / "2_model_infer.py"
FRAME_SCRIPT = PROJECT_ROOT / "scripts" / "1_frame_extract.py"
TEST_ROOT = PROJECT_ROOT / "outputs" / "_v2_test_workspace"


def load_infer_module():
    spec = importlib.util.spec_from_file_location("v2_test_model_infer", INFER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_frame_module():
    spec = importlib.util.spec_from_file_location("v2_test_frame_extract", FRAME_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FrameExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_frame_module()

    def setUp(self):
        self.output_dir = TEST_ROOT / "抽帧输出"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if TEST_ROOT.exists():
            shutil.rmtree(TEST_ROOT)

    def test_unicode_output_is_written_and_included_in_zip(self):
        frame = np.zeros((12, 12, 3), dtype=np.uint8)

        class FakeCapture:
            def __init__(self):
                self.reads = 0

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == self_module.cv2.CAP_PROP_FPS:
                    return 25.0
                if prop == self_module.cv2.CAP_PROP_FRAME_COUNT:
                    return 1
                return 0

            def read(self):
                self.reads += 1
                return (True, frame.copy()) if self.reads == 1 else (False, None)

            def release(self):
                return None

        self_module = self.module
        with patch.object(self.module.cv2, "VideoCapture", return_value=FakeCapture()):
            result = self.module.extract_frames("video.mp4", str(self.output_dir))

        files = list(self.output_dir.glob("*.jpg"))
        self.assertEqual(result["saved_frames"], 1)
        self.assertEqual(len(files), 1)
        with zipfile.ZipFile(BytesIO(deferred_frames_zip(self.output_dir)())) as archive:
            self.assertEqual(archive.namelist(), ["frame_000000.jpg"])


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
            patch.object(
                self.module, "imread_unicode", return_value=np.zeros((8, 8, 3), dtype=np.uint8)
            ),
            patch.object(self.module.cv2, "imwrite", return_value=True),
        ):
            self.module.infer(
                "model.pt", str(input_dir), str(output_dir), device="cpu", classes=[1]
            )
        self.assertEqual(calls[0][1]["classes"], [1])

    def test_infer_rejects_empty_class_list(self):
        with self.assertRaises(ValueError):
            self.module.infer("model.pt", ".", classes=[])

    def test_infer_returns_summary_and_per_image_statistics(self):
        input_dir = TEST_ROOT / "stats_input"
        output_dir = TEST_ROOT / "stats_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "frame.jpg").write_bytes(b"test")

        class Values:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class Boxes:
            cls = Values([1, 1])
            conf = Values([0.8, 0.6])

            def __len__(self):
                return 2

        class FakeModel:
            names = {0: "person", 1: "car"}

            def to(self, _device):
                return self

            def __call__(self, _image, **_kwargs):
                return [type("Result", (), {"boxes": Boxes(), "names": self.names})()]

        with (
            patch.object(self.module, "YOLO", return_value=FakeModel()),
            patch.object(
                self.module, "draw_boxes", return_value=np.zeros((8, 8, 3), dtype=np.uint8)
            ),
            patch.object(
                self.module, "imread_unicode", return_value=np.zeros((8, 8, 3), dtype=np.uint8)
            ),
            patch.object(self.module.cv2, "imwrite", return_value=True),
        ):
            stats = self.module.infer(
                "model.pt", str(input_dir), str(output_dir), device="cpu", classes=[1]
            )
        self.assertEqual(stats["matched_images"], 1)
        self.assertEqual(stats["class_counts"], {"1:car": 2})
        self.assertTrue((output_dir / "images" / "frame.jpg").is_file())
        self.assertEqual(stats["images"][0]["detections"], 2)


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

    def test_cleanup_removes_empty_children_but_keeps_task_with_artifacts(self):
        image = self.current / "a.jpg"
        image.write_bytes(b"image")
        empty = self.task / "empty" / "nested"
        empty.mkdir(parents=True)
        self.assertFalse(cleanup_empty_directories(self.task))
        self.assertFalse(empty.exists())
        self.assertTrue(self.task.exists())

    def test_delete_is_confined_to_task_root(self):
        inside = self.current / "inside.jpg"
        outside = self.outputs / "outside.jpg"
        inside.write_bytes(b"inside")
        outside.write_bytes(b"outside")
        with self.assertRaises(ValueError):
            delete_files([outside], self.task)
        self.assertTrue(outside.exists())
        self.assertEqual(delete_files([inside], self.task), 1)
        self.assertFalse(inside.exists())
        self.assertFalse(self.task.exists())

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

    def test_preview_moves_to_next_image_after_delete(self):
        names = ["a.jpg", "b.jpg", "c.jpg"]
        self.assertEqual(next_preview_after_delete(names, 1), "c.jpg")
        self.assertIsNone(next_preview_after_delete(names, 2))

    def test_single_full_result_does_not_offer_batch_zip(self):
        results = {
            "demo": VideoResult(
                stem="demo",
                output_video=Path("demo.mp4"),
                frames_dir=Path("frames"),
                annotated_dir=Path("annotated"),
            )
        }
        self.assertFalse(_should_show_session_zip(results))
        results["demo2"] = VideoResult(
            stem="demo2",
            output_video=Path("demo2.mp4"),
            frames_dir=Path("frames2"),
            annotated_dir=Path("annotated2"),
        )
        self.assertTrue(_should_show_session_zip(results))

    def test_result_start_time_format(self):
        self.assertEqual(_format_started_at("2026-08-29T09:15:00+08:00"), "2026-08-29 09:15:00")
        self.assertEqual(_format_started_at("2026-08-29T01:15:00+00:00"), "2026-08-29 09:15:00")

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

    def test_task_metadata_and_statistics_are_internal(self):
        image = self.current / "a.jpg"
        image.write_bytes(b"image")
        update_task(self.task, task_id="task_a", status="completed", mode="infer")
        write_inference_stats(
            self.task,
            {"total_images": 1, "images": [{"file_name": "a.jpg", "status": "ok"}]},
        )
        self.assertEqual(read_task(self.task)["status"], "completed")
        self.assertEqual(list_directories(self.task), [self.current.resolve()])

    def test_snapshot_is_cached_until_revision_changes(self):
        (self.current / "a.jpg").write_bytes(b"image")
        first = scan_artifact_snapshot(str(self.outputs.resolve()), 0, 0)
        (self.current / "b.jpg").write_bytes(b"image")
        cached = scan_artifact_snapshot(str(self.outputs.resolve()), 0, 0)
        refreshed = scan_artifact_snapshot(str(self.outputs.resolve()), 0, 1)
        self.assertEqual(first, cached)
        self.assertEqual(refreshed[0]["summary"]["file_count"], 2)

    def test_corrupt_image_thumbnail_degrades_gracefully(self):
        broken = self.current / "broken.jpg"
        broken.write_bytes(b"not an image")
        stat = broken.stat()
        self.assertEqual(_thumbnail_data_uri(str(broken), stat.st_mtime_ns, stat.st_size), "")

    def test_direct_download_revalidates_allowed_root(self):
        inside = self.current / "inside.txt"
        outside = self.outputs / "outside.txt"
        inside.write_bytes(b"inside")
        outside.write_bytes(b"outside")
        self.assertEqual(deferred_file_bytes(inside, self.task)(), b"inside")
        with self.assertRaises(ValueError):
            deferred_file_bytes(outside, self.task)()

    def test_pipeline_can_cancel_before_processing_and_persists_status(self):
        source = self.current / "a.jpg"
        source.write_bytes(b"image")
        results = run_pipeline(
            video_paths=[],
            model_path=None,
            frame_interval=1,
            conf=0.25,
            iou=0.45,
            device="cpu",
            box_color=(0, 0, 255),
            label_map=None,
            fps=30,
            mode="encode",
            annotated_dir=self.current,
            outputs_root=self.outputs,
            uploads_root=TEST_ROOT / "uploads",
            cancel_cb=lambda: True,
            task_context={"batch_id": "batch_test", "config": {}, "inputs": {}},
        )
        self.assertEqual(results[0].status, "cancelled")
        self.assertEqual(read_task(self.task)["status"], "cancelled")


class BackgroundJobTests(unittest.TestCase):
    def setUp(self):
        self.outputs = TEST_ROOT / "job_outputs"
        self.outputs.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if TEST_ROOT.exists():
            shutil.rmtree(TEST_ROOT)

    def test_background_job_persists_progress_and_results(self):
        batch_id = "batch_background_test"

        def fake_pipeline(*, progress_cb, cancel_cb, **_kwargs):
            self.assertFalse(cancel_cb())
            progress_cb(PipelineEvent("demo", "infer", 1, 2, "推理 1/2"))
            return [VideoResult(stem="demo", status="completed")]

        with patch.object(job_manager, "run_pipeline", side_effect=fake_pipeline):
            job_manager.submit_pipeline(
                batch_id=batch_id,
                outputs_root=self.outputs,
                kwargs={"mode": "infer"},
                stems=["demo"],
            )
            deadline = time.time() + 3
            status = None
            while time.time() < deadline:
                status = job_manager.active_status(self.outputs)
                if status and status.get("status") == "completed":
                    break
                time.sleep(0.02)
        self.assertEqual(status["status"], "completed")
        restored = job_manager.results_from_status(status)[0]
        self.assertEqual(restored.stem, "demo")
        self.assertIsNotNone(restored.started_at)
        self.assertEqual(restored.mode, "infer")

    def test_orphaned_running_job_is_marked_interrupted(self):
        path = job_file(self.outputs, "orphan")
        write_json(path, {"batch_id": "orphan", "status": "running"})
        status = job_manager.active_status(self.outputs)
        self.assertEqual(status["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
