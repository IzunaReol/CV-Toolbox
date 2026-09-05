"""v2.2.0 回归：使用真实图片/视频 IO 和替代模型，不下载权重。"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np

from web import job_manager, media, pipeline
from web.app import _should_show_session_zip
from web.helpers import deferred_infer_zip
from web.media import VerifiedVideoWriter, list_images

ROOT = Path(__file__).resolve().parent.parent


class FakeModel:
    names = {0: "person"}

    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(image)
        return [type("Result", (), {"boxes": None, "names": self.names})()]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "outputs" / f"_v220_test_{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.model_file = self.root / "model.pt"
        self.model_file.write_bytes(b"fake weight")
        self.model = FakeModel()
        self.runtime = patch.object(
            pipeline._infer_mod, "load_model", return_value=(self.model, "cpu")
        )
        self.loader = self.runtime.start()

    def tearDown(self):
        self.runtime.stop()
        resolved = self.root.resolve()
        assert resolved.parent == (ROOT / "outputs").resolve() and resolved.name.startswith(
            "_v220_test_"
        )
        shutil.rmtree(resolved)

    def image(self, directory, name="frame.png", value=100, shape=(32, 48, 3)):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / name
        pipeline._infer_mod.imwrite_unicode(target, np.full(shape, value, dtype=np.uint8))
        return target

    def video(self, name="视频", count=12, fps=29.97):
        target = self.root / "uploads" / name / "video.mp4"
        with VerifiedVideoWriter(target, fps) as writer:
            for index in range(count):
                writer.write(np.full((32, 48, 3), index * 10, dtype=np.uint8))
        return target

    def run_pipeline(self, **kwargs):
        options = dict(
            video_paths=[],
            model_path=self.model_file,
            frame_interval=1,
            conf=0.25,
            iou=0.45,
            device="cpu",
            box_color=(0, 0, 255),
            label_map=None,
            fps=None,
            outputs_root=self.root / "results",
            uploads_root=self.root / "uploads",
        )
        return pipeline.run_pipeline(**(options | kwargs))

    def test_same_name_same_size_uploads_are_isolated(self):
        first = pipeline.save_uploaded_video(b"AAAA", "same.mp4", self.root / "uploads")
        second = pipeline.save_uploaded_video(b"BBBB", "same.mp4", self.root / "uploads")
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), b"AAAA")
        self.assertEqual(second.read_bytes(), b"BBBB")

    def test_models_are_immutable_and_content_addressed(self):
        cache = self.root / "models"
        first = pipeline.cache_uploaded_model(b"AAAA", "same.pt", cache)
        second = pipeline.cache_uploaded_model(b"BBBB", "same.pt", cache)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), b"AAAA")
        self.assertEqual(pipeline.cache_uploaded_model(b"AAAA", "same.pt", cache), first)
        self.assertEqual(len(first.stem.rsplit("_", 1)[-1]), 64)

    def test_png_jpeg_tiff_and_uppercase_inputs_and_zip(self):
        images = self.root / "input" / "_uploaded"
        for name in ("a.PNG", "b.jpeg", "c.tiff", "d.webp"):
            self.image(images, name)
        (images / "fake.jpg").mkdir()
        result = self.run_pipeline(mode="infer", frames_dir=images)[0]
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(result.stats["processed_images"], 4)
        self.assertEqual(len(list_images(result.annotated_dir)), 4)
        with zipfile.ZipFile(io.BytesIO(deferred_infer_zip(result.annotated_dir)())) as archive:
            self.assertEqual(set(archive.namelist()), {"a.PNG", "b.jpeg", "c.tiff", "d.webp"})

    def test_images_decoded_once_and_model_receives_array(self):
        images = self.root / "images"
        self.image(images)
        reader = pipeline._infer_mod.imread_unicode
        with patch.object(pipeline._infer_mod, "imread_unicode", wraps=reader) as read:
            self.run_pipeline(mode="infer", frames_dir=images)
        self.assertEqual(read.call_count, 1)
        self.assertIsInstance(self.model.calls[0], np.ndarray)

    def test_corrupt_input_skipped_before_model_call(self):
        images = self.root / "images"
        self.image(images)
        (images / "broken.png").write_bytes(b"broken")
        result = self.run_pipeline(mode="infer", frames_dir=images)[0]
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(result.stats["failed_images"], 1)
        self.assertEqual(len(self.model.calls), 1)

    def test_all_corrupt_inputs_fail(self):
        images = self.root / "images"
        images.mkdir()
        (images / "broken.png").write_bytes(b"broken")
        result = self.run_pipeline(mode="infer", frames_dir=images)[0]
        self.assertEqual(result.status, "failed")
        self.assertFalse(self.model.calls)

    def test_repeat_video_run_does_not_mix_historical_frames(self):
        video = self.video()
        first = self.run_pipeline(mode="full", video_paths=[video])[0]
        second = self.run_pipeline(mode="full", video_paths=[video], frame_interval=3)[0]
        self.assertEqual(first.status, "completed", first.error)
        self.assertEqual(second.status, "completed", second.error)
        self.assertNotEqual(first.stem, second.stem)
        self.assertEqual(len(list_images(first.frames_dir)), 12)
        self.assertEqual(len(list_images(second.frames_dir)), 4)
        self.assertEqual(pipeline.read_video_meta(second.output_video)[1], 4)

    def test_batch_reuses_model(self):
        videos = [self.video("one"), self.video("two")]
        results = self.run_pipeline(mode="full", video_paths=videos)
        self.assertTrue(all(r.status == "completed" for r in results), results)
        self.assertEqual(self.loader.call_count, 1)

    def test_original_fractional_fps_is_used_without_interval_conversion(self):
        self.assertEqual(pipeline.resolve_output_fps(29.97), 29.97)
        self.assertEqual(pipeline.resolve_output_fps(29.97, fps=23.976), 23.976)
        video = self.video()
        result = self.run_pipeline(mode="full", video_paths=[video], frame_interval=3)[0]
        source_fps, source_count = pipeline.read_video_meta(video)
        fps, count = pipeline.read_video_meta(result.output_video)
        self.assertAlmostEqual(fps, source_fps, places=2)
        self.assertAlmostEqual(count / fps, source_count / source_fps / 3, places=2)

    def test_streaming_has_video_and_statistics_without_intermediate_images(self):
        results = self.run_pipeline(
            mode="full",
            video_paths=[self.video("a"), self.video("b")],
            streaming=True,
            frame_interval=3,
        )
        for result in results:
            self.assertEqual(result.status, "completed", result.error)
            self.assertIsNone(result.frames_dir)
            self.assertIsNone(result.annotated_dir)
            self.assertEqual(pipeline.read_video_meta(result.output_video)[1], 4)
            self.assertEqual(result.stats["processed_images"], 4)
            task_root = result.output_video.parent
            self.assertFalse((task_root / "frames").exists())
            self.assertFalse((task_root / "annotated").exists())
            self.assertTrue((task_root / "_meta" / "inference_stats.json").is_file())
        self.assertTrue(_should_show_session_zip({r.stem: r for r in results}))
        self.assertEqual(self.loader.call_count, 1)

    def test_streaming_cancel_removes_incomplete_video(self):
        video = self.video()
        result = self.run_pipeline(
            mode="full",
            video_paths=[video],
            streaming=True,
            cancel_cb=lambda: len(self.model.calls) >= 2,
        )[0]
        self.assertEqual(result.status, "cancelled")
        self.assertIsNone(result.output_video)
        self.assertEqual(list((self.root / "results").rglob("*.mp4")), [])

    def test_encoder_open_failure_does_not_report_success_or_replace_existing(self):
        output = self.root / "existing.mp4"
        output.write_bytes(b"existing")
        writer = MagicMock()
        writer.isOpened.return_value = False
        with patch.object(media.cv2, "VideoWriter", return_value=writer):
            with self.assertRaises(RuntimeError):
                with VerifiedVideoWriter(output, 30) as verified:
                    verified.write(np.zeros((32, 48, 3), dtype=np.uint8))
        self.assertEqual(output.read_bytes(), b"existing")
        writer.release.assert_called_once()

    def test_encoder_silent_write_failure_detected(self):
        writer = MagicMock()
        writer.isOpened.return_value = True
        with patch.object(media.cv2, "VideoWriter", return_value=writer):
            with self.assertRaises(RuntimeError):
                with VerifiedVideoWriter(self.root / "missing.mp4", 30) as verified:
                    verified.write(np.zeros((32, 48, 3), dtype=np.uint8))
        self.assertFalse((self.root / "missing.mp4").exists())

    def test_mismatched_images_fail_instead_of_silently_skipping(self):
        images = self.root / "images"
        self.image(images, "a.png")
        self.image(images, "b.png", shape=(64, 48, 3))
        output = self.root / "invalid.mp4"
        with self.assertRaises(ValueError):
            pipeline.create_video_from_images(str(images), str(output))
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("*.partial.mp4")))

    def test_extract_releases_capture_on_write_failure(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 1
        capture.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
        with (
            patch.object(pipeline._extract_mod.cv2, "VideoCapture", return_value=capture),
            patch.object(
                pipeline._extract_mod, "imwrite_unicode", side_effect=RuntimeError("disk full")
            ),
        ):
            with self.assertRaises(RuntimeError):
                pipeline.extract_frames("input.mp4", str(self.root / "frames"))
        capture.release.assert_called_once()

    def test_extract_cli_uses_arguments(self):
        video = self.video()
        frames = self.root / "cli_frames"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "1_frame_extract.py"),
                "--video",
                str(video),
                "--output",
                str(frames),
                "--interval",
                "3",
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list_images(frames)), 4)

    def test_invalid_interval_rejected_without_opening_video(self):
        with patch.object(pipeline._extract_mod.cv2, "VideoCapture") as capture:
            with self.assertRaises(ValueError):
                pipeline.extract_frames("input.mp4", str(self.root / "frames"), 0)
        capture.assert_not_called()

    def test_background_progress_is_throttled_but_terminal_state_persisted(self):
        path = self.root / "job.json"
        job = job_manager.BackgroundJob("review", path)

        def run(**kwargs):
            for index in range(100):
                kwargs["progress_cb"](pipeline.PipelineEvent("task", "infer", index, 100))
            kwargs["progress_cb"](pipeline.PipelineEvent("task", "done", 1, 1))
            return [pipeline.VideoResult("task", status="completed")]

        with (
            patch.object(job_manager, "run_pipeline", side_effect=run),
            patch.object(job_manager.time, "monotonic", return_value=10.0),
            patch.object(job_manager, "_write_status", wraps=job_manager._write_status) as write,
        ):
            job_manager._run_job(job, {})
        self.assertEqual(write.call_count, 4)
        self.assertEqual(job_manager.read_json(path)["status"], "completed")

    def test_font_is_cached_by_size(self):
        pipeline._infer_mod.find_cjk_font.cache_clear()
        first = pipeline._infer_mod.find_cjk_font(18)
        self.assertIs(pipeline._infer_mod.find_cjk_font(18), first)

    def test_cli_functions_reject_old_images_without_modifying_them(self):
        frames = self.root / "frames"
        old = self.image(frames, "frame_000000.jpg")
        before = old.read_bytes()
        with self.assertRaises(FileExistsError):
            pipeline.extract_frames("unused.mp4", str(frames))
        self.assertEqual(old.read_bytes(), before)
        annotated = self.root / "annotated"
        old_result = self.image(annotated / "images")
        with self.assertRaises(FileExistsError):
            pipeline.infer(
                str(self.model_file), str(frames), str(annotated), runtime=(self.model, "cpu")
            )
        self.assertTrue(old_result.is_file())

    def test_encode_mode_does_not_apply_video_sampling_interval(self):
        images = self.root / "compose" / "_uploaded"
        self.image(images)
        result = self.run_pipeline(mode="encode", annotated_dir=images, frame_interval=5)[0]
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(pipeline.read_video_meta(result.output_video)[0], 30)

    def test_ui_modes_and_fractional_fps_controls(self):
        from streamlit.testing.v1 import AppTest

        from web import app

        with patch.object(app, "OUTPUTS_DIR", self.root / "ui_outputs"):
            entry = self.root / "ui_test_app.py"
            entry.write_text("from web.app import main\nmain()\n", encoding="utf-8")
            ui = AppTest.from_file(str(entry)).run(timeout=15)
            self.assertFalse(ui.exception)
            self.assertNotIn("播放方式", [radio.label for radio in ui.radio])
            ui.checkbox(key="streaming_v0").check().run()
            ui.selectbox(key="fps_choice_v0").set_value("自定义").run()
            ui.number_input(key="fps_custom_v0").set_value(23.976).run()
            self.assertEqual(ui.number_input(key="fps_custom_v0").value, 23.976)
            for mode in ("仅抽帧", "仅推理", "仅合成", "文件浏览", "全流程"):
                ui.radio(key="mode_radio_v0").set_value(mode).run()
                self.assertFalse(ui.exception, mode)

    def test_upload_clear_only_resets_target_and_model_derived_state(self):
        from web import app

        state = {
            "_reset_token": 0,
            "model_path": self.model_file,
            "class_names": {0: "person"},
            "model_upload_id": "first",
            "cached_model_path": str(self.model_file),
            "lbl_0": "人",
            "infer_classes_v0": [0],
            "videos_uploader_v0_u0": ["video"],
            "model_uploader_v0_u0": "model",
            "results": {"old": "result"},
            "active_batch_id": "running_batch",
        }
        with patch.object(app.st, "session_state", state):
            app._clear_uploaded("videos_uploader")
            self.assertEqual(state["_upload_revision_videos_uploader"], 1)
            self.assertNotIn("videos_uploader_v0_u0", state)
            self.assertEqual(state["model_path"], self.model_file)
            app._clear_uploaded("videos_uploader")
            self.assertEqual(state["_upload_revision_videos_uploader"], 2)
            app._clear_uploaded("model_uploader")
            self.assertIsNone(state["model_path"])
            self.assertEqual(state["class_names"], {})
            self.assertNotIn("lbl_0", state)
            self.assertNotIn("infer_classes_v0", state)
            self.assertEqual(state["results"], {"old": "result"})
            self.assertEqual(state["active_batch_id"], "running_batch")
        self.assertTrue(self.model_file.exists())

    def test_all_uploaders_have_repeatable_clear_buttons(self):
        from streamlit.testing.v1 import AppTest

        from web import app

        with patch.object(app, "OUTPUTS_DIR", self.root / "ui_outputs"):
            entry = self.root / "upload_test_app.py"
            entry.write_text("from web.app import main\nmain()\n", encoding="utf-8")
            ui = AppTest.from_file(str(entry)).run(timeout=15)
            modes = {
                "全流程": ["model_uploader", "videos_uploader"],
                "仅推理": ["model_uploader", "infer_images"],
                "仅合成": ["model_uploader", "encode_images", "encode_source_video"],
            }
            for mode, keys in modes.items():
                ui.radio(key="mode_radio_v0").set_value(mode).run()
                for base in keys:
                    for _ in range(2):
                        previous = {
                            item.proto.label: item.proto.id for item in ui.get("file_uploader")
                        }
                        ui.button(key=f"clear_upload_{base}_v0").click().run()
                        self.assertFalse(ui.exception)
                        current = {
                            item.proto.label: item.proto.id for item in ui.get("file_uploader")
                        }
                        changed = [label for label in previous if previous[label] != current[label]]
                        self.assertEqual(len(changed), 1)

    def test_task_ids_are_unique_even_in_same_second(self):
        from web.app import _timestamp_id

        ids = [_timestamp_id("infer") for _ in range(100)]
        self.assertEqual(len(set(ids)), 100)

    def test_running_job_blocks_both_bulk_and_artifact_deletion(self):
        from concurrent.futures import Future

        from web import app, artifact_browser

        outputs = self.root / "protected"
        task = outputs / "task"
        image = self.image(task)
        path = job_manager.job_file(outputs, "active")
        job = job_manager.BackgroundJob("active", path)
        job.future = Future()
        job_manager.write_json(path, {"batch_id": "active", "status": "running"})
        with patch.dict(job_manager._JOBS, {"active": job}):
            with self.assertRaisesRegex(ValueError, "批次正在运行"):
                artifact_browser.delete_files([image], task)
            with self.assertRaisesRegex(ValueError, "批次正在运行"):
                app._delete_dir_contents(outputs)
            self.assertTrue(image.exists())
            self.assertTrue(path.exists())
            job.future.set_result(None)
            self.assertEqual(artifact_browser.delete_files([image], task), 1)

    def test_bulk_delete_reports_files_that_could_not_be_removed(self):
        from web import app

        outputs = self.root / "delete_failure"
        outputs.mkdir()
        artifact = outputs / "occupied.mp4"
        artifact.write_bytes(b"file")
        with patch.object(Path, "unlink", side_effect=PermissionError("occupied")):
            with self.assertRaisesRegex(OSError, "occupied.mp4"):
                app._delete_dir_contents(outputs)
        self.assertTrue(artifact.exists())


if __name__ == "__main__":
    unittest.main()
