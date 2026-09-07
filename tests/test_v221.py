"""v2.2.1：完整性、统计、轮询与磁盘下载。"""

import asyncio
import shutil
import subprocess
import sys
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage

from web import job_manager, pipeline
from web.downloads import DownloadRegistry, FileDownload, install_disk_downloads
from web.media import VerifiedVideoWriter, validate_decoded_frames
from web.task_store import InferenceJournal, read_json, write_json

ROOT = Path(__file__).resolve().parent.parent


class Capture:
    def __init__(self, total=10):
        self.total = total
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 30 if prop == pipeline.cv2.CAP_PROP_FPS else self.total

    def read(self):
        self.index += 1
        return (True, np.zeros((32, 48, 3), dtype=np.uint8)) if self.index <= 2 else (False, None)

    def release(self):
        self.released = True


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "outputs" / f"_v221_test_{uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self):
        resolved = self.root.resolve()
        assert resolved.parent == (ROOT / "outputs").resolve() and resolved.name.startswith(
            "_v221_test_"
        )
        shutil.rmtree(resolved)

    def test_extraction_and_streaming_reject_obvious_early_eof(self):
        for stream in (False, True):
            cap = Capture()
            with patch.object(pipeline.cv2, "VideoCapture", return_value=cap):
                with self.assertRaisesRegex(RuntimeError, "提前中断"):
                    if stream:
                        list(pipeline.iter_video_frames("video.mp4", 1))
                    else:
                        pipeline.extract_frames("video.mp4", str(self.root / "frames"))
            self.assertTrue(cap.released)

    def test_small_metadata_error_and_unknown_count_are_allowed(self):
        for expected, actual in [(100, 99), (1000, 995), (0, 2), (float("nan"), 2), (10, 11)]:
            validate_decoded_frames(expected, actual)
        with self.assertRaises(RuntimeError):
            validate_decoded_frames(0, 0)

    def test_extraction_accepts_unknown_frame_count(self):
        with patch.object(pipeline.cv2, "VideoCapture", return_value=Capture(float("nan"))):
            result = pipeline.extract_frames("video.mp4", str(self.root / "unknown"))
        self.assertEqual(result["saved_frames"], 2)

    def test_odd_dimensions_are_padded_without_resizing(self):
        output = self.root / "odd.mp4"
        frame = np.full((33, 49, 3), 120, dtype=np.uint8)
        with VerifiedVideoWriter(output, 29.97) as writer:
            writer.write(frame)
            self.assertEqual(writer.source_size, (49, 33))
            self.assertEqual(writer.size, (50, 34))
        cap = pipeline.cv2.VideoCapture(str(output))
        try:
            ok, decoded = cap.read()
            self.assertTrue(ok)
            self.assertEqual(decoded.shape, (34, 50, 3))
            self.assertLess(float(decoded[-1].mean()), 25)
        finally:
            cap.release()

    def test_different_source_sizes_that_pad_to_same_size_still_fail(self):
        with self.assertRaises(ValueError):
            with VerifiedVideoWriter(self.root / "mixed.mp4", 30) as writer:
                writer.write(np.zeros((33, 49, 3), dtype=np.uint8))
                writer.write(np.zeros((34, 50, 3), dtype=np.uint8))
        self.assertFalse((self.root / "mixed.mp4").exists())

    def test_journal_retains_cancelled_rows_without_a_list(self):
        journal = InferenceJournal(self.root / "task")
        with self.assertRaises(InterruptedError):
            with journal:
                for i in range(200):
                    journal.append(
                        {
                            "file_name": str(i),
                            "status": "ok",
                            "detections": 1,
                            "class_counts": {"0:person": 1},
                            "max_confidence": 0.9,
                            "avg_confidence": 0.9,
                            "error": "",
                        }
                    )
                self.assertFalse(any(isinstance(v, list) for v in vars(journal).values()))
                raise InterruptedError("cancelled")
        data = read_json(journal.paths[0])
        self.assertEqual(data["status"], "cancelled")
        self.assertTrue(data["partial"])
        self.assertEqual(data["total_images"], 200)
        self.assertEqual(len(data["images"]), 200)
        self.assertEqual(data["class_counts"], {"0:person": 200})
        self.assertEqual(len(journal.paths[1].read_text(encoding="utf-8-sig").splitlines()), 201)

    def test_pipeline_early_eof_keeps_partial_stats_but_no_video(self):
        model_path = self.root / "model.pt"
        model_path.touch()
        video = self.root / "source" / "video.mp4"
        video.parent.mkdir()
        video.touch()
        model = MagicMock(return_value=[type("Result", (), {"boxes": None})()])
        with (
            patch.object(pipeline.cv2, "VideoCapture", side_effect=lambda *a: Capture()),
            patch.object(pipeline._infer_mod, "load_model", return_value=(model, "cpu")),
        ):
            result = pipeline.run_pipeline(
                video_paths=[video],
                model_path=model_path,
                frame_interval=1,
                conf=0.25,
                iou=0.45,
                device="cpu",
                box_color=(0, 0, 255),
                label_map=None,
                fps=None,
                streaming=True,
                outputs_root=self.root / "results",
                uploads_root=self.root / "uploads",
            )[0]
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.output_video)
        stats = read_json(self.root / "results" / result.stem / "_meta" / "inference_stats.json")
        self.assertTrue(stats["partial"])
        self.assertEqual(stats["processed_images"], 2)
        self.assertFalse(list((self.root / "results").rglob("*.mp4")))

    def test_polling_legacy_history_migrates_once_then_only_reads_index(self):
        for i in range(30):
            write_json(
                job_manager.job_file(self.root, f"job{i}"),
                {
                    "batch_id": f"job{i}",
                    "status": "completed",
                    "results": [
                        {"stem": "task", "stats": {"total_images": 500, "images": [{}] * 500}}
                    ],
                },
            )
        status = job_manager.active_status(self.root)
        self.assertNotIn("images", status["results"][0]["stats"])
        with patch.object(job_manager, "read_json", wraps=job_manager.read_json) as read:
            for _ in range(3):
                self.assertEqual(job_manager.active_status(self.root), status)
        self.assertEqual(read.call_count, 3)
        self.assertTrue(all(call.args[0].name == "_current.json" for call in read.call_args_list))

    def test_corrupt_index_falls_back_and_rebuilds(self):
        write_json(
            job_manager.job_file(self.root, "job"), {"batch_id": "job", "status": "completed"}
        )
        index = job_manager.jobs_dir(self.root) / "_current.json"
        index.write_text("broken", encoding="utf-8")
        self.assertEqual(job_manager.active_status(self.root)["batch_id"], "job")
        self.assertIsNotNone(read_json(index))

    def test_new_result_serialization_does_not_copy_image_rows(self):
        result = pipeline.VideoResult("task", stats={"total_images": 2, "images": [object()]})
        data = job_manager._serialize_result(result)
        self.assertEqual(data["stats"], {"total_images": 2})

    def test_journal_memory_stays_small_for_long_runs(self):
        row = {
            "file_name": "frame.jpg",
            "status": "ok",
            "detections": 0,
            "class_counts": {},
            "max_confidence": None,
            "avg_confidence": None,
            "error": "",
        }
        tracemalloc.start()
        try:
            with InferenceJournal(self.root / "long") as journal:
                for _ in range(10000):
                    journal.append(row)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 2 * 1024 * 1024)
        self.assertFalse(journal.rows_path.exists())

    def test_download_adapter_avoids_read_bytes_and_preserves_fallback(self):
        file = self.root / "result.zip"
        file.write_bytes(b"hello")
        registry = DownloadRegistry(self.root)
        manager = MediaFileManager(MemoryMediaFileStorage("/media"))
        manager._deferred_callables["file"] = {
            "callable": FileDownload(lambda: file),
            "filename": "result.zip",
            "mimetype": "application/zip",
        }
        restore = install_disk_downloads(manager, registry)
        try:
            with patch.object(Path, "read_bytes", side_effect=AssertionError("full read")):
                url = manager.execute_deferred("file")
            self.assertTrue(url.startswith("/_cv_downloads/"))
            self.assertFalse(manager._storage._files_by_id)
        finally:
            restore()

    def test_file_route_supports_range_head_expiry_and_revalidation(self):
        file = self.root / "video.mp4"
        file.write_bytes(b"0123456789")
        registry = DownloadRegistry(self.root)
        url = registry.register(file)
        app = Starlette(
            routes=[Route("/_cv_downloads/{token}", registry.serve, methods=["GET", "HEAD"])]
        )
        with TestClient(app) as client:
            result = client.get(url, headers={"Range": "bytes=2-5"})
            self.assertEqual(result.status_code, 206)
            self.assertEqual(result.content, b"2345")
            self.assertEqual(client.head(url).headers["content-length"], "10")
            file.write_bytes(b"changed")
            self.assertEqual(client.get(url).status_code, 409)
            file.unlink()
            self.assertEqual(client.get(url).status_code, 404)
            self.assertEqual(client.get("/_cv_downloads/unknown").status_code, 404)
        with self.assertRaises(ValueError):
            registry.register(ROOT / "README.md")

    def test_download_tokens_expire_and_registry_is_bounded(self):
        file = self.root / "file.bin"
        file.write_bytes(b"x")
        registry = DownloadRegistry(self.root, ttl=10, capacity=2)
        with patch("web.downloads.time.monotonic", return_value=1):
            first = registry.register(file)
            registry.register(file)
            registry.register(file)
        self.assertEqual(len(registry.entries), 2)
        self.assertNotIn(first.rsplit("/", 1)[-1], registry.entries)
        app = Starlette(routes=[Route("/_cv_downloads/{token}", registry.serve)])
        with TestClient(app) as client, patch("web.downloads.time.monotonic", return_value=12):
            for token in registry.entries:
                self.assertEqual(client.get("/_cv_downloads/" + token).status_code, 404)

    def test_new_server_serves_existing_page_and_downloads(self):
        file = self.root / "server.bin"
        file.write_bytes(b"server-file")
        script = """
import sys
from pathlib import Path
from starlette.testclient import TestClient
from streamlit.runtime import get_instance
from web.server import create_app
from web.downloads import FileDownload
with TestClient(create_app()) as client:
    assert client.get('/').status_code == 200
    manager = get_instance().media_file_mgr
    manager._deferred_callables['test'] = {'callable': FileDownload(lambda: Path(sys.argv[1])),
                                          'filename': 'server.bin', 'mimetype': 'application/octet-stream'}
    response = client.get(manager.execute_deferred('test'), headers={'Range': 'bytes=0-5'})
    assert response.status_code == 206 and response.content == b'server'
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(file)], cwd=ROOT, capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_download_body_is_sent_in_bounded_chunks(self):
        file = self.root / "large.bin"
        file.write_bytes(b"x" * (1024 * 1024))
        registry = DownloadRegistry(self.root)
        url = registry.register(file)
        app = Starlette(routes=[Route("/_cv_downloads/{token}", registry.serve)])
        sizes = []

        async def run():
            async def receive():
                return {"type": "http.request", "body": b""}

            async def send(message):
                if message["type"] == "http.response.body":
                    sizes.append(len(message.get("body", b"")))

            await app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "method": "GET",
                    "path": url,
                    "raw_path": url.encode(),
                    "root_path": "",
                    "headers": [],
                    "query_string": b"",
                    "scheme": "http",
                    "server": ("localhost", 80),
                    "client": ("localhost", 1),
                    "http_version": "1.1",
                },
                receive,
                send,
            )

        asyncio.run(run())
        self.assertEqual(sum(sizes), 1024 * 1024)
        self.assertLessEqual(max(sizes), 64 * 1024)


if __name__ == "__main__":
    unittest.main()
