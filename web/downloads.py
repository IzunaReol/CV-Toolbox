"""现有下载按钮的磁盘传输适配器；不改变页面控件。"""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path
from types import MethodType

from starlette.responses import FileResponse, Response


class FileDownload:
    def __init__(self, resolver):
        self.get_download_path = resolver

    def __call__(self):
        # 兼容原 streamlit run 入口和直接调用。新入口不走这条整文件读取路径。
        return self.get_download_path().read_bytes()


class DownloadRegistry:
    def __init__(self, root: Path, ttl=3600, capacity=256):
        self.root = Path(root).resolve()
        self.ttl = ttl
        self.capacity = capacity
        self.entries = {}
        self.lock = threading.Lock()

    def validate(self, path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("不允许下载符号链接")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError("下载路径超出结果目录")
        return resolved

    def register(self, path, filename=None, mimetype=None):
        path = self.validate(path)
        stat = path.stat()
        now = time.monotonic()
        with self.lock:
            self.entries = {k: v for k, v in self.entries.items() if v[0] > now}
            while len(self.entries) >= self.capacity:
                self.entries.pop(next(iter(self.entries)))
            token = secrets.token_urlsafe(32)
            self.entries[token] = (
                now + self.ttl,
                path,
                stat.st_mtime_ns,
                stat.st_size,
                filename or path.name,
                mimetype,
            )
        return f"/_cv_downloads/{token}"

    async def serve(self, request):
        with self.lock:
            entry = self.entries.get(request.path_params["token"])
        if not entry or entry[0] <= time.monotonic():
            return Response("下载链接已过期，请重新点击下载", status_code=404)
        _, path, mtime, size, filename, mimetype = entry
        try:
            resolved = self.validate(path)
            stat = resolved.stat()
            if stat.st_mtime_ns != mtime or stat.st_size != size:
                return Response("文件已变化，请重新点击下载", status_code=409)
        except (OSError, ValueError):
            return Response("文件已移除", status_code=404)
        return FileResponse(
            resolved,
            filename=filename,
            media_type=mimetype,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )


def install_disk_downloads(manager, registry):
    """隔离 Streamlit 1.62 的内部适配；普通图片和非文件下载沿用原实现。"""
    if not hasattr(manager, "_deferred_callables") or not hasattr(manager, "_lock"):
        raise RuntimeError("当前 Streamlit 不支持磁盘下载适配，请使用项目锁定的版本")
    original = manager.execute_deferred

    def execute(self, file_id):
        with self._lock:
            entry = self._deferred_callables.get(file_id)
        callback = entry["callable"] if entry else None
        resolver = getattr(callback, "get_download_path", None)
        if callable(resolver):
            return registry.register(resolver(), entry.get("filename"), entry.get("mimetype"))
        return original(file_id)

    manager.execute_deferred = MethodType(execute, manager)

    def restore():
        manager.execute_deferred = original
        with registry.lock:
            registry.entries.clear()

    return restore
