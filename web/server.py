"""后台入口：python -m web.server。页面仍使用原来的 app.py。"""

from contextlib import asynccontextmanager
from pathlib import Path

import streamlit as st
import uvicorn
from starlette.routing import Route

from .downloads import DownloadRegistry, install_disk_downloads

ROOT = Path(__file__).resolve().parent.parent


def create_app():
    registry = DownloadRegistry(ROOT / "outputs")

    @asynccontextmanager
    async def lifespan(app):
        from streamlit.runtime import get_instance

        restore = install_disk_downloads(get_instance().media_file_mgr, registry)
        try:
            yield
        finally:
            restore()

    return st.App(
        ROOT / "web" / "app.py",
        lifespan=lifespan,
        routes=[Route("/_cv_downloads/{token}", registry.serve, methods=["GET", "HEAD"])],
    )


def main():
    import argparse

    from streamlit import config

    parser = argparse.ArgumentParser(description="启动 CV 工具箱（文件分块下载）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--max-upload-size", type=int, default=1024, help="单文件上传上限，MB")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.max_upload_size < 1:
        parser.error("端口须为 1–65535，上传上限必须大于 0")
    config.set_option("server.maxUploadSize", args.max_upload_size)
    config.set_option("server.port", args.port)
    config.set_option("server.address", args.host)
    print("正在启动 CV 工具箱，请保持此终端打开。按 Ctrl+C 停止服务。", flush=True)
    # Streamlit 自己管理服务器日志；避免 Uvicorn 再配置时移除其日志处理器。
    uvicorn.run(create_app(), host=args.host, port=args.port, log_config=None, log_level="info")


if __name__ == "__main__":
    main()
