"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from fastapi import FastAPI

from src.config import get_settings
from src.engine.game_master import handle_message
from src.engine.state import get_state_manager
from src.feishu.receiver import register_message_handler, router as feishu_router
from src.feishu.sender import get_sender
from src.llm.client import get_llm_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用生命周期：启动初始化与关闭清理。"""

    # 启动阶段：加载配置并初始化核心单例。
    settings = get_settings()
    get_llm_client()
    get_state_manager()
    get_sender()

    # 将消息处理入口注册到飞书接收器。
    register_message_handler(handle_message)
    logger.info("应用启动完成，监听端口: %s", settings.server_port)
    yield
    logger.info("应用已关闭")


app = FastAPI(title="Text RPG Engine", lifespan=lifespan)
app.include_router(feishu_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查接口。"""

    return {"status": "ok"}
