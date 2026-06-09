"""全局配置模块。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，统一从环境变量加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek
    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        alias="DEEPSEEK_BASE_URL",
    )
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # Claude（演化推进判定）
    claude_api_key: str = Field(default="", alias="CLAUDE_API_KEY")
    claude_base_url: str = Field(default="https://api.anthropic.com", alias="CLAUDE_BASE_URL")
    claude_model: str = Field(default="claude-sonnet-4-6", alias="CLAUDE_MODEL")

    # 飞书旁白Bot
    narrator_app_id: str = Field(alias="NARRATOR_APP_ID")
    narrator_app_secret: str = Field(alias="NARRATOR_APP_SECRET")

    # 飞书NPC Bot
    npc_app_id: str = Field(alias="NPC_APP_ID")
    npc_app_secret: str = Field(alias="NPC_APP_SECRET")

    # 飞书事件验证
    feishu_encrypt_key: str = Field(default="", alias="FEISHU_ENCRYPT_KEY")
    feishu_verification_token: str = Field(alias="FEISHU_VERIFICATION_TOKEN")
    npc_encrypt_key: str = Field(default="", alias="NPC_ENCRYPT_KEY")
    npc_verification_token: str = Field(default="", alias="NPC_VERIFICATION_TOKEN")

    # 游戏群
    game_chat_id: str = Field(default="", alias="GAME_CHAT_ID")

    # 服务
    server_port: int = Field(default=8001, alias="SERVER_PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回单例配置对象。"""

    return Settings()
