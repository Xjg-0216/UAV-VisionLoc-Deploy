
from loguru import logger
import sys

def configure_logging(config):
    """配置日志"""
    log_level = config['logging_level'].upper()
    logger.remove()  # 移除默认配置
    logger.add(
        config["log_path"],
        level=log_level,
        format="{time} - {name} - {level} - {message}",
        rotation="10 MB",  # 设置日志文件大小
        retention="10 days"  # 设置日志文件保留时间
    )
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time} - {name} - {level} - {message}"
    )
