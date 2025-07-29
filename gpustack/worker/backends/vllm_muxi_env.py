import logging
import os
import platform as py_platform
from gpustack.utils import platform
from gpustack.utils.command import is_command_available

logger = logging.getLogger(__name__)

def setup_muxi_env(env: dict) -> dict:
    """
    为沐曦显卡设置必要的环境变量，确保vLLM能够正确识别和使用沐曦显卡
    """
    if platform.device() != platform.DeviceTypeEnum.MUXI.value:
        return env
    if not is_command_available("mx-smi"):
        logger.warning("无法找到mx-smi命令，可能没有安装沐曦显卡驱动")
        return env
    try:
        from gpustack.detectors.muxi_smi.muxi_smi import MuxiSMI
        muxi_detector = MuxiSMI()
        driver_version = muxi_detector.get_driver_version()
        logger.info(f"检测到沐曦显卡驱动版本: {driver_version}")
    except Exception as e:
        logger.warning(f"无法获取沐曦显卡驱动信息: {e}")
    muxi_env_vars = [
        "CUDA_VISIBLE_DEVICES"
    ]
    for var in muxi_env_vars:
        if var in os.environ:
            env[var] = os.environ[var]
            logger.debug(f"设置沐曦显卡环境变量: {var}={os.environ[var]}")
            if var in ["MUSA_VISIBLE_DEVICES", "MT_VISIBLE_DEVICES"] and "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = os.environ[var]
                logger.debug(f"同步设置CUDA_VISIBLE_DEVICES={os.environ[var]}")
    system_type = py_platform.system()
    if system_type == "Windows":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        logger.debug("Windows环境下设置PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
    logger.info("已为沐曦显卡设置环境变量")
    return env
