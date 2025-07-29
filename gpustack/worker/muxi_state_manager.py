"""
Muxi GPU Worker 状态增强模块
解决沐曦显卡环境下worker unreachable问题
"""
import logging
import threading
import time
from typing import Dict, List
from gpustack.detectors.muxi_smi.muxi_smi import MuxiSMI
from gpustack.schemas.workers import Worker, GPUDeviceInfo
from gpustack.utils import platform
from gpustack.utils.command import is_command_available

logger = logging.getLogger(__name__)

class MuxiWorkerStateManager:
    """
    沐曦显卡Worker状态管理器
    """
    def __init__(self):
        self.muxi_detector = MuxiSMI()

    def is_muxi_environment(self) -> bool:
        """
        检查是否为沐曦显卡环境
        """
        return (
            platform.device() == platform.DeviceTypeEnum.MUXI.value
            and is_command_available("mx-smi")
        )

    def get_enhanced_gpu_status(self) -> List[GPUDeviceInfo]:
        """
        获取增强的GPU状态信息（带超时保护）
        """
        if not self.is_muxi_environment():
            return []
        try:
            result_container = [None]
            exception_container = [None]
            def gpu_detection_thread():
                try:
                    result_container[0] = self.muxi_detector.gather_gpu_info()
                except Exception as e:
                    exception_container[0] = e
            thread = threading.Thread(target=gpu_detection_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=15)
            if thread.is_alive():
                logger.warning("沐曦GPU检测超时，返回空结果")
                return []
            if exception_container[0]:
                raise exception_container[0]
            gpu_devices = result_container[0] or []
            logger.debug(f"沐曦增强检测发现 {len(gpu_devices)} 个GPU设备")
            return gpu_devices
        except Exception as e:
            logger.error(f"沐曦增强GPU检测失败: {e}")
            return []

    def diagnose_worker_connectivity(self, worker: Worker) -> Dict[str, any]:
        """
        诊断Worker连通性问题
        """
        diagnosis = {
            'reachable': True,
            'gpu_status': 'unknown',
            'issues': [],
            'recommendations': []
        }
        if not self.is_muxi_environment():
            return diagnosis
        try:
            gpu_devices = self.get_enhanced_gpu_status()
            if not gpu_devices:
                diagnosis['issues'].append("无法检测到沐曦GPU设备")
                diagnosis['recommendations'].append("检查GPU设备是否正常连接")
                diagnosis['gpu_status'] = 'no_devices'
                diagnosis['reachable'] = False
            else:
                diagnosis['gpu_status'] = 'healthy'
                logger.info(f"检测到 {len(gpu_devices)} 个沐曦GPU设备")
        except Exception as e:
            diagnosis['issues'].append(f"诊断过程中发生错误: {e}")
            diagnosis['reachable'] = False
        return diagnosis

    def repair_worker_state(self, worker: Worker) -> bool:
        """
        尝试修复Worker状态
        """
        if not self.is_muxi_environment():
            return False
        try:
            logger.info(f"尝试修复Worker {worker.name} 的状态")
            gpu_devices = self.get_enhanced_gpu_status()
            if gpu_devices:
                logger.info(f"成功检测到 {len(gpu_devices)} 个沐曦GPU设备")
                return True
            else:
                logger.warning("无法检测到沐曦GPU设备")
                return False
        except Exception as e:
            logger.error(f"修复Worker状态失败: {e}")
            return False

muxi_state_manager = MuxiWorkerStateManager()

async def enhance_worker_health_check(worker: Worker) -> Dict[str, any]:
    diagnosis = muxi_state_manager.diagnose_worker_connectivity(worker)
    return {
        'healthy': diagnosis['reachable'],
        'message': f"GPU状态: {diagnosis['gpu_status']}",
        'details': {
            'issues': diagnosis['issues'],
            'recommendations': diagnosis['recommendations']
        }
    }

async def attempt_worker_recovery(worker: Worker) -> bool:
    return muxi_state_manager.repair_worker_state(worker)
