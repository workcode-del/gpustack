import csv
import io
import os
import re
import tempfile
import subprocess
import logging
from gpustack.detectors.base import GPUDetector
from gpustack.schemas.workers import (
    GPUCoreInfo,
    GPUDeviceInfo,
    GPUDevicesInfo,
    MemoryInfo,
    VendorEnum,
)
from gpustack.utils import platform
from gpustack.utils.command import is_command_available
from gpustack.utils.convert import safe_float, safe_int

logger = logging.getLogger(__name__)

class MuxiSMI(GPUDetector):
    def is_available(self) -> bool:
        return is_command_available("mx-smi")

    def gather_gpu_info(self) -> GPUDevicesInfo:
        command = self._command_gather_gpu()
        results = self._run_command(command)
        if results is None:
            return []
        return self.decode_gpu_devices(results)

    def decode_gpu_devices(self, result) -> GPUDevicesInfo:
        devices = []
        reader = csv.DictReader(io.StringIO(result))
        for row in reader:
            device_id = row.get('deviceId', '')
            index_match = None
            if device_id:
                index_match = re.search(r'GPU#(\d+)', device_id)
            if not index_match:
                continue
            index = int(index_match.group(1))
            name = row.get('deviceName', '').strip()
            memory_total = safe_int(row.get('utilization.vis_vram.total [KB]', 0)) * 1024
            memory_used = safe_int(row.get('utilization.vis_vram.used [KB]', 0)) * 1024
            utilization_gpu = safe_float(row.get('utilization.vis_vram.usage [%]', 0))
            temperature_gpu = safe_float(row.get('temperature.hotspot [C]', 0))
            device = GPUDeviceInfo(
                index=index,
                name=name,
                vendor=VendorEnum.MUXI.value,
                memory=MemoryInfo(
                    is_unified_memory=False,
                    used=memory_used,
                    total=memory_total,
                    utilization_rate=(
                        (memory_used / memory_total) * 100 if memory_total > 0 else 0
                    ),
                ),
                core=GPUCoreInfo(
                    utilization_rate=utilization_gpu,
                    total=0,
                ),
                temperature=temperature_gpu,
                type=platform.DeviceTypeEnum.MUXI.value,
            )
            devices.append(device)
        return devices

    def _run_command(self, command):
        result = None
        temp_csv_file = None
        try:
            fd, temp_csv_file = tempfile.mkstemp(suffix='.csv')
            os.close(fd)
            csv_command = command + ["-o", temp_csv_file]
            result = subprocess.run(
                csv_command, capture_output=True, text=True, encoding="utf-8", timeout=10
            )
            if result is None or result.returncode != 0:
                logger.error(f"mx-smi命令返回异常代码: {result.returncode if result else 'None'}")
                return None
            output = result.stdout
            if output and "no devices" in output.lower():
                logger.warning("未检测到沐曦显卡设备")
                return None
            if os.path.exists(temp_csv_file):
                with open(temp_csv_file, 'r', encoding='utf-8') as f:
                    csv_content = f.read()
                return csv_content
            logger.error("CSV文件未生成")
            return None
        except subprocess.TimeoutExpired as e:
            logger.error(f"mx-smi命令执行超时: {e}")
            return None
        except Exception as e:
            error_message = f"执行mx-smi命令失败: {e}"
            if result:
                error_message += f", stdout: {result.stdout}, stderr: {result.stderr}"
            logger.error(error_message)
            return None
        finally:
            if temp_csv_file and os.path.exists(temp_csv_file):
                try:
                    os.unlink(temp_csv_file)
                except Exception as e:
                    logger.warning(f"删除临时CSV文件失败: {e}")

    def _command_gather_gpu(self):
        executable_command = [
            "mx-smi",
            "--show-memory",
            "--show-temperature"
        ]
        return executable_command

    def get_driver_version(self) -> str:
        try:
            command = ["mx-smi", "--show-version"]
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                encoding="utf-8",
                timeout=5
            )
            if result.returncode == 0:
                output = result.stdout
                version_match = re.search(r'Driver Version\s*:\s*([\d\.]+)', output)
                if version_match:
                    return version_match.group(1)
                lines = output.strip().split('\n')
                if lines:
                    return lines[0]
            return "Unknown"
        except subprocess.TimeoutExpired as e:
            logger.warning(f"获取沐曦显卡驱动版本超时: {e}")
            return "Unknown"
        except Exception as e:
            logger.warning(f"获取沐曦显卡驱动版本失败: {e}")
            return "Unknown"
