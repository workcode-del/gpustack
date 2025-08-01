import asyncio
from datetime import datetime, timezone
import multiprocessing
import psutil
import requests
import setproctitle
import os
from typing import Dict, Optional, Set
import logging

from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.logging import (
    RedirectStdoutStderr,
)
from gpustack.utils import network, platform
from gpustack.utils.attrs import set_attr
from gpustack.utils.process import terminate_process_tree, add_signal_handlers
from gpustack.worker.daily_rotating_logger import DailyRotatingLogFile
from gpustack.worker.backends.llama_box import LlamaBoxServer
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker.backends.ascend_mindie import AscendMindIEServer
from gpustack.client import ClientSet
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    get_backend,
    DistributedServerCoordinateModeEnum,
    ModelInstanceSubordinateWorker,
)
from gpustack.server.bus import Event, EventType

logger = logging.getLogger(__name__)


class ServeManager:
    def __init__(
        self,
        worker_id: int,
        clientset: ClientSet,
        cfg: Config,
    ):
        self._worker_id = worker_id
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._serving_model_instances: Dict[int, multiprocessing.Process] = {}
        self._serving_model_instance_ports: Dict[int, Set[int]] = {}
        self._starting_model_instances: Dict[int, ModelInstance] = {}
        self._error_model_instances: Dict[int, ModelInstance] = {}
        self._model_cache_by_instance: Dict[int, Model] = {}
        self._clientset = clientset
        self._cache_dir = cfg.cache_dir

        os.makedirs(self._serve_log_dir, exist_ok=True)

    async def watch_model_instances(self):
        while True:
            try:
                logger.info("Started watching model instances.")
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failed watching model instances: {e}")
                await asyncio.sleep(5)

    async def monitor_error_instances(self):
        """Periodically checks cached ERROR state instances and attempts to restart them."""
        while True:
            try:
                logger.trace(
                    f"Monitoring error instances, instances: {self._error_model_instances.keys()}"
                )

                for mi_id, mi in list(self._error_model_instances.items()):
                    self._restart_error_instance(mi)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error while monitoring instances: {e}")
                await asyncio.sleep(5)

    def _handle_model_instance_event(self, event: Event):  # noqa: C901
        mi = ModelInstance.model_validate(event.data)

        logger.trace(
            f"Received model instance event: {event.type} {mi.name} {mi.state}"
        )

        is_main_worker = mi.worker_id == self._worker_id

        if is_main_worker:
            # Return if all subordinate workers aren't running.
            if (
                mi.distributed_servers
                and mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.RUN_FIRST
                and mi.distributed_servers.subordinate_workers
            ):
                ready = all(
                    sw.state == ModelInstanceStateEnum.RUNNING
                    for sw in mi.distributed_servers.subordinate_workers
                )
                if not ready:
                    logger.info(
                        f"Model instance {mi.name} waits for all subordinate workers to be ready."
                    )
                    return
        else:
            # Return if it isn't a distribution serving.
            if not mi.distributed_servers:
                return
            # Return if it's a delegated distribution,
            # which means the main worker is responsible for serving.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.DELEGATED
            ):
                return
            # Return if it isn't the member of the distribution serving.
            joined = any(
                sw.worker_id == self._worker_id
                for sw in mi.distributed_servers.subordinate_workers or []
            )
            if not joined:
                return
            # Return if the main worker isn't initialized.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                and mi.state
                not in [
                    ModelInstanceStateEnum.STARTING,
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.ERROR,
                ]
            ):
                logger.info(
                    f"Model instance {mi.name} waits for main worker {mi.worker_ip} to be initialized."
                )
                return
            # FIXME: This is a temporary solution to prevent the main worker from being unable to start due to phantom reads.
            #        We confirm whether the operation should be performed by checking the state of the earlier subordinate worker.
            for sw in mi.distributed_servers.subordinate_workers:
                if sw.worker_id == self._worker_id:
                    break
                if sw.state not in [
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.ERROR,
                ]:
                    logger.info(
                        f"Model instance {mi.name} waits for previous subordinate worker {sw.worker_ip} to be ready."
                    )
                    return

        if mi.state == ModelInstanceStateEnum.ERROR and event.type == EventType.DELETED:
            self._error_model_instances.pop(mi.id, None)
            return
        elif mi.state == ModelInstanceStateEnum.ERROR:
            m = self._get_model_with_cache(mi)
            if m.restart_on_error:
                self._error_model_instances[mi.id] = mi
            return
        elif mi.id in self._serving_model_instances and event.type == EventType.DELETED:
            self._stop_model_instance(mi)
        elif (
            mi.id in self._serving_model_instances
            and mi.state == ModelInstanceStateEnum.SCHEDULED
        ):
            # In case when the worker is offline and reconnected, the model instance state
            # is out of sync with existing serving process. Restart it.
            self._restart_serve_process(mi)
        elif (
            event.type in {EventType.CREATED, EventType.UPDATED}
        ) and not self._serving_model_instances.get(mi.id):
            self._start_serve_process(mi)

    def _start_serve_process(self, mi: ModelInstance):  # noqa: C901
        is_main_worker = mi.worker_id == self._worker_id

        # 创建基于模型名/副本名的日志目录结构
        model_log_dir = f"{self._serve_log_dir}/{mi.model_name}"
        replica_log_dir = f"{model_log_dir}/{mi.name}"
        os.makedirs(replica_log_dir, exist_ok=True)
        
        logger.debug(f"Created log directory for instance {mi.name} (ID: {mi.id}): {replica_log_dir}")
        
        # 更新实例ID到日志目录的映射文件，避免重复条目
        mapping_file = f"{self._serve_log_dir}/instance_mapping.txt"
        self._update_instance_mapping(mapping_file, mi.id, replica_log_dir)

        # 保留向后兼容的日志文件路径（用于老式日志查找）
        log_file_path = f"{self._serve_log_dir}/{mi.id}.log"
        if os.path.exists(log_file_path) and platform.system() != "windows":
            # TODO Windows does not support os.remove() on open files.
            # Investigate file occupation issue.
            os.remove(log_file_path)

        sw_pos: Optional[int] = None
        sw: Optional[ModelInstanceSubordinateWorker] = None
        if not is_main_worker:
            sw_pos = next(
                (
                    i
                    for i, sw in enumerate(mi.distributed_servers.subordinate_workers)
                    if sw.worker_id == self._worker_id
                ),
            )
            sw = mi.distributed_servers.subordinate_workers[sw_pos]

        try:
            model = self._get_model_with_cache(mi)
            backend = get_backend(model)

            # Assign port.
            if not mi.port:
                if self._serving_model_instance_ports:
                    unavailable_ports = set.union(
                        *self._serving_model_instance_ports.values()
                    )
                else:
                    unavailable_ports = set()
                mi.port = network.get_free_port(
                    port_range=self._config.service_port_range,
                    unavailable_ports=unavailable_ports,
                )
                mi.ports = [mi.port]
                if (
                    mi.distributed_servers
                    and mi.distributed_servers.subordinate_workers
                ):
                    if backend == BackendEnum.ASCEND_MINDIE:
                        # Get port for subordinate worker watching.
                        unavailable_ports.add(mi.port)
                        connecting_port = network.get_free_port(
                            port_range=self._config.service_port_range,
                            unavailable_ports=unavailable_ports,
                        )
                        mi.ports.append(connecting_port)

            logger.info(
                f"Starting model instance {mi.name}"
                f"{'' if not is_main_worker else f'on port {mi.ports if mi.ports else [mi.port]}'}"
            )

            process = multiprocessing.Process(
                target=ServeManager.serve_model_instance,
                args=(
                    mi,
                    backend,
                    self._clientset.headers,
                    replica_log_dir,
                    self._config,
                    self._worker_id,
                ),
            )
            process.daemon = False
            process.start()
            self._serving_model_instances[mi.id] = process
            self._serving_model_instance_ports[mi.id] = set(mi.ports)
            self._starting_model_instances[mi.id] = mi

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.INITIALIZING,
                    "port": mi.port,
                    "ports": mi.ports,
                    "pid": process.pid,
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.INITIALIZING
                # For Ascend MindIE, the state is set to RUNNING directly,
                if backend == BackendEnum.ASCEND_MINDIE:
                    sw.state = ModelInstanceStateEnum.RUNNING
                sw.pid = process.pid
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.info(
                f"Started model instance {mi.name} "
                f"{'' if not is_main_worker else f'on port {mi.ports if mi.ports else [mi.port]}'}, "
                f"pid {process.pid}"
            )

        except Exception as e:
            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.ERROR,
                    "state_message": f"Failed to start model instance: {e}",
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.ERROR
                sw.state_message = f"Failed to start model instance: {e}"
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to start model instance {mi.name}: {e}")

    def _restart_serve_process(self, mi: ModelInstance):
        logger.debug(f"Restart serving model instance {mi.name}")
        self._stop_model_instance(mi)
        self._start_serve_process(mi)

    @staticmethod
    def serve_model_instance(
        mi: ModelInstance,
        backend: BackendEnum,
        client_headers: dict,
        log_dir_path: str,
        cfg: Config,
        worker_id: int,
    ):

        setproctitle.setproctitle(f"gpustack_serving_process: model_instance_{mi.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.server_url,
            headers=client_headers,
        )

        # 使用按日期轮转的日志文件
        rotating_log = DailyRotatingLogFile(log_dir_path)
        
        # 确保日志文件立即创建并可写入
        test_msg = f"Model instance {mi.name} (ID: {mi.id}) started at {datetime.now()}\n"
        rotating_log.write(test_msg)
        
        try:
            with RedirectStdoutStderr(rotating_log):
                if backend == BackendEnum.LLAMA_BOX:
                    LlamaBoxServer(clientset, mi, cfg, worker_id).start()
                elif backend == BackendEnum.VLLM:
                    VLLMServer(clientset, mi, cfg, worker_id).start()
                elif backend == BackendEnum.VOX_BOX:
                    VoxBoxServer(clientset, mi, cfg, worker_id).start()
                elif backend == BackendEnum.ASCEND_MINDIE:
                    AscendMindIEServer(clientset, mi, cfg, worker_id).start()
                else:
                    raise ValueError(f"Unsupported backend {backend}")
        finally:
            rotating_log.close()

    def _update_instance_mapping(self, mapping_file: str, instance_id: int, log_dir: str):
        """更新实例ID到日志目录的映射，避免重复条目"""
        mappings = {}
        
        # 读取现有映射
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and ':' in line:
                            inst_id, dir_path = line.split(':', 1)
                            try:
                                mappings[int(inst_id)] = dir_path
                            except ValueError:
                                continue
            except Exception:
                pass
        
        # 更新当前实例的映射
        mappings[instance_id] = log_dir
        
        # 写回映射文件
        try:
            with open(mapping_file, "w", encoding="utf-8") as f:
                for inst_id, dir_path in mappings.items():
                    f.write(f"{inst_id}:{dir_path}\n")
        except Exception:
            pass  # 映射文件更新失败不影响主流程

    def _remove_instance_mapping(self, mapping_file: str, instance_id: int):
        """从映射文件中移除指定实例的映射"""
        if not os.path.exists(mapping_file):
            return
            
        mappings = {}
        
        # 读取现有映射
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        inst_id, dir_path = line.split(':', 1)
                        try:
                            if int(inst_id) != instance_id:
                                mappings[int(inst_id)] = dir_path
                        except ValueError:
                            continue
        except Exception:
            return
        
        # 写回映射文件
        try:
            with open(mapping_file, "w", encoding="utf-8") as f:
                for inst_id, dir_path in mappings.items():
                    f.write(f"{inst_id}:{dir_path}\n")
        except Exception:
            pass

    def _update_model_instance(self, id: str, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            set_attr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)

    def _get_model_with_cache(self, mi: ModelInstance) -> Model:
        """Get model from cache or fetch from clientset."""
        if mi.id in self._model_cache_by_instance:
            return self._model_cache_by_instance[mi.id]

        model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = model
        return model

    def _stop_model_instance(self, mi: ModelInstance):
        instance_name = mi.name or f"ModelInstance-{mi.id}"
        if mi.id not in self._serving_model_instances:
            logger.warning(f"Model instance {instance_name} is not running. Skipping.")
            return
        else:
            pid = self._serving_model_instances[mi.id].pid
            try:
                terminate_process_tree(pid)
                logger.info(f"Stopped model instance {instance_name} with pid {pid}.")
            except psutil.NoSuchProcess:
                logger.warning(
                    f"Model instance {instance_name} with pid {pid} is already stopped."
                )
                pass
            except Exception as e:
                logger.error(f"Failed to stop model instance: {pid}: {e}")

            self._post_stop_model_instance(mi)

    def _restart_error_instance(self, mi: ModelInstance):
        """Attempts to restart a model instance that is in error state with exponential backoff."""
        if mi.id in self._serving_model_instances:
            logger.warning(
                f"Model instance {mi.name} is already running, skipping restart."
            )
            return

        restart_count = mi.restart_count or 0
        last_restart_time = mi.last_restart_time or mi.updated_at

        current_time = datetime.now(timezone.utc)
        delay = min(
            10 * (2 ** (restart_count - 1)), 300
        )  # Exponential backoff, max 5 minutes
        if last_restart_time:
            elapsed_time = (current_time - last_restart_time).total_seconds()
            if elapsed_time < delay:
                logger.trace(
                    f"Delaying restart of {mi.name} for {delay - elapsed_time:.2f} seconds."
                )
                return

        logger.info(
            f"Restarting model instance {mi.name} (attempt {restart_count + 1}) after {delay} seconds delay."
        )

        self._update_model_instance(
            mi.id,
            restart_count=restart_count + 1,
            last_restart_time=current_time,
            state=ModelInstanceStateEnum.SCHEDULED,
            state_message="",
        )

        self._error_model_instances.pop(mi.id, None)

    def health_check_serving_instances(self):  # noqa: C901
        for id, process in list(self._serving_model_instances.items()):
            # Skip if the process is alive and not in starting instances.
            if process.is_alive() and id not in self._starting_model_instances:
                continue
            try:
                mi = self._clientset.model_instances.get(id=id)
            except NotFoundException as e:
                logger.warning(
                    f"Model instance {id} not found because of {str(e)}, stopping serving process."
                )
                self._stop_model_instance(ModelInstance(id=id))
                continue

            is_main_worker = mi.worker_id == self._worker_id

            # Monitor inference server process exit
            if not process.is_alive():
                try:
                    if mi.state != ModelInstanceStateEnum.ERROR:
                        # Get patch dict for main worker.
                        if is_main_worker:
                            patch_dict = {
                                "state": ModelInstanceStateEnum.ERROR,
                                "state_message": f"Inference server exited with code {process.exitcode}.",
                            }
                        # Get patch dict for subordinate worker.
                        else:
                            sw_pos = next(
                                (
                                    i
                                    for i, sw in enumerate(
                                        mi.distributed_servers.subordinate_workers
                                    )
                                    if sw.worker_id == self._worker_id
                                ),
                            )
                            sw = mi.distributed_servers.subordinate_workers[sw_pos]
                            sw.state = ModelInstanceStateEnum.ERROR
                            sw.state_message = (
                                f"Inference server exited with code {process.exitcode}."
                            )
                            patch_dict = {
                                f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                            }
                        # Update model instance.
                        self._update_model_instance(mi.id, **patch_dict)
                except NotFoundException:
                    pass
                # Post process stop model instance.
                self._post_stop_model_instance(mi)
                return

            # Otherwise, check if the process is ready to serve.
            model = self._get_model_with_cache(mi)
            backend = get_backend(model)
            try:
                # Get patch dict for main worker.
                if is_main_worker:
                    sw_error_msg = None
                    if (
                        mi.distributed_servers
                        and mi.distributed_servers.subordinate_workers
                    ):
                        for sw in mi.distributed_servers.subordinate_workers:
                            if sw.state == ModelInstanceStateEnum.ERROR:
                                sw_error_msg = f"Distributed serving error in subordinate worker {sw.worker_ip}: {sw.state_message}."
                                break
                    # If there is no error message from subordinate workers,
                    # check the main worker's health.
                    if not sw_error_msg:
                        if not is_ready(backend, mi):
                            continue
                        if mi.state == ModelInstanceStateEnum.RUNNING:
                            continue
                        patch_dict = {
                            "state": ModelInstanceStateEnum.RUNNING,
                            "state_message": "",
                        }
                    # Otherwise, update the main worker state to ERROR.
                    else:
                        patch_dict = {
                            "state": ModelInstanceStateEnum.ERROR,
                            "state_message": sw_error_msg,
                        }
                # Get patch dict for subordinate worker.
                else:
                    # Skip health check for Ascend MindIE subordinate workers
                    if backend == BackendEnum.ASCEND_MINDIE:
                        # Remove from starting instances directly.
                        self._starting_model_instances.pop(mi.id, None)
                        continue
                    # Otherwise, update subordinate worker state to RUNNING.
                    sw_pos = next(
                        (
                            i
                            for i, sw in enumerate(
                                mi.distributed_servers.subordinate_workers
                            )
                            if sw.worker_id == self._worker_id
                        ),
                    )
                    sw = mi.distributed_servers.subordinate_workers[sw_pos]
                    if sw.state == ModelInstanceStateEnum.RUNNING:
                        continue
                    sw.state = ModelInstanceStateEnum.RUNNING
                    sw.state_message = ""
                    patch_dict = {
                        f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                    }
                # Update model instance.
                self._update_model_instance(mi.id, **patch_dict)
                # Remove from starting instances if it was started.
                self._starting_model_instances.pop(mi.id, None)
            except NotFoundException:
                pass

    def _post_stop_model_instance(self, mi: ModelInstance):
        """
        Post process after stopping a model instance.
          - Remove from serving model instances.
          - Remove ports from serving model instance ports.
          - Remove from starting model instances.
          - Remove from model cache by instance.
          - Remove from instance mapping file.
        """

        self._serving_model_instances.pop(mi.id, None)
        self._serving_model_instance_ports.pop(mi.id, None)
        self._starting_model_instances.pop(mi.id, None)
        self._model_cache_by_instance.pop(mi.id, None)
        
        # 清理映射文件中的条目
        mapping_file = f"{self._serve_log_dir}/instance_mapping.txt"
        self._remove_instance_mapping(mapping_file, mi.id)


def is_ready(backend: str, mi: ModelInstance) -> bool:
    """
    Access the health endpoint of the given model instance to check if it is servable.
    """

    try:
        hostname = "127.0.0.1"
        if backend == BackendEnum.ASCEND_MINDIE:
            # Connectivity to the loopback address does not work for Ascend MindIE.
            # Use worker IP instead.
            hostname = mi.worker_ip

        # Check /v1/models by default if dedicated health check endpoint is not available.
        # This is served by all backends (llama-box, vox-box, vllm, mindIE)
        health_check_url = f"http://{hostname}:{mi.port}/v1/models"
        if backend == BackendEnum.LLAMA_BOX:
            # For llama-box, use /health to avoid printing error logs.
            health_check_url = f"http://{hostname}:{mi.port}/health"

        response = requests.get(health_check_url, timeout=1)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False
