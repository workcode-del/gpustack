import logging
from typing import List

from gpustack.policies.base import ModelInstanceScore, ModelInstanceScorer
from gpustack.schemas.models import Model, ModelInstance

MaxScore = 100

logger = logging.getLogger(__name__)


class OffloadLayerScorer(ModelInstanceScorer):
    def __init__(self, model: Model):
        self._model = model

    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        """
        Score the instances with offload layers.
        """

        logger.debug(
            f"model {self._model.name}, score instances with offload layer policy"
        )

        scored_instances = []
        for instance in instances:

            if instance.computed_resource_claim is None:
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            if (
                instance.computed_resource_claim.total_layers is None
                or instance.computed_resource_claim.offload_layers is None
            ):
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            score = 0
            total_layers = instance.computed_resource_claim.total_layers
            offload_layers = instance.computed_resource_claim.offload_layers

            if total_layers == offload_layers:
                score = MaxScore
            else:
                score = offload_layers / total_layers * MaxScore

            scored_instances.append(
                ModelInstanceScore(model_instance=instance, score=score)
            )

        return scored_instances
