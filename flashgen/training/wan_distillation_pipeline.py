# SPDX-License-Identifier: Apache-2.0
import sys

from flashgen.flashgen_args import FlashgenArgs, TrainingArgs
from flashgen.logger import init_logger
from flashgen.models.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
from flashgen.training.distillation_pipeline import DistillationPipeline

logger = init_logger(__name__)


class WanDistillationPipeline(DistillationPipeline):
    """
    A distillation pipeline for Wan that uses a single transformer model.
    The main transformer serves as the student model, and copies are made for teacher and critic.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, flashgen_args: FlashgenArgs):
        """Initialize Wan-specific scheduler."""
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=flashgen_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Validation video generation is disabled in train-only Flashgen.")
        self.validation_pipeline = None


def main(args) -> None:
    logger.info("Starting Wan distillation pipeline...")

    # Create pipeline with original args
    pipeline = WanDistillationPipeline.from_pretrained(args.pretrained_model_name_or_path, args=args)

    args = pipeline.training_args
    # Start training
    pipeline.train()
    logger.info("Wan distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from flashgen.flashgen_args import TrainingArgs
    from flashgen.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FlashgenArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
