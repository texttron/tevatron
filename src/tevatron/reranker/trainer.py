import os
from typing import Optional

from transformers.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(RerankerTrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def compute_loss(self, model, inputs):
        return model(inputs).loss
