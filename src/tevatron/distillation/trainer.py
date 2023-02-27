import os
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch import nn
import logging


logger = logging.getLogger(__name__)

class DistilTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super(DistilTrainer, self).__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        if self.args.negatives_x_device:
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        query, passage, pair = inputs
        student_scores = model(query=query, passage=passage).scores
        with torch.no_grad():
            teacher_scores = self.teacher_model(pair=pair).scores
            if self.args.negatives_x_device:
                teacher_scores = self._dist_gather_tensor(teacher_scores)
        teacher_mat = torch.zeros(student_scores.shape, dtype=student_scores.dtype, device=teacher_scores.device)
        index = torch.arange(teacher_scores.size(0), device=teacher_scores.device)
        teacher_scores = torch.softmax(teacher_scores.view(student_scores.size(0), -1) / self.args.teacher_temp, dim=1, dtype=student_scores.dtype)
        teacher_mat = torch.scatter(teacher_mat,
                                    dim=-1,
                                    index=index.view(student_scores.size(0), -1), 
                                    src=teacher_scores)
        student_scores = nn.functional.log_softmax(student_scores / self.args.student_temp, dim=1)
        loss = nn.functional.kl_div(student_scores, teacher_mat, reduction='batchmean') * self._dist_loss_scale_factor
        return loss


    def training_step(self, *args):
        return super(DistilTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors