from tevatron.reranker.modeling import RerankerOutput
from tevatron.retriever.trainer import TevatronTrainer
from grad_cache import GradCache

def split_inputs(model_input, chunk_size):
    keys = list(model_input.keys())
    chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
    return [dict(zip(keys, tt)) for tt in zip(*chunked_tensors)]

def get_rep(x: RerankerOutput):
    return x.scores

class RerankerTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loss_fn = lambda x, y: self.compute_loss(self.model, {'input_ids': x, 'labels': y})
        self.gc = GradCache(
            models=[self.model],
            chunk_sizes=[self.args.gc_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_inputs,
            get_rep_fn=get_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        _distributed = self.args.local_rank > -1
        self.gc.models = [model]
        loss = self.gc(inputs, no_sync_except_last=_distributed)
        return loss
