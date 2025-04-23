---
title: Huggingface Transformers - Trainer
description: Huggingface Transformers - Trainer
author: hchvhp1543
date: 2025-04-19 11:33:00 +0900
categories: [AI, NLP, DL]
tags: [AI, NLP, DL, Huggingface, Transformers, Trainer]
pin: false
math: true
mermaid: true

---


Data 관련
- raw data
- dataset
- data_loader
- data_collator

Model 관련
- model
- loss_function
- optimizer (+scheduler)

학습 관련
- epoch/step
- batch_size
- logging : progress bar, visualization, metric
- saving : checkpoint
- metric

```
```

## Trainer
- args :
  - 가장 중요 
  - (TrainingArguments, optional) 
- data_collator
  - (DataCollator, optional)
- train_dataset / eval_dataset
  - (torch.utils.data.IterableDataset, datasets.Dataset)
- processing_class
  - 보통 tokenizer
  - PreTrainedTokenizerBase
- compute_metrics
- callbacks
- optimizers


Customize
- get_train/evel/test_dataloader() + eval, test
- logs()
- create_optimizer_and_scheduler()	
- compute_loss()
- training_step()
- prediction_step() / evaluate() / predict()

자주 쓰는 메소드
- train
- evaluate / predict
- save_model / save_state
- log_metrics / save_metrics


참고 : https://huggingface.co/docs/transformers/en/trainer
```python
from transformers import Trainer

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}
      
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback()],
)

trainer.train()
```

```python

def compute_loss(self, model, inputs, return_outputs=False):
    outputs = model(**inputs)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    return (loss, outputs) if return_outputs else loss
```


```python
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```


```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

####
# Metrics!
if (
    self.compute_metrics is not None
    and all_preds is not None
    and all_labels is not None
    and not self.args.batch_eval_metrics
):
    eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
    eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
    metrics = self.compute_metrics(
        EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
    )
elif metrics is None:
    metrics = {}

```


## TrainingArgs
- 공통
  - ★ output_dir 
- 학습, 평가
  - ★ per_device_train(eval)_batch_size
  - ★ num_train_epochs /max_steps
  - bf16 / fp16
  - gradient_checkpointing
  - gradient_accumulation_steps
  - max_grad_norm
  - eval_strategy : no / steps / epoch
  - load_best_model_at_end
  - accelerator_config
- logging
  - ★ logging_dir : output_dir/runs/CURRENT_DATETIME_HOSTNAME
  - ★ logging_strategy : no / steps / epoch
  - ★ logging_steps
  - ★ report_to : "tensorboard"
- saving
  - ★ save_strategy : no / steps / epoch
  - ★ save_steps
  - save_total_limit
- optimizer
  - ★ learning_rate
  - weight_decay
  - lr_scheduler_type
  - warmup_ratio
  - optim
- 기타
  - do_train / do_eval / do_predict : script 파일 사용시 활용
  - resume_from_checkpoint : script 파일 사용시 활용
  - label_smoothing_factor
  - label_names
  - dataloader_drop_last
  - dataloader_num_workers

### 예시
-  https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora?hl=ko

```python
args = TrainingArguments(
    output_dir="gemma-text-to-sql",         # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    report_to="tensorboard",                # report metrics to tensorboard
)
```

```python
training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
```


### 연관성 있는 필드
- load_best_model_at_end : 이걸 True로 설정하면, Trainer가 metric_for_best_model 기준으로 성능이 가장 좋은 체크포인트를 찾고, 그걸 학습 마지막에 자동으로 불러와 trainer.model로 덮어씀
- metric_for_best_model : 체크포인트 저장 및 "최고 모델"을 판단할 때 사용할 metric 이름. load_best_model_at_end=True일 때 꼭 지정
  - 예: "eval_accuracy", "eval_loss", "eval_f1" 등 → compute_metrics에서 리턴한 딕셔너리 키와 동일해야 함
- greater_is_better : 위에서 지정한 metric이 클수록 좋은지, 작을수록 좋은지 설정
  - accuracy, f1, precision 등은 True 
  - loss, perplexity 등은 False
- save_total_limit

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)
```


> Must be the name of a metric returned by the evaluation with or without the prefix "eval_".

compute_metrics 함수가 리턴한 metric 이름을 기준으로 metric_for_best_model을 지정해야 하는데, 그 이름 앞에 eval_을 붙이든 안 붙이든 상관없다.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1_score(labels, preds)
    }

metric_for_best_model="accuracy"     # 가능
metric_for_best_model="eval_accuracy"  # 이것도 가능
```

---

torch.Tensor

Parameter
- param.data
- param.grad



---
Distributed Training
- DP (잘 사용 X)
- DDP
- FSDP
  - model parameters, gradients, and optimizer states
- DeepSpeed
- 참고
  - pytorch lightning
    - Strategy
      - SingleDeviceStrategy
      - ParallelStrategy
        - FSDPStrategy
        - DDPStrategy
        - DeepSpeedStrategy
        - ModelParallelStrategy
    - Accelerator
      - CPUAccelerator
      - CUDAAccelerator
