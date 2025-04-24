---
title: Huggingface Transformers - Trainer 심화
description: Huggingface Transformers - Trainer 심화
author: hchvhp1543
date: 2025-04-24 11:33:00 +0900
categories: [AI, NLP, DL]
tags: [AI, NLP, DL, Huggingface, Transformers, Trainer]
pin: false
math: true
mermaid: true

---

## metric(validation) 관련
- load_best_model_at_end
- metric_for_best_model
- greater_is_better

## grad 관련
- max_grad_norm (gradient clipping))

## gpu 메모리 관련
- gradient_checkpointing, gradient_checkpointing_kwargs
- fp16, bf16, half_precision_backend

## 분산학습 관련
- fsdp, fsdp_config
- accelerator_config
- deepspeed
- local_rank
- ddp_backend, ddp_timeout, ddp_find_unused_parameters



## _maybe_log_save_evaluate
- 학습 뿐만 아니라 로깅 / 저장 / 평가 도 참 중요하다

```python

tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
    # loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        # outputs = model(**inputs)
    # self.accelerator.backward(loss, **kwargs)

self.optimizer.step()

self._maybe_log_save_evaluate(...)
```

```python
## 1. log
logs: dict[str, float] = {}

logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
if grad_norm is not None:
    logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
if learning_rate is not None:
    logs["learning_rate"] = learning_rate
else:
    logs["learning_rate"] = self._get_learning_rate()

self.log(logs, start_time)

## 2. evaluate
metrics = None
if self.control.should_evaluate:
    metrics = self._evaluate(trial, ignore_keys_for_eval)

## 3. save
if self.control.should_save:
    self._save_checkpoint(model, trial)

```


1. log
   - trainer.state 에 기록한다
   - 각종 callback에 값을 넘겨 처리하게끔 한다 (tensorboard, wandb, ...)
   - tensorboard 저장 위치
     - log_dir = log_dir or args.logging_dir
     - self.tb_writer = self._SummaryWriter(log_dir=log_dir)
     - log_dir = os.path.join(args.logging_dir, trial_name)
     - self.tb_writer.add_text(k, v, ...) / self.tb_writer.add_scalar(k, v, ...)
   - 학습, 평가 모두 로깅함

2. evaluate
   - metric 을 구한다. metric 의 key에는 prefix 'eval_' 이 붙는다. (로깅시 구분 가능)
   - compute_metrics 메소드 사용
     - metrics = self.compute_metrics(EvalPrediction(predictions=logits, label_ids=labels, ...), ...)

3. save
   - rotate를 한다
   - checkpoint 저장 위치
     - checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
       - PREFIX_CHECKPOINT_DIR = "checkpoint"
     - run_dir = self._get_output_dir(trial=trial)
       - run_dir = self.args.output_dir
     - output_dir = os.path.join(run_dir, checkpoint_folder)
