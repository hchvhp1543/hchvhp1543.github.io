---
title: Huggingface, Lightning - log 메소드
description: Huggingface, Lightning - log 메소드
author: hchvhp1543
date: 2025-04-22 11:33:00 +0900
categories: [AI, NLP, DL]
tags: [AI, NLP, DL, Huggingface, Lightning]
pin: false
math: true
mermaid: true

---

## Huggingface - logs
- log(logs:dict)
  ```python
  self.log({"train_loss": loss.item()})
  ```
- Trainer에서 log()를 호출하면 내부적으로 trainer.state.log_history에 누적 저장. 시간순으로 저장되는 list of dict
  ```python
  [ 
      {"loss": 0.523, "learning_rate": 3e-5, "epoch": 1.0, "step": 100},
      {"eval_loss": 0.401, "eval_accuracy": 0.88, "epoch": 1.0, "step": 200},
       ...
  ]
  ```
- log 호출 시 내부적으로 callback 의 on_log() 메서드가 호출되고, 거기서 WandB, TensorBoard 등으로 전달
- 소스 코드

```python
def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
    # 생략
    output = {**logs, **{"step": self.state.global_step}}
    self.state.log_history.append(output)
    self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
```


# Lightning - logs
- Trainer 내부적으로 3군데 저장
  - Trainer.callback_metrics
    - on_epoch=True일 때 저장. 여러 step 동안 기록된 값을 평균내서 callback_metrics에 저장
    - 내부적으로 각 step에서 기록한 train_loss들을 리스트로 누적. epoch 끝에 평균 내서 callback_metrics[key]에 저장
    - validation_epoch_end() 후에 접근 가능 
  - Trainer.logged_metrics
    - on_step=True일 때 저장
    - 평균 내지 않고 그냥 현재 step 값 저장
  - Logger (e.g. TensorBoard, WandB 등)
- 예시
  ```python
  def training_step(self, batch, batch_idx):
      loss = self.compute_loss(batch)
      self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      return loss
  
  # trainer.logged_metrics["train_loss"] → 현재 스텝에서 기록된 값
  # trainer.callback_metrics["train_loss"] → 전체 에폭에서 집계된 값 (평균 등)
  ```

- 소스 코드

```python
def training_step(self, batch, batch_idx):
    self.log("a_val", 2.0)

callback_metrics = trainer.callback_metrics
assert callback_metrics["a_val"] == 2.0
```

```python
@property
def callback_metrics(self) -> _OUT_DICT:
    return self._logger_connector.callback_metrics

@property
def logged_metrics(self) -> _OUT_DICT:
    return self._logger_connector.logged_metrics
```
