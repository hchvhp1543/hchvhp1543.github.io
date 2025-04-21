---
title: TorchMetrics - 기초
description: TorchMetrics - 기초
author: hchvhp1543
date: 2025-04-20 11:33:00 +0900
categories: [AI, NLP, DL]
tags: [AI, NLP, DL, TorchMetrics, metric]
pin: false
math: true
mermaid: true

---

## TorchMetrics 핵심 메소드
- `__init__` : 객체 생성
- update : 한 배치(batch)의 결과를 내부 상태(state)에 누적 저장할 때 사용. 계산하지 않고, 내부적으로 변수(preds, targets)만 기록/저장함
- compute : update()로 누적한 모든 결과를 기반으로 최종 metric을 계산할 때 사용
- forward (`__call__`) : update() + compute() 한번에 실행

## 객체 구성
- nn.Module > Metric > Accuracy
- 내부에 state 존재
- Accuracy 경우, tp/fp/tn/fn 존재


## 코드 들여다보기
- 객체 생성

```python
accuracy = Accuracy(task="multiclass", num_classes=2)

return MulticlassAccuracy(num_classes, top_k, average, **kwargs)

self._create_state(
    size=1 if (average == "micro" and top_k == 1) else (num_classes or 1), multidim_average=multidim_average
)

self.add_state("tp", default(), dist_reduce_fx=dist_reduce_fx)
self.add_state("fp", default(), dist_reduce_fx=dist_reduce_fx)
self.add_state("tn", default(), dist_reduce_fx=dist_reduce_fx)
self.add_state("fn", default(), dist_reduce_fx=dist_reduce_fx)

if isinstance(default, Tensor):
    default = default.contiguous()
setattr(self, name, default)
self._defaults[name] = deepcopy(default)
```

- metric 값 계산

```python
target = tensor([0, 1, 0, 1])
preds = tensor([1, 1, 1, 1])
acc = accuracy(preds, target)

def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
    # calculate batch state and compute batch value
    self.update(*args, **kwargs)
    batch_val = self.compute()

def update(self, preds: Tensor, target: Tensor) -> None:
    self._update_state(tp, fp, tn, fn)

def _update_state(self, tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> None:
    self.tp = self.tp + tp if not isinstance(self.tp, list) else [*self.tp, tp]
    self.fp = self.fp + fp if not isinstance(self.fp, list) else [*self.fp, fp]
    self.tn = self.tn + tn if not isinstance(self.tn, list) else [*self.tn, tn]
    self.fn = self.fn + fn if not isinstance(self.fn, list) else [*self.fn, fn]

def compute(self) -> Tensor:
    tp, fp, tn, fn = self._final_state()
    return _accuracy_reduce(
        tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average, top_k=self.top_k
    )

def _final_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    tp = dim_zero_cat(self.tp)
    fp = dim_zero_cat(self.fp)
    tn = dim_zero_cat(self.tn)
    fn = dim_zero_cat(self.fn)
    return tp, fp, tn, fn
```

## lightning 코드와 함께 쓰기
- 참고 : https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py

```python
def __init__(self, ):
    # metric objects for calculating and averaging accuracy across batches
    self.train_acc = Accuracy(task="multiclass", num_classes=10)
    self.val_acc = Accuracy(task="multiclass", num_classes=10)
    self.test_acc = Accuracy(task="multiclass", num_classes=10)

    # for averaging loss across batches
    self.train_loss = MeanMetric()
    self.val_loss = MeanMetric()
    self.test_loss = MeanMetric()

    # for tracking best so far validation accuracy
    self.val_acc_best = MaxMetric()

def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
    loss, preds, targets = self.model_step(batch)

    # update and log metrics
    self.val_loss(loss)
    self.val_acc(preds, targets)
    self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

def on_validation_epoch_end(self) -> None:
    "Lightning hook that is called when a validation epoch ends."
    acc = self.val_acc.compute()  # get current val acc
    self.val_acc_best(acc)  # update best so far val acc
    # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
    # otherwise metric would be reset by lightning after each epoch
    self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
```
