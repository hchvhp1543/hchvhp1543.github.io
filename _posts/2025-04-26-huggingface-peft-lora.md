---
title: Hugginface Peft - Lora
description: Hugginface Peft - Lora
author: hchvhp1543
date: 2025-04-26 11:33:00 +0900
categories: [AI, NLP, DL]
tags: [AI, NLP, DL, Hugginface, Peft, Lora]
pin: false
math: true
mermaid: true

---

## Lora란?
- 중요 필드

## 기본 코드

```python
from peft import LoraConfig, TaskType, get_peft_model
import torch
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    # r=16,
    # target_modules=["q_proj", "v_proj"],
    # task_type=TaskType.CAUSAL_LM,
    # lora_alpha=32,
    # lora_dropout=0.05
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
```

## Lora 적용 전
```text
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(110592, 1024, padding_idx=100257)
    (layers): ModuleList(
      (0-23): 24 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=1024, out_features=4096, bias=False)
          (up_proj): Linear(in_features=1024, out_features=4096, bias=False)
          (down_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((1024,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=110592, bias=False)
)
```

## Lora 적용 후

- version 1
```python
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(110592, 1024, padding_idx=100257)
        (layers): ModuleList(
          (0-23): 24 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=2048, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear(in_features=1024, out_features=4096, bias=False)
              (up_proj): Linear(in_features=1024, out_features=4096, bias=False)
              (down_proj): Linear(in_features=4096, out_features=1024, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((1024,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): Linear(in_features=1024, out_features=110592, bias=False)
    )
  )
)
```

- version 2

```python
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(110592, 1024, padding_idx=100257)
        (layers): ModuleList(
          (0-23): 24 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=2048, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
            )
            (mlp): LlamaMLP(
              (gate_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=4096, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (up_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=4096, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (down_proj): lora.Linear(
                (base_layer): Linear(in_features=4096, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((1024,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): ModulesToSaveWrapper(
        (original_module): Linear(in_features=1024, out_features=110592, bias=False)
        (modules_to_save): ModuleDict(
          (default): Linear(in_features=1024, out_features=110592, bias=False)
        )
      )
    )
  )
)
```

- 변경점

```python
(q_proj): Linear(in_features=1024, out_features=2048, bias=False)

-->

(q_proj): lora.Linear(
  (base_layer): Linear(in_features=1024, out_features=2048, bias=False)
  (lora_dropout): ModuleDict(
    (default): Dropout(p=0.05, inplace=False)
  )
  (lora_A): ModuleDict(
    (default): Linear(in_features=1024, out_features=16, bias=False)
  )
  (lora_B): ModuleDict(
    (default): Linear(in_features=16, out_features=2048, bias=False)
  )
  (lora_embedding_A): ParameterDict()
  (lora_embedding_B): ParameterDict()
  (lora_magnitude_vector): ModuleDict()
)
```

- 변경 후 파라미터

```text
print(name, param.requires_grad, param.shape, sep=" | ")

model.embed_tokens.weight | True | torch.Size([110592, 1024])

model.layers.0.self_attn.q_proj.weight | True | torch.Size([2048, 1024])

model.layers.0.self_attn.k_proj.weight | True | torch.Size([1024, 1024])

model.layers.0.self_attn.v_proj.weight | True | torch.Size([1024, 1024])

model.layers.0.self_attn.o_proj.weight | True | torch.Size([1024, 2048])

model.layers.0.mlp.gate_proj.weight | True | torch.Size([4096, 1024])

model.layers.0.mlp.up_proj.weight | True | torch.Size([4096, 1024])

model.layers.0.mlp.down_proj.weight | True | torch.Size([1024, 4096])

model.layers.0.input_layernorm.weight | True | torch.Size([1024])

model.layers.0.post_attention_layernorm.weight | True | torch.Size([1024])
...
model.norm.weight | True | torch.Size([1024])
```

```text
print(name, param.requires_grad, param.shape, sep=" | ")

base_model.model.model.embed_tokens.weight | False | torch.Size([110592, 1024])
base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight | False | torch.Size([2048, 1024])
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight | True | torch.Size([16, 1024])
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight | True | torch.Size([2048, 16])

base_model.model.model.layers.0.self_attn.k_proj.base_layer.weight | False | torch.Size([1024, 1024])
base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight | True | torch.Size([16, 1024])
base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight | True | torch.Size([1024, 16])

base_model.model.model.layers.0.self_attn.v_proj.base_layer.weight | False | torch.Size([1024, 1024])
base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight | True | torch.Size([16, 1024])
base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight | True | torch.Size([1024, 16])

base_model.model.model.layers.0.self_attn.o_proj.base_layer.weight | False | torch.Size([1024, 2048])
base_model.model.model.layers.0.self_attn.o_proj.lora_A.default.weight | True | torch.Size([16, 2048])
base_model.model.model.layers.0.self_attn.o_proj.lora_B.default.weight | True | torch.Size([1024, 16])

base_model.model.model.layers.0.mlp.gate_proj.base_layer.weight | False | torch.Size([4096, 1024])
base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight | True | torch.Size([16, 1024])
base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight | True | torch.Size([4096, 16])

base_model.model.model.layers.0.mlp.up_proj.base_layer.weight | False | torch.Size([4096, 1024])
base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight | True | torch.Size([16, 1024])
base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight | True | torch.Size([4096, 16])

base_model.model.model.layers.0.mlp.down_proj.base_layer.weight | False | torch.Size([1024, 4096])
base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight | True | torch.Size([16, 4096])
base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight | True | torch.Size([1024, 16])

base_model.model.model.layers.0.input_layernorm.weight | False | torch.Size([1024])

base_model.model.model.layers.0.post_attention_layernorm.weight | False | torch.Size([1024])
...
base_model.model.model.norm.weight | False | torch.Size([1024])
base_model.model.lm_head.modules_to_save.default.weight | True | torch.Size([110592, 1024])
```
