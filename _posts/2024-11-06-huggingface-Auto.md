---
layout: post
title: HF Transformers - Auto*
tags: [Katex, Mermaid, Markdown]
categories: Demo
---

# HF Transformers - Auto*

- Auto Config/Tokenizer/Model은 생성자 메소드(__init__)이 아닌 클래스 메소드(from_pretrained) 통해 생성
- 예시

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

model_name_or_path = "bert-base-uncased"

config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path)
```

- 각각에 맞게 필요한 파일들이 존재함
    - AutoConfig
        - (필수) config.json
    - AutoTokenizer
        - (필수) tokenzier_config.json
        - (필수) tokenzier.json
        - (거의 필수) vocab.txt / vocab.json … ← model에 따라 다름
        - (옵션) special_tokens_map.json
        - (옵션) added_tokens.json
    - AutoModel
        - model.safetensors
        - 또는 pytorch_model.bin

- Auto*.from_pretrained → 클래스 타입 확인 → 클래스.from_pretrained → 파일 로드 : json.loads, json.load  → 인스턴스 생성 : cls(…)

- AutoConfig

```python
config = AutoConfig.from_pretrained(model_name_or_path)

config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)

config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)

resolved_config_file

config_dict = cls._dict_from_json_file(resolved_config_file)

with open(json_file, "r", encoding="utf-8") as reader:
		text = reader.read()
return json.loads(text)

return config_class.from_dict(config_dict, **unused_kwargs)

config = cls(**config_dict)
```

- AutoTokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)

resolved_config_file

with open(resolved_config_file, encoding="utf-8") as reader:
		result = json.load(reader)
		
config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
)

return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

vocab_files = {**cls.vocab_files_names, **additional_files_names}

return cls._from_pretrained(
    resolved_vocab_files,
    pretrained_model_name_or_path,
    ...
)

with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
		init_kwargs = json.load(tokenizer_config_handle)
		
with open(tokenizer_file, encoding="utf-8") as tokenizer_file_handle:
		tokenizer_file_handle = json.load(tokenizer_file_handle)
		
tokenizer = cls(*init_inputs, **init_kwargs)
```

- AutoModel

```python
model = AutoModel.from_pretrained(model_name_or_path)

config, kwargs = AutoConfig.from_pretrained(
		pretrained_model_name_or_path,
		...
)

return model_class.from_pretrained(
		pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
)

from_pt = not (from_tf | from_flax)

filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
filename = _add_variant(WEIGHTS_NAME, variant)

resolved_archive_file

with safe_open(resolved_archive_file, framework="pt") as f:
		metadata = f.metadata()
		
state_dict = load_state_dict(resolved_archive_file)

with safe_open(filename, framework="pt", device=device) as f:
    for k in f.keys():
        result[k] = f.get_tensor(k)
        
model = cls(config, *model_args, **model_kwargs)

(
    model,
    missing_keys,
    unexpected_keys,
    mismatched_keys,
    offload_index,
    error_msgs,
) = cls._load_pretrained_model(
    model,
    state_dict,
    loaded_state_dict_keys,  # XXX: rename?
    resolved_archive_file,
    pretrained_model_name_or_path,
    ...
)
```