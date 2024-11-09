---
layout: post
title: HF Tokenizer
tags: [Katex, Mermaid, Markdown]
categories: Demo
---

# tokenizer

- 환경
    - mac (m1)
    - pycharm
    - pytorch 1.13.1
    - transformers 4.42.3
    - tokenizers 0.19.1
    - datasets 2.19.2

---

예시 코드

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

model_name_or_path = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

text = "안녕하세요. 반갑습니다. 잘 부탁드립니다."
output = tokenizer(text, return_tensors=True)

texts = [
    "안녕하세요. 반갑습니다. 잘 부탁드립니다.",
    "감사합니다. 안녕히 계세요. 다음에 또 뵙겠습니다."
]
output = tokenizer(texts, return_tensors=True)

```

---

설명

- tokenizer 과정은 크게 2단계를 거침
    - encode : 텍스트를 토큰으로 나눈 후, 각 토큰을 인덱스 번호로 바꾸는 단계
    - prepare_for_model : 모델에 활용하기 위한 추가 작업 - pad, truncate
- 입력이 str인지, list[str] 인지에 따라 is_batched 결정.

- is_batched == True
    - encode
        - self.batch_encode_plus → self._batch_encode_plus
        - get_input_ids
            - [self.tokenize] → [self._tokenize : 모델별 tokenizer에 따라 구현이 다름]
            - [self.convert_tokens_to_ids → self._convert_token_to_id_with_added_voc] → [self._convert_token_to_id : 모델별 tokenizer에 따라 구현이 다름]
    - prepare_for_model
        - self._batch_prepare_for_model
        - for first_ids, second_ids in batch_ids_pairs:
            - self.prepare_for_model # we pad in batch afterward
                - self.truncate_sequences
                - self.pad
        - self.pad
- is_batched == False
    - encode
        - self.encode_plus → self._encode_plus
        - get_input_ids
            - self.prepare_for_model
                - self.truncate_sequences
                - self.pad

---

코드 흐름

```python
if is_batched:
    return self.batch_encode_plus(...)

###
return self._batch_encode_plus(...)

###
def get_input_ids(text):
    tokens = self.tokenize(text, **kwargs)
    return self.convert_tokens_to_ids(tokens)

tokenized_text.extend(self._tokenize(token))

ids.append(self._convert_token_to_id_with_added_voc(token))
return self._convert_token_to_id(token)

###
batch_outputs = self._batch_prepare_for_model(...)

###
batch_outputs = {}
for first_ids, second_ids in batch_ids_pairs:
    outputs = self.prepare_for_model(
        ...
        # we pad in batch afterward
    )
    
batch_outputs = self.pad(...)
    
########################

else:
    return self.encode_plus(...)
		
###
return self._encode_plus(...)

###
def get_input_ids(text):
    tokens = self.tokenize(text, **kwargs)
    return self.convert_tokens_to_ids(tokens)
    
tokenized_text.extend(self._tokenize(token))

ids.append(self._convert_token_to_id_with_added_voc(token))
return self._convert_token_to_id(token)

###
return self.prepare_for_model(...)

###
ids, pair_ids, overflowing_tokens = self.truncate_sequences(...)
encoded_inputs = self.pad(...)

########################

encoded_inputs = {}
encoded_inputs["overflowing_tokens"] = overflowing_tokens
encoded_inputs["num_truncated_tokens"] = total_len - max_length
encoded_inputs["input_ids"] = sequence
encoded_inputs["token_type_ids"] = token_type_ids
encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
encoded_inputs["length"] = len(encoded_inputs["input_ids"])

batch_outputs = BatchEncoding(
    encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
)
```