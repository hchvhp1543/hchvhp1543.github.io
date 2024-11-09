---
layout: post
title: HF Dataset - map - Tokenizer
tags: [Katex, Mermaid, Markdown]
categories: Demo
---

# dataset - map - tokenizer

- 환경
    - mac (m1)
    - pycharm
    - pytorch 1.13.1
    - transformers 4.42.3
    - tokenizers 0.19.1
    - datasets 2.19.2

---

코드 예시

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

model_name_or_path = "bert-base-uncased"

from datasets import load_dataset

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

    """
    examples : LazyBatch/LazyRow
    - data : {dict: 3}
      - 'text' : {list: 1000}  / [hi, hello, bye, ...]
      - 'label' : {list: 1000} / [0, 0, 0, ...]

    tokenizer() : BatchEncoding
    - data : {dict: 3}
      - 'input_ids' : {list: 1000}      / [[1,2,3,...], [4,5,6,...], ...]
      - 'token_type_ids' : {list: 1000} / [[0,0,0,...], [0,0,0,...], ...]
      - 'attention_mask' : {list: 1000} / [[1,1,1,...], [1,1,1,...], ...]
    """

tokenized_imdb = imdb.map(preprocess_function, batched=True)
tokenized_imdb = imdb.map(preprocess_function, batched=False)

```

---

배경 지식

- tokenizer 의 결과 : BatchEncoding ← UserDict ← MutableMapping ← Mapping
- Dataset(pyarrow.Table) 불러오기 : LazyBatch/LazyRow ← LazyDict ← MutableMapping ← Mapping
- Mapping은 dict와 거의 동일
- UserDict/LazyDict/BatchEncoding는 데이터를 self.data 필드(dict)에 저장
- __getitem__, __setitem__ 메소드를 통해 self.data 필드에 값 저장 및 불러옴

```python
from collections.abc import MutableMapping, Mapping
from collections import UserDict

from transformers import BatchEncoding
from datasets.formatting.formatting import LazyDict, LazyRow, LazyBatch
```

- UserDict + UserDict 결합(merge)시 keys, __getitem__ 메소드 사용

```python
from collections import UserDict
a = UserDict({"a" : 1})
b = UserDict({"b" : 1})
c = {**a, **b}

def keys(self):
    "D.keys() -> a set-like object providing a view on D's keys"
    return KeysView(self)
    
def __getitem__(self, key):
    if key in self.data:
        return self.data[key]
```

---

UserDict

```python
class UserDict(_collections_abc.MutableMapping):
    def __init__(self, dict=None, /, **kwargs):
        self.data = {}
        if dict is not None:
            self.update(dict)
        if kwargs:
            self.update(kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        ...

    def __setitem__(self, key, item):
        self.data[key] = item
        

class MutableMapping(Mapping):
    def update(self, other=(), /, **kwds):
        if isinstance(other, Mapping):
            for key in other:
                self[key] = other[key] # __setitem__ 으로 이동
            ...
```

LazyDict

```python
class LazyDict(MutableMapping):
    def __init__(self, pa_table: pa.Table, formatter: "Formatter"):
        self.data = {key: None for key in pa_table.column_names}
				
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, key):
        value = self.data[key]
        ...
        return value
        
    def __setitem__(self, key, value):
        ...
        self.data[key] = value
		    
```

BatchEncoding

```python
class BatchEncoding(UserDict):
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        ...
    ):
        super().__init__(data)
		    
    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:
        if isinstance(item, str):
            return self.data[item]
        ...
        elif isinstance(item, slice):
            return {key: self.data[key][item] for key in self.data.keys()}
            
    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()
```

---

Dataset/DatasetDict map 메소드

- 일부 dataset (batch/example)의 결과와 function의 결과값을 merge
- dict(Mapping)과 dict(Mapping)의 합

코드 흐름

```python
def map(
    self,
    function: Optional[Callable] = None,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    ...
):
    return DatasetDict(
        {
            k: dataset.map(
                function: Optional[Callable] = None,
                batched: bool = False,
                batch_size: Optional[int] = 1000,
                ...
            )
            for k, dataset in self.items()
        }
    )
		

def map(
    self,
    function: Optional[Callable] = None,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    ...
) -> "Dataset":
    for rank, done, content in Dataset._map_single(**dataset_kwargs):
        ...
    
    return transformed_dataset
		
def _map_single(...):
    def apply_function_on_filtered_inputs(pa_inputs, indices, check_same_num_examples=False, offset=0):
        inputs = format_table(
            pa_inputs, 
            ...
        )

        processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
            
        elif isinstance(inputs, LazyDict):
            inputs_to_merge = {
                k: (v if k not in inputs.keys_to_format else pa_inputs[k]) for k, v in inputs.data.items()
            }
            
        if isinstance(inputs, Mapping) and isinstance(processed_inputs, Mapping):
            # The .map() transform *updates* the dataset:
            # the output dictionary contains both the the input data and the output data.
            # The output dictionary may contain Arrow values from `inputs_to_merge` so that we can re-write them efficiently.
            return {**inputs_to_merge, **processed_inputs}
            
            
    if not batched:
        for i, example in shard_iterable:
            example = apply_function_on_filtered_inputs(example, i, offset=offset)
    else:
        for i, batch in shard_iterable:
            batch = apply_function_on_filtered_inputs(
                batch, ...
            )
```