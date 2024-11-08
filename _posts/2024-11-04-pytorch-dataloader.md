---
layout: post
title: Pytorch - DataLoader
tags: [Katex, Mermaid, Markdown]
categories: Demo
---

# DataLoader

생성시 주요 파라미터 옵션

- batch_size
- shuffle
- drop_last

---

주요 구성요소

- dataset : Dataset
- sampler : RandomSampler, SequentialSampler
- batch_sampler : BatchSampler
- collator(collate_fn)
- iterator : _BaseDataLoaderIter (single/multi process)
    - (dataset_) fetcher : _MapDatasetFetcher (_IterableDatasetFetcher)

---

sampler : List[int]

- 주요 필드 : data_source → num_samples
- RandomSampler
    - yield from torch.randperm(n, generator=generator).tolist() # 난수 순열
    - EX) [2, 5, 7, 0, 3, 8, 9, 6, 1, 4]
- SequentialSampler
    - return iter(range(len(self.data_source)))

---

batch_sampler : Iterator[List[int]]

- 주요 필드 : sampler, batch_size, drop_last
- batch = [next(sampler_iter) for _ in range(self.batch_size)]
yield batch
- drop_last = False → [[0,1,2],[3,4,5],[6,7,8],[9]]
drop_last = True → [[0,1,2],[3,4,5],[6,7,8]]

---

(dataset_) fetcher

- 주요 필드 : dataset, collate_fn, drop_last
- possibly_batched_index : List[int]
- dataset 의  __getitem__, __getitems__ 메소드 활용
- data = [self.dataset[idx] for idx in possibly_batched_index]
- return self.collate_fn(data)

---

collate_fn : Callable[List[T], Any]

- batch : list of dataset’s getitem # 중요
- 함수 형태 : def collate(batch, …)
클래스 형태 : def __call__(self, features: List[Dict[str, Any]])

---

iterator

- data = _next_data()
    - index = self._next_index() ← sampler + bach_sampler
    - data = self._dataset_fetcher.fetch(index)

---

코드 흐름

```python
for idx, batch in enumerate(dataloader):
		...
		
def __iter__(self) -> '_BaseDataLoaderIter':
    return self._get_iterator()

def _get_iterator(self) -> '_BaseDataLoaderIter':
    return _SingleProcessDataLoaderIter(self)

def __next__(self) -> Any:
    data = self._next_data()
    return data
		
def _next_data(self):
    index = self._next_index()  # may raise StopIteration
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    
    return data
  
def _next_index(self):
    return next(self._sampler_iter)  # may raise StopIteration
    
def __iter__(self) -> Iterator[List[int]]:
    sampler_iter = iter(self.sampler)
		
    batch = [next(sampler_iter) for _ in range(self.batch_size)]
    yield batch
    
def __iter__(self) -> Iterator[int]:
    yield from torch.randperm(n, generator=generator).tolist()
		
def fetch(self, possibly_batched_index):
    data = [self.dataset[idx] for idx in possibly_batched_index]
		
    return self.collate_fn(data)
		
def collate(batch,...)
    return ...
		
class DataCollatorWithPadding:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return batch
```