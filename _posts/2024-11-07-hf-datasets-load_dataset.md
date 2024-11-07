---
layout: post
title: HF Datasets - load_dataset
tags: [Katex, Mermaid, Markdown]
categories: Demo
---

# HF datasets - load_dataset

```python
from datasets import load_dataset
dataset = load_dataset("lhoestq/demo1", download_mode="force_redownload", trust_remote_code=True)
```

- load_dataset 메소드의 결과는 크게 2가지이다
    - Dataset
    - DatasetDict
- load_dataset 메소드 내부의 핵심은 builder 인스턴스이며, 크게 3가지 과정으로 진행된다
    - builder 인스턴스 생성 : load_dataset_builder
    - 다운로드 및 준비 : download_and_prepare
    - dataset 생성 : as_dataset

---

- builder 인스턴스
    - parent class : DatasetBuilder
    - 주요 : child class : ArrowBasedBuilder(csv, json, sql 등), GeneratorBasedBuilder(custom)
    - 주요 메소드
        - def _info(self)
        - def _split_generators(self, …)
        - def _generate_tables(self, …) / def _generate_examples(self, …)
- ArrowBasedBuilder 예시

```python
class Csv(datasets.ArrowBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)
        
    def _split_generators(self, dl_manager):
		splits = []
		for split_name, files in data_files.items():
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
		return splits
		    
    def _generate_tables(self, files):
		csv_file_reader = pd.read_csv(file, iterator=True, dtype=dtype, **self.config.pd_read_csv_kwargs)
		    
		yield (file_idx, batch_idx), self._cast_table(pa_table)
```

- GeneratorBasedBuilder 예시

```python
class CustomSquad(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(...)
            
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]
		    
    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            
            yield id_, {
		        "title": title,
		        "context": context,
		        "question": question,
		        "id": id_,
		        "answers": {
		            "answer_start": answer_starts,
		            "text": answers,
		        },
		    }
```

- 흐름

```python
builder_instance.download_and_prepare(...)

self._download_and_prepare(...)

# Generating data for all splits
split_generators = self._split_generators(dl_manager, **split_generators_kwargs)
 
# Build splits
for split_generator in split_generators:
    self._prepare_split(split_generator, **prepare_split_kwargs)
		
for job_id, done, content in self._prepare_split_single(
    gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args
):

generator = self._generate_examples(**gen_kwargs)
generator = self._generate_tables(**gen_kwargs)

writer = writer_class(...)

for key, record in generator:
    writer.write(example, key)
		
for _, table in generator:
    writer.write_table(table)
	
self.write_examples_on_file()
self.write_batch(batch_examples=batch_examples)
self.write_table(pa_table, writer_batch_size)		

# Save info
self._save_info()

logger.info(
    f"Dataset {self.dataset_name} downloaded and prepared to {self._output_dir}. "
    f"Subsequent calls will reuse this data."
)

${HOME}/.cache/huggingface/datasets/lhoestq___custom_squad/plain_text/1.0.0/397916d1ae99584877e0fb4f5b8b6f01e66fcbbeff4d178afb30c933a8d0d93a
dataset_info.json
custom_squad-train.arrow
custom_squad-validation.arrow
```

---

- dataset 생성

```python
ds = builder_instance.as_dataset(split=split, verification_mode=verification_mode, in_memory=keep_in_memory)

# By default, return all splits
if split is None:
    split = {s: s for s in self.info.splits}
    
# Create a dataset for each of the given splits
datasets = map_nested(
    partial(
        self._build_single_dataset,
        ...
    ),
    ...
)    

# Build base dataset
ds = self._as_dataset(
    split=split,
    in_memory=in_memory,
)

def _as_dataset(self, ...) -> Dataset:
    dataset_kwargs = ArrowReader(cache_dir, self.info).read(...)
    return Dataset(fingerprint=fingerprint, **dataset_kwargs)

def read(...):
    return self.read_files(files=files, original_instructions=instructions, in_memory=in_memory)
  
if isinstance(datasets, dict):
    datasets = DatasetDict(datasets)
```