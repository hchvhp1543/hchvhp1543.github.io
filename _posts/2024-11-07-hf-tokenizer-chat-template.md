---
layout: post
title: HF Tokenizer - Chat Template
tags: [Katex, Mermaid, Markdown]
categories: Demo
---

# chat template

- 환경
    - mac (m1)
    - pycharm
    - pytorch 2.5.1
    - transformers 4.46.2
    - tokenizers 0.20.3
    - datasets 2.19.2


참고 페이지
- https://huggingface.co/docs/transformers/main/chat_templating#introduction
- https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct


개요
- 현재 LLM은 채팅 형태로 많이 활용함
- 기본 베이스 모델이 같더라도, 다른 chat 포맷으로 학습할수 있음.
    - Mistral-7B-v0.1 → Zephyr
    - Mistral-7B-v0.1 → Mistral-Instruct
- chat_template 이 없다면 각가의 모델에 맞는 포맷을 작성해야함. 작성 중 실수로 인해 성능에 영향을 줄 수 있음.
- chat_template는 tokenizer의 일부. tokenizer_config.json 에 포함되어 있음
- tokenizer.get_chat_template 메소드를 통해 각 모델에 맞는 chat_template을 알수 있음
- 채팅 목록을 chat_template에 적용하려면 tokenizer.apply_chat_template 메소드를 활용
- apply_chat_template 메소드 주요 파라미터
    - tokenizer = True/False
    - add_generation_prompt=True/False
    - return_dict=True/False
    - return_tensors="pt”

---

코드 흐름

```python
def apply_chat_template(
    self,
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    add_generation_prompt: bool = False,
    tokenize: bool = True,
    padding: bool = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_dict: bool = False,
    ...
):
    chat_template = self.get_chat_template(chat_template, tools)
		
    # Compilation function uses a cache to avoid recompiling the same template
    compiled_template = _compile_jinja_template(chat_template)
    
    if isinstance(conversation, (list, tuple)) and (
        isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
    ):
        conversations = conversation
        is_batched = True
    else:
        conversations = [conversation]
        is_batched = False
    
    rendered = []
    for chat in conversations:
        ...
        rendered_chat = compiled_template.render(
            messages=chat,
            ...
        )
        
    if not is_batched:
        rendered = rendered[0]

    if tokenize:
        out = self(
            rendered,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        if return_dict:
            ...
    else:
        return rendered
```

---

chat_template 예제1

- model : ibm-granite/granite-3.0-1b-a400m-instruct

{% raw %}
```python
# tokenizer_config.json
{
    ...
    "chat_template": "{%- if tools %}\n    {{- '<|start_of_role|>available_tools<|end_of_role|>\n' }}\n    {%- for tool in tools %}\n    {{- tool | tojson(indent=4) }}\n    {%- if not loop.last %}\n        {{- '\n\n' }}\n    {%- endif %}\n    {%- endfor %}\n    {{- '<|end_of_text|>\n' }}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n    {{- '<|start_of_role|>system<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'user' %}\n    {{- '<|start_of_role|>user<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'assistant' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>'  + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'assistant_tool_call' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'tool_response' %}\n    {{- '<|start_of_role|>tool_response<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- endif %}\n    {%- if loop.last and add_generation_prompt %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>' }}\n    {%- endif %}\n{%- endfor %}",
    ...
}

# Stores a Jinja template that formats chat histories into tokenizable strings
self.chat_template = kwargs.pop("chat_template", None)
```
{% endraw %}

{% raw %}
```python
chat_template = tokenizer.get_chat_template()

"""
{%- if tools %}
    {{- '<|start_of_role|>available_tools<|end_of_role|>
' }}
    {%- for tool in tools %}
    {{- tool | tojson(indent=4) }}
    {%- if not loop.last %}
        {{- '
' }}
    {%- endif %}
    {%- endfor %}
    {{- '<|end_of_text|>
' }}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
    {{- '<|start_of_role|>system<|end_of_role|>' + message['content'] + '<|end_of_text|>
' }}
    {%- elif message['role'] == 'user' %}
    {{- '<|start_of_role|>user<|end_of_role|>' + message['content'] + '<|end_of_text|>
' }}
    {%- elif message['role'] == 'assistant' %}
    {{- '<|start_of_role|>assistant<|end_of_role|>'  + message['content'] + '<|end_of_text|>
' }}
    {%- elif message['role'] == 'assistant_tool_call' %}
    {{- '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' + message['content'] + '<|end_of_text|>
' }}
    {%- elif message['role'] == 'tool_response' %}
    {{- '<|start_of_role|>tool_response<|end_of_role|>' + message['content'] + '<|end_of_text|>
' }}
    {%- endif %}
    {%- if loop.last and add_generation_prompt %}
    {{- '<|start_of_role|>assistant<|end_of_role|>' }}
    {%- endif %}
{%- endfor %}
"""

conversation = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
"""
'<|start_of_role|>system<|end_of_role|>You are a pirate chatbot who always responds in pirate speak!<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Who are you?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>'
"""

# tokenize the text
input_tokens = tokenizer(rendered, return_tensors="pt").to(device)
"""
{
    'input_ids': tensor([[49152,  2946, 49153,  4282,   884,   312,   298,   476,   332, 11210,
          2195,  6560,  5182, 18641,  3210,   328,   298,   476,   332, 24498,
            19,     0,   203, 49152,   496, 49153, 27868,   884,   844,    49,
             0,   203, 49152, 17594, 49153]]), 
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
"""

# generate output tokens
model_output = model.generate(**input_tokens, max_new_tokens=100)
"""
tensor([[49152,  2946, 49153,  4282,   884,   312,   298,   476,   332, 11210,
          2195,  6560,  5182, 18641,  3210,   328,   298,   476,   332, 24498,
            19,     0,   203, 49152,   496, 49153, 27868,   884,   844,    49,
             0,   203, 49152, 17594, 49153,    51,    31,   107,   503,  2734,
            30,   439,  3464,   312,   298,   476,   332, 11210,  2195,    30,
           312,    31,  3073,   941,    19,   439,  3464,  2442,   372,  3049,
           844,   623,  1370, 10017,    30,  2258,  2124,   312,    31,  4543,
           298,   476,   332,    19,   203,   203,  8197,  1182,  1370,   636,
            30,   345,   332,   107,    49,     0]])
"""

# decode output tokens into text
decoded_output = tokenizer.batch_decode(model_output)
"""
[
    "<|start_of_role|>system<|end_of_role|>You are a pirate chatbot who always responds in pirate speak!<|end_of_text|>\n
    <|start_of_role|>user<|end_of_role|>Who are you?<|end_of_text|>\n
    <|start_of_role|>assistant<|end_of_role|>A-yessher, I'm a pirate chatbot, a-wayne! I'm here to help you with your questions, just like a-real pirate!\n\nWhat's your name, matey?<|end_of_text|>"
]
"""

####
input_tokens = tokenizer.apply_chat_template(
		conversation, 
		tokenize=True, 
		add_generation_prompt=True, 
		return_dict=True, 
		return_tensors="pt"
)
model_output = model.generate(**input_tokens, max_new_tokens=100)
decoded_output = tokenizer.batch_decode(model_output)
```
{% endraw %}

chat_template 예제2

- model : HuggingFaceH4/zephyr-7b-beta

```python

messages = [
    { "role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate", },
    { "role": "user", "content": "How many helicopters can a human eat in one sitting?" },
 ]

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
"""
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
"""

outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
"""
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
"""
```