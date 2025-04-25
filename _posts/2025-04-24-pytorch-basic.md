---
title: Pytorch - 기초
description: Pytorch - 기초
author: hchvhp1543
date: 2025-04-25 11:33:00 +0900
categories: [AI, NLP, DL]
tags: [AI, NLP, DL, Pytorch]
pin: false
math: true
mermaid: true

---

## Tensor
- 중요 필드

```python
class Tensor(torch._C.TensorBase):

class TensorBase(metaclass=_TensorMeta):
    requires_grad: _bool
    shape: Size
    data: Tensor
    device: _device
    dtype: _dtype
    ndim: _int
    grad: Optional[Tensor]
```

```python
class Size(tuple[_int, ...]):

class device:
    type: str  
    index: _int  
# torch.device('cpu')
# torch.device('cuda:0')
# torch.device('cuda', 0)
    
class dtype(object):
```

## Tensor 타입
- dtype
  - tensor.dtype 으로 확인 가능 (attribute)
  - torch.float32, torch.float16, torch.int8, ...

```python
float32: dtype = ...
float: dtype = ...
float64: dtype = ...
double: dtype = ...
float16: dtype = ...
bfloat16: dtype = ...
```

- type
  - tensor.type() 으로 확인 가능 (method)
  - torch.LongTensor, torch.FloatTensor, ...

```python
class DoubleTensor(Tensor): ...
class FloatTensor(Tensor): ...
class BFloat16Tensor(Tensor): ...
class LongTensor(Tensor): ...
```

## Tensor 생성 방법
- torch.Tensor(data) : `__init__`메소드. deprecated. torch.tensor() 사용 권장
  - torch.LongTensor, torch.FloatTensor 으로도 가능
- torch.tensor(data, dtype, device) -> Tensor : 함수
- torch.as_tensor(data, dtype, device) -> Tensor
- torch.from_numpy(ndarray) -> Tensor : 


## nn.Module
- 딥러닝에서 모델(model) 또는 레이어(layer)의 기본 타입
- 클래스의 속성에 접근하거나 값을 할당시 `__getattr__`, `__setattr__` 통해 module, buffer, parameter 접근/할당 가능 

```python
class Module:
    training: bool
    _parameters: dict[str, Optional[Parameter]]
    _buffers: dict[str, Optional[Tensor]]
    _modules: dict[str, Optional["Module"]]
    forward: Callable[..., Any] = _forward_unimplemented

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_buffers", {})
        super().__setattr__("_modules", {})
        
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        params = self.__dict__.get("_parameters")
        self.register_parameter(name, value)

        modules = self.__dict__.get("_modules")
        modules[name] = value

        buffers = self.__dict__.get("_buffers")
        self.register_buffer(name, value, persistent)

        super().__setattr__(name, value)
    
    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        self._buffers[name] = tensor
      
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self._parameters[name] = param
      
    def register_module(self, name: str, module: Optional["Module"]) -> None:
        self.add_module(name, module)
      
    def add_module(self, name: str, module: Optional["Module"]) -> None:
        self._modules[name] = module

    def get_submodule(self, target: str) -> "Module":
    def get_parameter(self, target: str) -> "Parameter":
    def get_buffer(self, target: str) -> "Tensor":
```

##  nn.Sequential, nn.ModuleList, nn.ModuleDict
- nn.Sequential 와 nn.ModuleList 차이
  - nn.Sequential : forward 함수를 따로 정의할 필요 없이, 자동으로 순서대로 실행
  - nn.ModuleList : 자동으로 forward가 정의되지는 않음. 어떻게 실행할지는 사용자가 직접 forward 메서드에서 정해야 함
- nn.ModuleDict는 Lora 내부 구현시 사용

```python
class Sequential(Module):
    _modules: dict[str, Module]
    def forward(self, input):
        for module in self:
            input = module(input)
        return input
    
class ModuleList(Module):
    _modules: dict[str, Module]
  
class ModuleDict(Module):
    _modules: dict[str, Module]
```

```python
class LoraLayer(BaseTunerLayer):
    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
    
    self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
    self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)  
    
    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight
```

## Parameter
- torch.Tensor 와 유사
- nn.Module에서 주요하게 담고 있는 값의 타입
- parameters(), named_parameters() 함수 자주 사용
- nn.ParameterList, nn.ParameterDict 도 존재

```python
class _ParameterMeta(torch._C._TensorMeta):

class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    def __new__(cls, data=None, requires_grad=True):

class Module:
    _parameters: dict[str, Optional[Parameter]]

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _name, param in self.named_parameters(recurse=recurse):
            yield param
        
    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        # self._named_members -> self.named_modules -> self._modules.items()
```

## state_dict
- Module 내의 모든 Paramter와 Buffer 정보를 담은 딕셔너리
- 모델 저장/로드 할시 활용

```python
def _save_to_state_dict(self, destination, prefix, keep_vars):
    for name, param in self._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in self._buffers.items():
        if buf is not None and name not in self._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()
        
def state_dict(self, *args, destination=None, prefix="", keep_vars=False) -> dict[str, Any]:
    destination = OrderedDict()
    self._save_to_state_dict(destination, prefix, keep_vars)

    for name, module in self._modules.items():
        if module is not None:
            module.state_dict(
                destination=destination,
                prefix=prefix + name + ".",
                keep_vars=keep_vars,
            )
        
    return destination
```

```python
def _load_from_state_dict(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):  
    persistent_buffers = {
        k: v
        for k, v in self._buffers.items()
        if k not in self._non_persistent_buffers_set
    }
    local_name_params = itertools.chain(
        self._parameters.items(), persistent_buffers.items()
    )
    local_state = {k: v for k, v in local_name_params if v is not None}
    
    for name, param in local_state.items():
        setattr(self, name, input_param)
        param.copy_(input_param)
        
        error_msgs.append(...)
        missing_keys.append(key)
        unexpected_keys.append(key)
    
def load_state_dict(
    self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
):
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    error_msgs: list[str] = []
    
    state_dict = OrderedDict(state_dict)
    
    def load(module, local_state_dict, prefix=""):
        module._load_from_state_dict(...)
        for name, child in module._modules.items():
            load(child, child_state_dict, child_prefix)
        
    load(self, state_dict)
    
    return _IncompatibleKeys(missing_keys, unexpected_keys)

```
