## LLaVA

- https://huggingface.co/xtuner/llava-llama-3-8b-v1_1
- https://llava.hliu.cc/
```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")
```

## Idefics
- https://huggingface.co/HuggingFaceM4/idefics-9b-instruct
```python
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)

```

## Moondream
- https://huggingface.co/vikhyatk/moondream2

```python
pip install transformers einops

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-05-08"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
```

## Reference

### Trelis Research
- This playlist does a great job of comparing vision models starting with LLaVA 1.5, ending with LLaVA 1.6 with Lllama3 as the LLM.
  - https://youtube.com/playlist?list=PLWG1mVtuzdxcYamFhHJjQg_TmgId6tETX&si=WKMDDJoueutUSIxM

#### Slides
- https://docs.google.com/presentation/d/1C4MmT3-JNRpVMgoA14pQC7Ie6vBo0gj70CR_jqTIpos/edit#slide=id.gcb9a0b074_1_0
- https://docs.google.com/presentation/d/1LTF8PLe2kwLaddeqwgCRWnTabt7b5EEviFq29x3zlyw/edit#slide=id.gcb9a0b074_1_0