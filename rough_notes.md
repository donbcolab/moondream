# Rough Notes - interesting and valuable patterns evolving in the Open Vision landscape
- almost a plug and play approach to model fusion for various use cases
- additional vision encoders beyon CLIP (SigLIP, etc.)
- refinement of adapter layer approaches between vision encoders and LLM layers (moving beyond original single linear layer in Llava 1.5 to MLP, etc.)
- increased variation in LLM models, expanding into lighter weight models with fewer parameters than can (almost) run purely on CPU
- patterns for fine tuning models are being established

## LLaVA

- Model Cards:
  - https://huggingface.co/liuhaotian/llava-v1.6-34b
  - https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b
  - https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b
  - https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b

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

## Idefics2
- model cards
  - https://huggingface.co/HuggingFaceM4/idefics2-8b
  - https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty
  - https://huggingface.co/HuggingFaceM4/idefics2-8b-base
  - https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
  - https://huggingface.co/datasets/HuggingFaceM4/OBELICS
- [Idefics Notebook](https://github.com/donbcolab/moondream/blob/main/notebooks/finetuning_idefics.ipynb) - at this time focuses on defining and displaying the architecture of the Idefics model using the HF Transofrmers model

```python
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)

```

## mantis
- model card
  - https://huggingface.co/TIGER-Lab/Mantis-8B-clip-llama3
  - https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3

## xtuner
- model card
  - https://huggingface.co/xtuner/llava-phi-3-mini-hf
  - https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-hf
  - https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers

## Moondream
- model card
  - https://huggingface.co/vikhyatk/moondream2
- [Moondream Notebook](https://github.com/donbcolab/moondream/blob/main/notebooks/finetuning_moondream.ipynb) - largely based on original notebook, fine tuned on CAPTCHA type dataset

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

### XTuner models
- have a variety of interesting LLaVA variants with various LLMs (Llama3, Phi3, etc.)

## arXiv papers

- https://arxiv.org/search/?query=idefics2&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- reason for picking these two papers is the open minded discussion around model fusion, and leveraging similar data models for fine tuning

```
arXiv:2405.02246  [pdf, other]  cs.CV cs.AI
What matters when building vision-language models?

Authors: Hugo Laurençon, Léo Tronchon, Matthieu Cord, Victor Sanh

Abstract: …issue, we conduct extensive experiments around pre-trained models, architecture choice, data, and training methods. Our consolidation of findings includes the development of Idefics2, an efficient foundational VLM of 8 billion parameters. Idefics2 achieves state-of-the-art performance within its size category across va… ▽ More
Submitted 3 May, 2024; originally announced May 2024.

arXiv:2405.01483  [pdf, other]  cs.CV cs.AI cs.CL
MANTIS: Interleaved Multi-Image Instruction Tuning

Authors: Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, Wenhu Chen

Abstract: …resources (i.e. 36 hours on 16xA100-40G), Mantis-8B can achieve state-of-the-art performance on all the multi-image benchmarks and beats the existing best multi-image LMM Idefics2-8B by an average of 9 absolute points. We observe that Mantis performs equivalently well on the held-in and held-out evaluation benchmarks. We further evaluate Mantis on single-ima… ▽ More
Submitted 2 May, 2024; originally announced May 2024.

Comments: 9 pages, 3 figures
```
