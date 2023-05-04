---
license: apache-2.0
---

# CodeGen2 (CodeGen2-1B)

## Model description

[CodeGen2](https://github.com/salesforce/CodeGen2) is a family of autoregressive language models for **program synthesis**, introduced in the paper:

[CodeGen2: Lessons for Training LLMs on Programming and Natural Languages](https://arxiv.org/abs/2305.02309) by Erik Nijkamp\*, Hiroaki Hayashi\*, Caiming Xiong, Silvio Savarese, Yingbo Zhou.

Unlike the original CodeGen model family (i.e., CodeGen1), CodeGen2 is capable of infilling, and supports more programming languages.

Four model sizes are released: `1B`, `3.7B`, `7B`, `16B`.

## How to use

This model can be easily loaded using the `AutoModelForCausalLM` functionality.

### Causal sampling

For regular causal sampling, simply generate completions given the context:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")

text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

### Infill sampling

For **infill** sampling, we introduce three new special token types:

* `<mask_N>`: N-th span to be masked. In practice, use `<mask_1>` to where you want to sample infill.
* `<sep>`: Seperator token between the suffix and the infilled sample. See below.
* `<eom>`: "End-Of-Mask" token that model will output at the end of infilling. You may use this token to truncate the output.

For example, if we want to generate infill for the following cursor position of a function:
```python
def hello_world():
    |
    return name
```
we construct an input to the model by

1. Inserting `<mask_1>` token in place of cursor position
2. Append `<sep>` token to indicate the boundary
3. Insert another `<mask_1>` to indicate which mask we want to infill.

The final snippet looks as follows:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B")


def format(prefix, suffix):
  return prefix + "<mask_1>" + suffix + "<|endoftext|>" + "<sep>" + "<mask_1>"


prefix = "def hello_world():\n    "
suffix = "    return name"
text = format(prefix, suffix)
input_ids = tokenizer(text, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=False)[len(text):])
```

You might want to truncate the model output with `<eom>`.

## Training data

This checkpoint is trained on the stricter permissive subset of [the deduplicated version of the Stack dataset (v1.1)](https://huggingface.co/datasets/bigcode/the-stack-dedup). Supported languages (and frameworks) are as follows:
`c`, `c++`, `c-sharp`, `dart`, `go`, `java`, `javascript`, `kotlin`, `lua`, `php`, `python`, `ruby`, `rust`, `scala`, `shell`, `sql`, `swift`, `typescript`, `vue`.

## Training procedure

CodeGen2 was trained using cross-entropy loss to maximize the likelihood of sequential inputs.
The input sequences are formatted in two ways: (1) causal language modeling and (2) file-level span corruption.
Please refer to the paper for more details.

## Evaluation results

We evaluate our models on HumanEval and HumanEval-Infill. Please refer to the [paper](https://arxiv.org/abs/2305.02309) for more details.

## Intended use and limitations

As an autoregressive language model, CodeGen2 is capable of extracting features from given natural language and programming language texts, and calculating the likelihood of them.
However, the model is intended for and best at **program synthesis**, that is, generating executable code given English prompts, where the prompts should be in the form of a comment string. The model can complete partially-generated code as well.


## BibTeX entry and citation info

```bibtex
@article{Nijkamp2023codegen2,
  title={CodeGen2: Lessons for Training LLMs on Programming and Natural Languages},
  author={Nijkamp, Erik and Hayashi, Hiroaki and Xiong, Caiming and Savarese, Silvio and Zhou, Yingbo},
  journal={arXiv preprint},
  year={2023}
}
```
