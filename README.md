<h1 style='text-align: left '>Storyteller</h1>
<h3 style='text-align: left '>Write your own Lord Of The Rings story!</h3>



# Description

In this project, I have designed an AI assistant that completes your stories in the LOTR style. During the development of the app, we have:
* Extracted the text from the official book,
* Prepared the dataset,
* Trained BLOOM-3B using Low-Rank-Adapters,
* Deployed the model on Inference Endpoints from Hugging Face,
* Built the app using Streamlit,
* Deployed it into Streamlit cloud.

*Notes: regarding the cost of deploying a model this large, the app is not available for testing*

## :gear: Model fine-tuning [[code]](https://github.com/Sajjad-Amjad/storyteller/tree/main/llm)

This LLM is fine-tuned on [Bloom-3B](https://huggingface.co/bigscience/bloom-3b) with texts extracted from the book "[The Lord of the Rings](https://gosafir.com/mag/wp-content/uploads/2019/12/Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010.pdf)".


The Hugging Face model card: [sajjadamjad/storyteller](https://huggingface.co/sajjadamjad/storyteller)

Finetuning Notebook: [colab](https://colab.research.google.com/drive/1tY00knVfb_TUI0HkOdhNkgp2KirWPOIG?usp=sharing)

## :rocket: Model deployment and app [[code]](https://github.com/Sajjad-Amjad/storyteller/tree/main/src)

The model is deployed on Inference Endpoints from Hugging Face, and the applicaiton is built and deployed on Streamlit.


# Load the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

# Import the model
config = PeftConfig.from_pretrained("sajjadamjad/storyteller")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# Load the Lora model
model = PeftModel.from_pretrained(model, "sajjadamjad/storyteller")
```

# Run the model

```python
prompt = "The hobbits were so suprised seeing their friend"

inputs = tokenizer(prompt, return_tensors="pt")
tokens = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=1,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True
)

# The hobbits were so suprised seeing their friend again that they did not 
# speak. Aragorn looked at them, and then he turned to the others.</s>
```

# Training parameters

```python
# Dataset
context_length = 2048

# Training
model_name = 'bigscience/bloom-3b'
lora_r = 16 # attention heads
lora_alpha = 32 # alpha scaling
lora_dropout = 0.05
lora_bias = "none"
lora_task_type = "CAUSAL_LM"

## Trainer config
per_device_train_batch_size = 1 
gradient_accumulation_steps = 1
warmup_steps = 100 
num_train_epochs=3
weight_decay=0.1
learning_rate = 2e-4 
fp16 = True
evaluation_strategy = "no"
```
