from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig


# example from https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completions-only
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


prompt_template = '''
### Instruction:
Write a restaurant description according to provided information: "{description}"

### Response:
{response}
'''

def format_prompts(example):
    output_texts = []
    for description, response in zip(example['meaning_representation'], example['human_reference']):
        text = prompt_template.format(description=description, response=response)
        output_texts.append(text)
    return output_texts


def finetune():
    model_name = 'facebook/opt-350m'
    dataset = load_dataset('e2e_nlg')

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )

    response_template = '### Response:\n'
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./opt-350m-e2e',
        num_train_epochs=5,
        evaluation_strategy='steps',
        save_strategy='epoch',
        fp16=True,
    )

    trainer = SFTTrainer(
        model,
        peft_config=peft_config,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        formatting_func=format_prompts,
        data_collator=collator,
    )
    trainer.train()
    # trainer.state


if __name__ == '__main__':
    finetune()
