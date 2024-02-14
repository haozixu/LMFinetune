import evaluate
import datasets

from transformers import AutoModelForCausalLM, AutoTokenizer

from collections import defaultdict
from tqdm import tqdm

prompt_template = '''
### Instruction:
Write a restaurant description according to provided information: "{description}"

### Response:
'''

if __name__ == '__main__':
    dataset = datasets.load_dataset('e2e_nlg')
    bleu_metric = evaluate.load('bleu')

    ckpt_path = 'opt-350m-e2e/checkpoint-26290'
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # merge inputs
    merged_samples = defaultdict(list)
    for sample in dataset['test']:
        description = sample['meaning_representation']
        ref_response = sample['human_reference']

        merged_samples[description].append(ref_response)
    print(f'length of merged dataset: {len(merged_samples)}')

    predictions, references = [], []
    for description, ref_responses in tqdm(merged_samples.items()):
        prompt = prompt_template.format(description=description)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
        input_len = input_ids.shape[1]

        outputs = model.generate(input_ids, max_new_tokens=64,
                                 do_sample=False, repetition_penalty=0.9, no_repeat_ngram_size=5)
        prediction = tokenizer.decode(outputs[0, input_len:])
        predictions.append(prediction)
        references.append(ref_responses)

    print(bleu_metric.compute(predictions=predictions, references=references))
        
    from IPython import embed
    embed()
