from transformers import pipeline, set_seed

import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer

import torch
import evaluate

from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

model="google/pegasus-cnn_dailymail"
tokenizer=AutoTokenizer.from_pretrained(model)
model_pegasus=AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

multi_news_dataset = load_dataset('multi_news',trust_remote_code=True)

def generate_tokens(data):
    input_encodings = tokenizer(data['document'] , max_length = 1024, truncation = True )
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(data['summary'], max_length = 64, truncation = True )
    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    } 

dataset_pt = multi_news_dataset.map(generate_tokens, batched = True)

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

trainer_args = TrainingArguments(
    num_train_epochs=5,
    warmup_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    evaluation_strategy='steps',
    eval_steps=500
)

trainer = Trainer(model=model_pegasus, args=trainer_args,tokenizer=tokenizer, data_collator=seq2seq_data_collator,train_dataset=dataset_pt["train"],eval_dataset=dataset_pt["validation"])
trainer.train()
batch_size=8
def generate_batch(list_of_elements, batch_size):
 
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]

def calculate_metric(dataset, metric, model, tokenizer,batch_size=16, device=device,column_text="document",column_summary="summary"):

    article_batches = list(generate_batch(dataset[column_text], batch_size))
    target_batches = list(generate_batch(dataset[column_summary], batch_size))


    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):

        inputs = tokenizer(article_batch, max_length=1024,  truncation=True,padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device),attention_mask=inputs["attention_mask"].to(device),length_penalty=0.8, num_beams=8, max_length=128)
        
        
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,clean_up_tokenization_spaces=True)
               for s in summaries]

        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    score = metric.compute()
    return score

rouge_names=["rouge1","rouge2","rougeL","rougeLsum"]
rouge_metric=evaluate.load('rouge')

score = calculate_metric(
    multi_news_dataset['test'], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'document', column_summary= 'summary'
)
print(score)
#rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

#print(pd.DataFrame(rouge_dict, index = [f'pegasus'] ))
