
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
from datasets import load_dataset

model_name = "bert-base-uncased"
batch_size = 16
datasets = load_dataset("json" , data_files={"train":"data.json", "validation":"data.json"})

# guided by: https://huggingface.co/docs/transformers/notebooks
    
tx = AutoTokenizer.from_pretrained(model_name)
max_length = 384
stride = 128

pad_on_right = tx.padding_side == "right"

def prepare_train_features(data_row):
    data_row["question"] = [q.lstrip() for q in data_row["question"]]

    tokenized_data_rows = tx(
        data_row["question" if pad_on_right else "context"],
        data_row["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_data_rows.pop("overflow_to_sample_mapping")
   
    offset_mapping = tokenized_data_rows.pop("offset_mapping")

    tokenized_data_rows["start_positions"] = []
    tokenized_data_rows["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        
        input_ids = tokenized_data_rows["input_ids"][i]
        cls_index = input_ids.index(tx.cls_token_id)

        sequence_ids = tokenized_data_rows.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = data_row["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_data_rows["start_positions"].append(cls_index)
            tokenized_data_rows["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_data_rows["start_positions"].append(cls_index)
                tokenized_data_rows["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_data_rows["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_data_rows["end_positions"].append(token_end_index + 1)

    return tokenized_data_rows

tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)





model = AutoModelForQuestionAnswering.from_pretrained(model_name)

model_nickname = model_name.split("/")[-1]
print('model_name',model_name)
args = TrainingArguments(
    f"{model_nickname}-finetuned-MITRE",
    evaluation_strategy = "epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.05,
)


data_collator = default_data_collator

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tx,
)
trainer.train()
trainer.save_model("devset-mitre-trained")
