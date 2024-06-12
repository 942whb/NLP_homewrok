from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 使用GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 创建数据集
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="jin_yong_corpus.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 训练模型
training_args = TrainingArguments(
    output_dir="./gpt2_jin_yong",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

def generate_text_transformer(seed_text, next_words, model, tokenizer):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=next_words + len(input_ids[0]), num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_text_transformer("张无忌", 50, model, tokenizer))
