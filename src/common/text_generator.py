from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments


class TextGenerator:

    def __init__(self, model_path: str) -> None:
        # 加载 GPT-2 分词器
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self._model_path = model_path
        self._model = GPT2LMHeadModel.from_pretrained('gpt2')

    def load_model_from_model_path(self):
        self._model = GPT2LMHeadModel.from_pretrained(self._model_path)

    def train(self, train_data, block_size=128):
        train_dataset = TextDataset(tokenizer=self._tokenizer, file_path=train_data, block_size=block_size)
        training_args = TrainingArguments(
            output_dir=self._model_path,  # 模型保存路径
            overwrite_output_dir=True,  # 是否覆盖之前的模型
            num_train_epochs=3,  # 训练轮数
            save_strategy='epoch',  # 将该值设置为no可以不生成checkpoint
            # evaluation_strategy='epoch',
            per_device_train_batch_size=16,  # 训练集 batch_size
            per_device_eval_batch_size=16,  # 验证集 batch_size
            eval_steps=100,  # 验证间隔步数
            save_steps=500,  # 保存间隔步数
            warmup_steps=500,  # warmup 步数
            learning_rate=2e-5,  # 学习率
            logging_dir='./logs',  # 日志保存路径
            logging_steps=100,  # 日志间隔步数
            # load_best_model_at_end=True,      # 加载最佳模型
            metric_for_best_model='eval_loss',  # 用于比较最佳模型的指标
            greater_is_better=False  # 是否越大越好
        )
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False),
        )
        # 训练模型
        trainer.train()

    def save(self):
        self._model.save_pretrained(self._model_path)

    def generate(self, prompt: str, max_length: int = 50, do_sample: bool = True, skip_special_tokens: bool = True):
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt')
        output = self._model.generate(input_ids, max_length=max_length, do_sample=do_sample)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
        print(generated_text)
