import os
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

    def train(self, train_data, epoch=5, block_size=128):
        if os.path.exists(self._model_path):
            self.load_model_from_model_path()

        train_dataset = TextDataset(tokenizer=self._tokenizer, file_path=train_data, block_size=block_size)
        training_args = TrainingArguments(
            output_dir=self._model_path,  # 模型保存路径
            overwrite_output_dir=True,  # 是否覆盖之前的模型
            num_train_epochs=epoch,  # 训练轮数
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
        encoded_input = self._tokenizer(prompt, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        output = self._model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,  # 明确设置attention mask
                                      pad_token_id=self._tokenizer.eos_token_id,  # 明确设置pad token id
                                      max_length=max_length,
                                      temperature=1.0,
                                      repetition_penalty=1.0,
                                      do_sample=do_sample,
                                      top_k=50,
                                      top_p=0.95,
                                      num_return_sequences=1)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
        return generated_text
