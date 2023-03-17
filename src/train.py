import os.path
from common.text_generator import TextGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-td', '--train-data', help='请指定样本集文件', required=True)
parser.add_argument('-mp', '--model-path', help='请指定模型路径', required=True)
parser.add_argument('-mn', '--model-name', help='请指定模型名称', required=True)
args = parser.parse_args()

train_data = args.train_data
model_path = args.model_path
model_name = args.model_name
model_path = os.path.join(model_path, model_name)

text_gen = TextGenerator(model_path=model_path)

text_gen.train(train_data=train_data)
text_gen.save()
