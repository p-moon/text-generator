from common.text_generator import TextGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='请指定输入内容', required=True)
args = parser.parse_args()

prompt = args.input

# 定义模型路径
model_path = './model/HongLouMen-model'

text_gen = TextGenerator(model_path=model_path)
text_gen.load_model_from_model_path()
text_gen.generate(prompt=prompt)
