from common.text_generator import TextGenerator
import argparse
import sys

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', help='请指定输入内容', required=True)
# args = parser.parse_args()

# prompt = args.input

# 定义模型路径
model_path = './model/HongLouMen-model'

text_gen = TextGenerator(model_path=model_path)
text_gen.load_model_from_model_path()
encoding = sys.stdin.encoding
if encoding != 'UTF-8':
    sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='UTF-8', buffering=True)

if __name__ == '__main__':
    while True:
        user_input = input("请输入:")
        print("生成内容：" + text_gen.generate(prompt=user_input, max_length=500))
