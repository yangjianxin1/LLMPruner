# LLMPruner：大语言模型裁剪工具

## 项目简介
LLMPruner是一个大语言模型裁剪工具，通过对大语言模型的冗余词表进行裁剪，减少模型参数量，降低显存占用，提升训练速度，并且能够保留预训练中学习到的知识。

大语言模型(LLM, Large Language Model)犹如雨后春笋般，其虽然效果惊艳，但参数量巨大，让普通玩家望而却步。
如今的大语言模型大多为多语种大预言模型(Multilingual Large Language Model)，如LLaMA、mT5、Bloom等，其词表规模巨大，占据非常大部分的模型参数，如Bloom具有25万词表。
在训练模型时，词表权重将会消耗非常大的显存，降低训练速度，产生OOM的现象。

然而在许多下游任务中，我们往往只需要使用到一两种语言，例如在中文场景中，一般只会用到中英文。
我们可以对大语言模型的词表进行裁剪，只留下所需的部分，这样不仅能够充分保留模型的预训练知识，并且能够使用更少的显卡进行下游任务的finetune，提升训练效率。

## 裁剪模型分享
### Bloom
对Bloom进行词表裁剪，保留常用的中英文token，词表由250880将至46145，缩减为原来的18.39%。

| 裁剪模型                                                                        | 原模型                                        | 参数量比例  | 
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|--------|
| [YeungNLP/bloom-396m-zh](https://huggingface.co/YeungNLP/bloom-396m-zh) | [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m)       | 70.96% |  
| [YeungNLP/bloom-820m-zh](https://huggingface.co/YeungNLP/bloom-820m-zh) | [bigscience/bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)         | 77.13% |     
| [YeungNLP/bloom-1b4-zh](https://huggingface.co/YeungNLP/bloom-1b4-zh)   | [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)         | 81.14% |     
| [YeungNLP/bloom-2b6-zh](https://huggingface.co/YeungNLP/bloom-2b6-zh)   | [bigscience/bloom-3b](https://huggingface.co/bigscience/bloom-3b)           | 86.48% |     
| [YeungNLP/bloom-6b4-zh](https://huggingface.co/YeungNLP/bloom-6b4-zh)   | [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)         |  90.81% |         
| [YeungNLP/bloomz-396m-zh](https://huggingface.co/YeungNLP/bloomz-396m-zh) | [bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m)     | 70.96% |     
| [YeungNLP/bloomz-820m-zh](https://huggingface.co/YeungNLP/bloomz-820m-zh) | [bigscience/bloomz-1b1](https://huggingface.co/bigscience/bloomz-1b1)       | 77.13% |     
| [YeungNLP/bloomz-1b4-zh](https://huggingface.co/YeungNLP/bloomz-1b4-zh) | [bigscience/bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7)       | 81.14% |     
| [YeungNLP/bloomz-2b6-zh](https://huggingface.co/YeungNLP/bloomz-2b6-zh) | [bigscience/bloomz-3b](https://huggingface.co/bigscience/bloomz-3b)         | 86.48% |     
| [YeungNLP/bloomz-6b4-zh](https://huggingface.co/YeungNLP/bloomz-6b4-zh) | [bigscience/bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)       | 90.81% |
| [YeungNLP/bloomz-6b4-mt-zh](https://huggingface.co/YeungNLP/bloomz-6b4-mt-zh) | [bigscience/bloomz-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt) | 90.81% |   


## 使用介绍

对Bloom进行词表裁剪：
```python
from pruners.vocabulary_pruner import BloomVocabularyPruner

# 需要进行裁剪的模型路径
model_name_or_path = 'bigscience/bloom-560m'
# 自己制作的词表的路
new_tokenizer_name_or_path = 'YeungNLP/bloom-560m-zh'
save_path = 'path-to-save'
pruner = BloomVocabularyPruner()
# 裁剪
pruner.prune(model_name_or_path, new_tokenizer_name_or_path, save_path)
# 检查裁剪的模型与原模型是否一致
pruner.check(model_name_or_path, save_path, text='长风破浪会有时')
```

使用模型：
```python
from transformers import BloomTokenizerFast, BloomForCausalLM
tokenizer = BloomTokenizerFast.from_pretrained('YeungNLP/bloom-1b4-zh')
model = BloomForCausalLM.from_pretrained('YeungNLP/bloom-1b4-zh')
print(tokenizer.batch_decode(model.generate(tokenizer.encode('长风破浪会有时', return_tensors='pt'))))
```
## 关注我们

<img src="pics/gongzhonghao.jpeg" width="250"> 




