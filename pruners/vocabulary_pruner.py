import os.path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class VocabularyPruner(object):

    def check(self, old_model_name_or_path, new_model_name_or_path, text):
        # 检查模型裁剪后，生成结果是否一致
        max_length = 20

        # 使用老模型对文本编码
        old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path)
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path)
        old_input_ids = old_tokenizer(text, return_tensors='pt').input_ids
        old_output = old_model.generate(old_input_ids, max_length=max_length)
        old_output_text = old_tokenizer.batch_decode(old_output)
        print('old_output:{}'.format(old_output_text))

        # 使用新模型对文本编码
        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path)
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_path)
        new_input_ids = new_tokenizer(text, return_tensors='pt').input_ids
        new_output = new_model.generate(new_input_ids, max_length=max_length)
        new_output_text = new_tokenizer.batch_decode(new_output)
        print('new_output:{}'.format(new_output_text))

        if old_output_text == new_output_text:
            print('output is same, succeed to prune.')
        else:
            print('output is not same, fail to prune.')

    def update_ebeddings(self, model, new2old_token_id, new_embeds, new_lm_head):
        raise NotImplemented

    def prune(self, model_name_or_path, new_tokenizer_name_or_path, save_path, new_name_or_path=None):
        # 创建输出目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 加载新词表。如果是中文，就是中文的词表
        new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_name_or_path)
        # 加载原词表。一般为多语言模型的词表
        old_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # 检查新词表是否为原词表的子集
        old_vocab = old_tokenizer.vocab
        new_vocab = new_tokenizer.vocab
        for token in tqdm(new_vocab.keys()):
            if token not in old_vocab:
                raise Exception('{} not exist'.format(token))
        print('new_tokenizer is subset of old_tokenizer')

        # 获得新词表中每个token_id到原词表的token_id的映射
        new2old_token_id = {}
        for token, token_id in tqdm(new_vocab.items()):
            old_token_id = old_vocab[token]
            new2old_token_id[token_id] = old_token_id

        # 加载多语言模型
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype='auto')
        # 计算原模型的参数量
        old_params = sum(p.numel() for p in model.parameters())
        print("Total params of original model: %.2fM" % (old_params / 1e6))

        # 对于新词表中的每个token，取出其对应的权重，复制到新模型中
        vocab_size = len(new_tokenizer)
        hidden_size = model.config.hidden_size

        new_embeds = torch.nn.Embedding(vocab_size, hidden_size, dtype=model.dtype)
        new_lm_head = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=False, dtype=model.dtype)
        # 更新词表权重
        self.update_ebeddings(model, new2old_token_id, new_embeds, new_lm_head)

        model.config.__dict__['vocab_size'] = vocab_size
        if new_name_or_path is not None:
            model.config.__dict__['_name_or_path'] = new_name_or_path

        # 计算新模型的参数量
        new_params = sum(p.numel() for p in model.parameters())
        print("Total params of new model : %.2fM" % (new_params / 1e6))

        print('词表缩小为原来的:{}%'.format(round(len(new_tokenizer) / len(old_tokenizer), 4)*100))
        print('模型参数量缩小为原来的:{}%'.format(round(new_params / old_params, 4)*100))
        model.save_pretrained(save_path)
        new_tokenizer.save_pretrained(save_path)


class BloomVocabularyPruner(VocabularyPruner):

    def update_ebeddings(self, model, new2old_token_id, new_embeds, new_lm_head):
        for token_id, old_token_id in tqdm(new2old_token_id.items()):
            new_embeds.weight.data[token_id] = model.transformer.word_embeddings.weight.data[old_token_id]
            new_lm_head.weight.data[token_id] = model.lm_head.weight.data[old_token_id]
        model.transformer.word_embeddings.weight = new_embeds.weight
        model.lm_head.weight = new_lm_head.weight

