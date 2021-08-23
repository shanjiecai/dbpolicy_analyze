# -*- coding: utf-8 -*-

import os
from pyltp import Segmentor, Postagger

# 分词
cws_model_path = os.path.join(os.path.dirname(__file__), 'data/cws.model')  # 分词模型路径，模型名称为`cws.model`
lexicon_path = os.path.join(os.path.dirname(__file__), 'data/lexicon.txt')  # 参数lexicon是自定义词典的文件路径

segmentor = Segmentor(cws_model_path)
# segmentor.load_with_lexicon(cws_model_path, lexicon_path)  新版pyltp废弃load_with_lexicon

sent = '据韩联社12月28日反映，美国防部发言人杰夫·莫莱尔27日表示，美国防部长盖茨将于2011年1月14日访问韩国。'
words = segmentor.segment(sent)  # 分词

# 词性标注
pos_model_path = os.path.join(os.path.dirname(__file__), 'data/pos.model')  # 词性标注模型路径，模型名称为`pos.model`

postagger = Postagger(pos_model_path)  # 初始化实例
# postagger.load(pos_model_path)  # 加载模型
postags = postagger.postag(words)  # 词性标注


ner_model_path = os.path.join(os.path.dirname(__file__), 'data/ner.model')   # 命名实体识别模型路径，模型名称为`pos.model`

from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer(ner_model_path) # 初始化实例
# recognizer.load(ner_model_path)  # 加载模型
# netags = recognizer.recognize(words, postags)  # 命名实体识别


# 提取识别结果中的人名，地名，组织机构名

persons, places, orgs = set(), set(), set()


netags = list(recognizer.recognize(words, postags))  # 命名实体识别
print(netags)
# print(netags)
i = 0
for tag, word in zip(netags, words):
    j = i
    # 人名
    if 'Nh' in tag:
        if str(tag).startswith('S'):
            persons.add(word)
        elif str(tag).startswith('B'):
            union_person = word
            while netags[j] != 'E-Nh':
                j += 1
                if j < len(words):
                    union_person += words[j]
            persons.add(union_person)
    # 地名
    if 'Ns' in tag:
        if str(tag).startswith('S'):
            places.add(word)
        elif str(tag).startswith('B'):
            union_place = word
            while netags[j] != 'E-Ns':
                j += 1
                if j < len(words):
                    union_place += words[j]
            places.add(union_place)
    # 机构名
    if 'Ni' in tag:
        if str(tag).startswith('S'):
            orgs.add(word)
        elif str(tag).startswith('B'):
            union_org = word
            while netags[j] != 'E-Ni':
                j += 1
                if j < len(words):
                    union_org += words[j]
            orgs.add(union_org)

    i += 1

print('人名：', '，'.join(persons))
print('地名：', '，'.join(places))
print('组织机构：', '，'.join(orgs))


# 释放模型
segmentor.release()
postagger.release()
recognizer.release()
