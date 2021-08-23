from database import Mysql
import nltk
import jieba.posseg as pseg
import os
import pyltp
from LAC import LAC
import thulac
from tqdm import tqdm
import csv
import jieba
from string import digits
from pyltp import Segmentor, Postagger, NamedEntityRecognizer
# from perceptron_ner import *
from crf_ner import *
import re
# 可以引入jionlp
import jionlp
# jionlp.clean_text(text)


def get_chinese(text):
    pre = re.compile(u'[\u4e00-\u9fa5]')
    res = re.findall(pre, text)
    res1 = ''.join(res)
    return res1


class AttributeAnnotation(object):
    def __init__(self):
        f = open("all.csv", "r", encoding="utf-8")
        policys = csv.reader(f)
        self.details = policys
        self.read_stopwords()
        # 加载pyltp模型
        cws_model_path = os.path.join(os.path.dirname(__file__), 'data/cws.model')  # 分词模型路径，模型名称为`cws.model`
        ner_model_path = os.path.join(os.path.dirname(__file__), 'data/ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        pos_model_path = os.path.join(os.path.dirname(__file__), 'data/pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        self.postagger = Postagger(pos_model_path)  # 初始化实例
        self.segmentor = Segmentor(cws_model_path)
        self.recognizer = NamedEntityRecognizer(ner_model_path)  # 初始化实例

        # 加载hanlp模型
        self.ha_recognizer = train(PKU199801_TRAIN, NER_MODEL)

    def length_bigger_than_30(self,text):
        if len(text) >= 20:
            return True
        else:
            return False

    # 文本清洗
    def clean_text(self,text):
        # 需要自定义在医疗政策领域的词典
        wordlist = jieba.cut(text)
        # 去除停用词和长度小于2的词语
        wordlist = [w for w in wordlist if w not in self.stopwords and len(w) > 1]
        # 将中文数据组织成类似西方于洋那样，词语之间以空格间隔
        document =  "".join(wordlist)
        return document

    def read_stopwords(self):
        # stopword_files = [
        #     'baidu_stopwords.txt', 'cn_stopwords.txt', 'hit_stopwords.txt',
        #     'my_stopwords.txt', 'scu_stopwords.txt', 'more_stopwords.txt'
        # ]
        stopword_files = ['stopword1.txt','stopword2.txt']
        self.stopwords = []
        for stopwords in stopword_files:
            f = open('./机器学习分类/stopwords/' + stopwords, encoding='utf-8')
            for l in f.readlines():
                self.stopwords.append(l.strip())

    def pyltp_annotation(self, text):
        try:
            labels = {
                'j': '缩略词', 'nh': '人名', 'ni': '组织名',
                'nl': '地点', 'ns': '地理位置', 'ws': '外语词汇'
            }
            line = []
            text = text.replace("[^\u4e00-\u9fa5]", "")  # 去除非汉字
            remove_digits = str.maketrans('', '', digits)  # 去除数字
            text = text.translate(remove_digits)
            text = self.clean_text(text)  # 分词，去除停用词
            # text = jionlp.clean_text(text)

            words = self.segmentor.segment(text)
            tags = self.postagger.postag(words)
            # for w, t in zip(words, tags):
            #     if t in labels.keys():
            #         line.append(w)
            # return line
            netags = list(self.recognizer.recognize(words, tags))  # 命名实体识别
            i = 0
            for tag, word in zip(netags, words):
                j = i
                # 人名
                if 'Nh' in tag:
                    if str(tag).startswith('S'):
                        line.append(word)
                    elif str(tag).startswith('B'):
                        union_person = word
                        while netags[j] != 'E-Nh':
                            j += 1
                            if j < len(words):
                                union_person += words[j]
                        line.append(union_person+'nh')
                # 地名
                if 'Ns' in tag:
                    if str(tag).startswith('S'):
                        line.append(word)
                    elif str(tag).startswith('B'):
                        union_place = word
                        while netags[j] != 'E-Ns':
                            j += 1
                            if j < len(words):
                                union_place += words[j]
                        line.append(union_place+'ns')
                # 机构名
                if 'Ni' in tag:
                    if str(tag).startswith('S'):
                        line.append(word)
                    elif str(tag).startswith('B'):
                        union_org = word
                        while netags[j] != 'E-Ni':
                            # print(netags[j])
                            j += 1
                            if j < len(words):
                                union_org += words[j]
                        line.append(union_org+'ni')
                i += 1
            return line
        except Exception as e:
            print(e)
            return []

    def lac_annotation(self, text):
        labels = {
            'nr': '人名', 'PER': '人名', 'nz': '其他专名',
            'ns': '地名', 'LOC': '地名', 'nt': '机构名',
            'ORG': '机构名'
        }
        lac = LAC(mode='lac')
        line = []
        text = text.replace("[^\u4e00-\u9fa5]", "")  # 去除非汉字
        remove_digits = str.maketrans('', '', digits)  # 去除数字
        text = text.translate(remove_digits)
        text = self.clean_text(text)  # 分词，去除停用词
        # text = jionlp.clean_text(text)

        words_tags = lac.run(text)
        for w, t in zip(words_tags[0], words_tags[1]):
            if t in labels.keys():
                line.append(w+t)
        return line

    def hanlp_annotation(self, text):
        labels = {
            'nr': '人名', 'ns': '地名', 'nt': '机构名',
            'nh': '医药疾病等健康相关名词','nhd': '疾病',
            'nhm': '药品','ni': '机构相关（不是独立机构名）',
            'ntc': '公司名', 'nit': '教育相关机构',
            'nto':'政府机构','ntu':'大学',

        }
        line = []
        text = text.replace("[^\u4e00-\u9fa5]", "")  # 去除非汉字
        remove_digits = str.maketrans('', '', digits)  # 去除数字
        text = text.translate(remove_digits)
        text = self.clean_text(text)  # 分词，去除停用词
        # text = jionlp.clean_text(text)

        words_tags = test(self.ha_recognizer, text)

        for words_tag in list(words_tags):
            w, t = str(words_tag).rsplit('/', 1)  # 只分割最右边的/
            w = get_chinese(w)  # 嵌套结果
            if t in labels.keys():
                w = get_chinese(w)
                line.append(w+t)
        return line

    def bagging(self):
        csv.field_size_limit(500 * 1024 * 1024)
        f = open("bagging_result.csv", "w", encoding="utf-8",newline='')
        csv_writer = csv.writer(f)
        for detail in tqdm(self.details):
            # print(detail[1])
            lac_result = self.lac_annotation(detail[1])
            print(lac_result)
            pyltp_result = self.pyltp_annotation(detail[1])
            print(pyltp_result)
            hanlp_result = self.hanlp_annotation(detail[1])
            print(hanlp_result)
            csv_writer.writerow([detail[1],str(lac_result),str(pyltp_result),str(hanlp_result)])


annotation = AttributeAnnotation()
annotation.bagging()