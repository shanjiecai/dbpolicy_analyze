from database import Mysql
import csv
import tqdm
from LAC import LAC
import jieba
from string import digits
WORD_MAX_LENGTH = 30



class GetSamples(object):
    '''
    获取样本 传入data_file为LAC分词后的文件 文件中第一列为文本的id
    获得两个样本文件 分别为普通的过滤结果文件和onehot编码后的文件
    has_data标记是否已经存在样本文件
    '''

    def __init__(self):
        self.read_stopwords()
        # self.get_original_samples(data_file)

    def length_bigger_than_30(self,text):
        if len(text)>=30:
            return True
        else:
            return False

    # 文本清洗
    def clean_text(self,text):
        # 需要自定义在医疗政策领域的词典
        wordlist = jieba.lcut(text)
        # 去除停用词和长度小于2的词语
        wordlist = [w for w in wordlist if w not in self.stopwords and len(w)>2]
        # 将中文数据组织成类似西方于洋那样，词语之间以空格间隔
        document =  "".join(wordlist)
        return document

    def read_stopwords(self):
        stopword_files = [
            'baidu_stopwords.txt', 'cn_stopwords.txt', 'hit_stopwords.txt',
            'my_stopwords.txt', 'scu_stopwords.txt'
        ]
        self.stopwords = list()
        for stopwords in stopword_files:
            f = open('./机器学习分类/stopwords/' + stopwords, encoding='utf-8')
            self.stopwords.append(f.readlines())

    def get_original_samples(self, data_file):
        self.mysql = Mysql(user='', password='', db='', port=3306, host='121.36.33.190')
        result = self.mysql.select(table="api_details,api_links",target="api_details.main_text,api_details.id",
                                   condition="api_details.links_id = api_links.id and api_links.zupei_type='其他' limit 10000")
        labels = {
            'nr': '人名', 'PER': '人名', 'nz': '其他专名',
            'ns': '地名', 'LOC': '地名', 'nt': '机构名',
            'ORG': '机构名', 'nw': '作品名'
        }
        lac = LAC(mode='lac')
        res_file = open(data_file, 'w', encoding='utf-8')
        csv_writer = csv.writer(res_file)
        for i, policy in enumerate(result):
            text = policy['main_text'].replace("[^\u4e00-\u9fa5]", "")  # 去除非汉字
            if not text:
                continue
            words_tags = lac.run(text)
            ner_str = ''
            for w, t in zip(words_tags[0], words_tags[1]):
                if t in labels.keys():
                    ner_str += w
            if len(ner_str)/len(text) > 0.3:
                print(policy['id'])
                continue
            remove_digits = str.maketrans('', '', digits)  # 去除数字
            text = text.translate(remove_digits)
            text = self.clean_text(text)  # 分词，去除停用词
            if not self.length_bigger_than_30(text):  # 文本长度需要大于30
                continue
            else:
                print(policy['id'])
                csv_writer.writerow([policy['id'],policy['main_text']])


# get = GetSamples('其他.csv')

def combine():
    csv.field_size_limit(500 * 1024 * 1024)
    csv_writer = csv.writer(open("all.csv","w",encoding="utf-8"))

    f1 = csv.reader(open("其他.csv","r",encoding="utf-8"))
    f2 = csv.reader(open("动态要闻.csv","r",encoding="utf-8"))
    f3 = csv.reader(open("政策文件.csv","r",encoding="utf-8"))
    f4 = csv.reader(open("政策解读.csv","r",encoding="utf-8"))
    f5 = csv.reader(open("规划计划.csv","r",encoding="utf-8"))
    for line in f1:
        line.append('其他')
        csv_writer.writerow(line)
    for line in f2:
        line.append('动态要闻')
        csv_writer.writerow(line)
    for line in f3:
        line.append('政策文件')
        csv_writer.writerow(line)
    for line in f4:
        line.append('政策解读')
        csv_writer.writerow(line)
    for line in f5:
        line.append('规划计划')
        csv_writer.writerow(line)


getsamples = GetSamples()
def LAC_process():
    labels = {
        'nr': '人名', 'PER': '人名', 'nz': '其他专名',
        'ns': '地名', 'LOC': '地名', 'nt': '机构名',
        'ORG': '机构名', 'nw': '作品名'
    }
    csv.field_size_limit(500 * 1024 * 1024)
    f = open("all.csv","r",encoding="utf-8")
    f2 = open("lac_process.csv", "w", encoding="utf-8")
    policys = csv.reader(f)
    result = csv.writer(f2)
    lac = LAC(mode='lac')
    for i,policy in enumerate(policys):
        print(i)
        line = [policy[0],policy[2]]
        text = policy[1].replace("[^\u4e00-\u9fa5]", "")  # 去除非汉字
        remove_digits = str.maketrans('', '', digits)  # 去除数字
        text = text.translate(remove_digits)
        text = getsamples.clean_text(text)  # 分词，去除停用词
        words_tags = lac.run(text)
        for w, t in zip(words_tags[0], words_tags[1]):
            if t in labels.keys():
                line.append(w)
        if not line:
            continue
        result.writerow(line)



LAC_process()




