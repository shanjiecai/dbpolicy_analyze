from pyhanlp import *
import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH


def test_data_path():
    """
    获取测试数据路径，位于$root/data/test，根目录由配置文件指定。
    :return:
    """
    data_path = os.path.join(HANLP_DATA_PATH, 'test')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    return data_path


## 验证是否存在 MSR语料库，如果没有自动下载
def ensure_data(data_name, data_url):
    root_path = test_data_path()
    dest_path = os.path.join(root_path, data_name)
    if os.path.exists(dest_path):
        return dest_path

    if data_url.endswith('.zip'):
        dest_path += '.zip'
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with zipfile.ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path[:-len('.zip')]
    return dest_path


## 指定 PKU 语料库
PKU98 = ensure_data("pku98", "http://file.hankcs.com/corpus/pku98.zip")
PKU199801 = os.path.join(PKU98, '199801.txt')
PKU199801_TRAIN = os.path.join(PKU98, '199801-train.txt')
PKU199801_TEST = os.path.join(PKU98, '199801-test.txt')
POS_MODEL = os.path.join(PKU98, 'pos.bin')
NER_MODEL = os.path.join(PKU98, 'ner.bin')

## ===============================================
## 以下开始 CRF 命名实体识别

CRFNERecognizer = JClass('com.hankcs.hanlp.model.crf.CRFNERecognizer')
AbstractLexicalAnalyzer = JClass('com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer')
Utility = JClass('com.hankcs.hanlp.model.perceptron.utility.Utility')
PerceptronSegmenter = JClass('com.hankcs.hanlp.model.perceptron.PerceptronSegmenter')
PerceptronPOSTagger = JClass('com.hankcs.hanlp.model.perceptron.PerceptronPOSTagger')


def train(corpus, model):
    # 零参数的构造函数代表加载配置文件默认的模型，必须用null None 与之区分。
    recognizer = CRFNERecognizer(None)  # 空白
    recognizer.train(corpus, model)
    recognizer = CRFNERecognizer(model) # 需要load以喜爱
    return recognizer


# def test(recognizer):
#     analyzer = AbstractLexicalAnalyzer(PerceptronSegmenter(), PerceptronPOSTagger(), recognizer)
#     print(analyzer.analyze("华北电力公司董事长谭旭光和秘书胡花蕊来到美国纽约现代艺术博物馆参观"))
#     scores = Utility.evaluateNER(recognizer, PKU199801_TEST)
#     Utility.printNERScore(scores)


def test(recognizer,text):
    # 包装了感知机分词器和词性标注器的词法分析器
    analyzer = AbstractLexicalAnalyzer(PerceptronSegmenter(), PerceptronPOSTagger(), recognizer)
    result = analyzer.analyze(text)
    print(result)
    return result
    # scores = Utility.evaluateNER(recognizer, PKU199801_TEST)
    # Utility.printNERScore(scores)


# if __name__ == '__main__':
#     recognizer = train(PKU199801_TRAIN, NER_MODEL)
#     test(recognizer)
