"""
    该文件用于生成词表
"""
from collections import Counter
from itertools import chain

import jieba
import time
import re


def print_run_time(func):
    """时间装饰器"""

    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print('[%s] run time is %.4f' % (func.__name__, time.time() - local_time))
        return res
    return wrapper

def clean_text(text):
    text = re.sub('[ ]+', ' ', text)
    text = re.sub('\n', '', text)
    return text

@print_run_time
def get_words(path):
    with open(path, 'r') as fp:
        lines = fp.read().strip().split('\n')
        print("总共有样本：{}".format(len(lines)))
        words = []
        for line in lines:
            text = line.strip().split('\t')[1]
            text = clean_text(text)
            words += jieba.lcut(text, cut_all=False)
        return words


if __name__ == '__main__':
    words_res = []
    train_words = get_words('../data/THUCNews/cnews.train.txt')
    val_words = get_words('../data/THUCNews/cnews.val.txt')
    test_words = get_words('../data/THUCNews/cnews.test.txt')
    words = train_words + val_words + test_words
    word_counts = Counter(words)
    use_stopwords = True
    if use_stopwords:
        with open('../stopwords/chinese_stopwords.txt', 'r') as fp:
            stopwords = fp.read().strip().split('\n')
        for k,v in dict(word_counts).items():
            if k not in stopwords:
                words_res.append(k)
        print("总共有词：{}".format(len(words_res)))
        with open('../data/THUCNews/cnews.word2.vocab.txt', 'w') as fp:
            fp.write("\n".join(words_res))
    else:
        words_res = word_counts.keys()
        print("总共有词：{}".format(len(words_res)))
        with open('../data/THUCNews/cnews.word.vocab.txt', 'w') as fp:
            fp.write("\n".join(words_res))

