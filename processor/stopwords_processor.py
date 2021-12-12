"""
合并不同的停用词文本
"""
import glob

path = '../stopwords/'
files = glob.glob(path + '*.txt')
words = []
for f in files:
    print(f)
    with open(f, 'r') as fp:
        words.extend(fp.read().strip().split('\n'))
print('总共有词汇：{}'.format(len(words)))
words_res = list(set(words))
print('去除重复后的有词汇：{}'.format(len(words_res)))
with open(path + 'chinese_stopwords.txt', 'w') as fp:
    fp.write("\n".join(words_res))
