import jieba
import random
import requests
from random import shuffle



class AugForCls:
    def __init__(self):
        # 停用词列表，已合并四种停用词列表
        with open('../stopwords/chinese_stopwords.txt', 'r') as fp:
            self.stop_words = fp.read().strip().split('\n')
        self.synonyms_service = True

    def get_synonyms(self, word):
        """
        这里使用synonyms库，获取与word最接近的词，也可使用其它方式
        :return:
        """
        if self.synonyms_service:
            url = "http://192.168.0.101:1314/synonyms"
            res = requests.post(url, data=word.encode('utf-8'))
            data = eval(res.text)
            data = data['data']
        else:
            import synonyms
            data = synonyms.nearby(word)[0]
        return data

    def synonym_replacement(self, words, n):
        """ 同义词替换
        替换一个语句中的n个单词为其同义词
        """
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def random_insertion(self, words, n):
        """随机插入
        随机在语句中插入n个词
        """
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        """插入主函数"""
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]  # 随机从单词列表中选择一个词
            synonyms = self.get_synonyms(random_word)  # 找到该词对应的相似词
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms)  # 随机选择一个相似词
        random_idx = random.randint(0, len(new_words) - 1)  # 随机选择一个位置
        new_words.insert(random_idx, random_synonym)  # 在该位置上插入随机词

    def random_swap(self, words, n):
        """随机交换"""
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        """交换主函数"""
        random_idx_1 = random.randint(0, len(new_words) - 1)  # 随机选择一个词的位置
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    def random_deletion(self, words, p):
        """这里以一定的概率随机删除"""
        if len(words) == 1:
            return words
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]
        return new_words


class EDA:
    def __init__(self,
                 augForCls,
                 replace_ratio=0.3,
                 insert_ratio=0.3,
                 swap_ratio=0.3,
                 delete_prob=0.3,
                 num_aug=9,
                 max_seq_len=None):
        """如果num_aug<4，那么每种数据增强都只会使用一次
            num_new_per_technique = int(self.num_aug / 4) + 1
        :param augForCls:
        :param replace_ratio:
        :param insert_ratio:
        :param swap_ratio:
        :param delete_prob:
        :param num_aug:
        """
        self.augForCls = augForCls
        self.replace_ratio = replace_ratio
        self.insert_ratio = insert_ratio
        self.swap_ratio = swap_ratio
        self.delete_prob = delete_prob
        self.num_aug = num_aug  # 用来控制每一种策略使用的次数
        self.max_seq_len = max_seq_len

    def augment(self, text):
        # text = "我们就像蒲公英，我也祈祷着能和你飞去同一片土地。"
        if self.max_seq_len:
            text = text[:self.max_seq_len]
        words = jieba.lcut(text, cut_all=False)
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(self.num_aug / 4) + 1
        replace_n = max(1, int(self.replace_ratio * num_words))
        insert_n = max(1, int(self.insert_ratio * num_words))
        swap_n = max(1, int(self.swap_ratio * num_words))

        # print(words, "\n")

        if self.replace_ratio != 0.0:
            # 同义词替换sr
            for _ in range(num_new_per_technique):
                a_words = self.augForCls.synonym_replacement(words, replace_n)
                augmented_sentences.append(''.join(a_words))

        if self.insert_ratio != 0.0:
            # 随机插入ri
            for _ in range(num_new_per_technique):
                a_words = self.augForCls.random_insertion(words, insert_n)
                augmented_sentences.append(''.join(a_words))

        if self.swap_ratio != 0.0:
            # 随机交换rs
            for _ in range(num_new_per_technique):
                a_words = self.augForCls.random_swap(words, swap_n)
                augmented_sentences.append(''.join(a_words))

        if self.delete_prob != 0.0:
            # 随机删除rd
            for _ in range(num_new_per_technique):
                a_words = self.augForCls.random_deletion(words, self.delete_prob)
                augmented_sentences.append(''.join(a_words))

        # print(augmented_sentences)
        shuffle(augmented_sentences)

        augmented_sentences = augmented_sentences[:self.num_aug]

        # 加上原语句
        augmented_sentences.append(text)

        return augmented_sentences


if __name__ == '__main__':
    # text = "我们就像蒲公英，我也祈祷着能和你飞去同一片土地。"
    text = """ 马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。7月31日下午6点，
国奥队的日常训练再度受到大雨的干扰，无奈之下队员们只慢跑了25分钟就草草收场。31日上午10点，国奥队在奥体中心外场训练的时候，天就是阴
沉沉的，气象预报显示当天下午沈阳就有大雨，但幸好队伍上午的训练并没有受到任何干扰。下午6点，当球队抵达训练场时，大雨已经下了几个小>时，而且丝毫没有停下来的意思。抱着试一试的态度，球队开始了当天下午的例行训练，25分钟过去了，天气没有任何转好的迹象，为了保护球员们
，国奥队决定中止当天的训练，全队立即返回酒店。在雨中训练对足球队来说并不是什么稀罕事，但在奥运会即将开始之前，全队变得“娇贵”了。在
沈阳最后一周的训练，国奥队首先要保证现有的球员不再出现意外的伤病情况以免影响正式比赛，因此这一阶段控制训练受伤、控制感冒等疾病的出
现被队伍放在了相当重要的位置。而抵达沈阳之后，中后卫冯萧霆就一直没有训练，冯萧霆是7月27日在长春患上了感冒，因此也没有参加29日跟塞>尔维亚的热身赛。队伍介绍说，冯萧霆并没有出现发烧症状，但为了安全起见，这两天还是让他静养休息，等感冒彻底好了之后再恢复训练。由于有
了冯萧霆这个例子，因此国奥队对雨中训练就显得特别谨慎，主要是担心球员们受凉而引发感冒，造成非战斗减员。而女足队员马晓旭在热身赛中受
伤导致无缘奥运的前科，也让在沈阳的国奥队现在格外警惕，“训练中不断嘱咐队员们要注意动作，我们可不能再出这样的事情了。”一位工作人员表
示。从长春到沈阳，雨水一路伴随着国奥队，“也邪了，我们走到哪儿雨就下到哪儿，在长春几次训练都被大雨给搅和了，没想到来沈阳又碰到这种>事情。”一位国奥球员也对雨水的“青睐”有些不解。"""
    words = jieba.lcut(text, cut_all=False)
    print("原始文本：", text)
    augForCls = AugForCls()
    print("===============================================")
    replace_ratio = 0.3
    replace_n = int(len(words) * replace_ratio)
    replace_res = augForCls.synonym_replacement(words, replace_n)
    print("随机替换后文本：", "".join(replace_res))
    print("===============================================")
    insert_ratio = 0.3
    insert_n = int(len(words) * insert_ratio)
    insert_res = augForCls.random_insertion(words, insert_n)
    print("随机插入后文本：", "".join(insert_res))
    print("===============================================")
    swap_ratio = 0.3
    swap_n = int(len(words) * swap_ratio)
    swap_res = augForCls.random_swap(words, swap_n)
    print("随机交换后文本：", "".join(swap_res))
    print("===============================================")
    delete_prob = 0.4
    delete_res = augForCls.random_deletion(words, delete_prob)
    print("随机删除后文本：", "".join(delete_res))

    print("===============================================")
    print("===============================================")
    eda = EDA(augForCls)
    sentences = eda.augment(text)
    for i in sentences:
        print(i)
        print("===============================================")
