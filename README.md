# eda_for_chinese_text_classification
基于EDA进行中文文本分类<br>
<a href="https://arxiv.org/abs/1901.11196">EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks</a>

# 依赖
```
jieba
synonyms
```
特别说明下synonyms的安装，使用```pip install synonyms```时，会下载一些所需的文件，由于网络问题下载很慢，这里准备好了，百度网盘：
链接: https://pan.baidu.com/s/1sKsBy81eAVemoLusHKQUAQ 提取码: 3wfk <br>
将其下载后放在python环境下的site-packages/synonyms/下。
# 说明
## 1、合并stopwords。
由于有4个不同的停止词文件，首先要将其合并，合并代码在processor/stopwords_processor.py，处理之后文件在stopwords下chinese_stopwords.txt。
## 2、测试synonyms
每次```import synonyms```的时候都会加载词向量模型，这一般会很慢，为了不用每次使用的时候都加载，采用预加载的方式。具体代码在script/下，具体文件说明如下：<br>
--synonyms_service.py：使用flask生成调用的接口。<br>
--start_synonyms.sh：启动服务。<br>
--stop_synonyms.sh：停止服务。<br>
--restart_synonyms.sh：重启服务。<br>
--synonyms_service.log：日志。<br>
--test_synonyms_service.py：测试调用接口。<br>
然后在augmentations下的aug.py中AugForCls类有一个初始化的属性：self.synonyms_service = True。如果不想用服务的这种方式，直接将其置为False即可。
## 3、测试数据增强
（1）在augmentations下：```aug.py```是数据增强文件，包含同义词替换、随机插入、随机交换、随机删除，里面__main__下有测试代码，直接运行```python aug.py```测试即可。<br>
（2）在augmentations下：```aug_for_file.py```是输入一个文件，并进行增强后输出一个文件。特别说明：
- 默认输入的txt中每一行是```标签\t文本```
- max_seq_len用于截断较长的句子。
- num_aug用于得到增强后的句子数，比如num_aug=4，那么每条数据返回的数目就是4+1（原始文本）=5。
- replace_ratio、insert_ratio、swap_ratio、delete_prob如果分别设置为0.0，那么表示不使用该项进行数据增强。ratio表示操作的词数占句子长度的比例，比如句子长度是256，replace_ratio=0.3，那么随机替换的词数是256×0.3。prob表示每个词被删除的概率。
- 为了加速增强的速度，使用了多进程的方式，具体可看代码。
（3）生成数据增强文件：```sh run.sh```。
```
nohup python -u aug_for_file.py \
--input "../data/THUCNews/cnews.train.txt" \
--num_aug 4 \
--replace_ratio 0.3 \
--insert_ratio 0.3 \
--swap_ratio 0.3 \
--delete_prob 0.3 > eda.log 2>&1 &
```
默认会在cnews.train.txt的同级目录下生成eda_cnews.train.txt。
## 4、使用Bilstm进行测试
最后使用bilstm进行测试，配置文件在config/config.py下。里面重点注意的是：
- use_word：表示是否基于词
- use_stopword：基于词时是否过滤掉停用词
- do_train：是否训练
- do_eval：是否验证
- do_test：是否测试
- do_predict：是否预测
在```main.py```里面是主运行函数，根据不同的数据可能需要修改data_name以及model_name，下面是一些结果：<br>

|  模型（biltm）   | accuracy  | precision | recall| macro_f1|
|  ------------  | ------------  |------------  |------------  |------------  |
| 基于字  | 0.8992 |0.9033|    0.8992 |   0.8943|
| 基于词（不过滤停止词）  | 0.8372 |0.8440  |  0.8372 |   0.8137 |
|基于词（过滤停止词）|0.8536|0.8579  |  0.8536  |  0.8408|
|基于词+数据增强（不过滤停止词）|0.9191|0.9225  |  0.9191 |   0.9168|
|基于词+数据增强（过滤停止词）|0.9184|0.9219  |  0.9184  |  0.9157|
|基于字+数据增强| 0.9357|0.9369  |  0.9357  |  0.9348|

# 补充
processor/processor_word.py用于生成词汇表，可酌情修改：<br>
cnews.word2.vocab.txt：使用停止词后的得到的vocab<br>
cnews.word.vocab.txt：不使用停止词后得到的vocab<br>
在这里面都会有很多数字，可以在创建词汇表的时候先过滤掉这些信息<br>

# 题外话：
数据增强的方式还有很多种，比如：<br>
1、随机交换句子；<br>
2、随机交换一段区间内的词；<br>
3、替换词的时候不仅仅使用同义词（类似于纠错）；<br>
4、依存句法分析进行句式转换；<br>
5、对同标签的文本之间进行句子抽取得到新样本；<br>
6、转译：中译英-英译中；<br>
7、训练生成模型，根据生成模型生成样本；<br>
等等等。<br>
针对于英文的：https://github.com/makcedward/nlpaug<br>
可参考其编写中文的增强代码。

# 参考
> EDA的实现参考：https://github.com/zhanlaoban/EDA_NLP_for_Chinese/
