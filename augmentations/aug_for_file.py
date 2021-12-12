from aug import AugForCls, EDA
from multiprocessing import Pool

import argparse

"""
    如果将replace_ratio、insert_ratio、swap_ratio、delete_prob中的某个设置为0.0，
    则该项数据增强不会使用
"""
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, default='../data/THUCNews/cnews.train.txt', type=str, help="原始数据的输入文件目录")
ap.add_argument("--output", required=False, type=str, help="增强数据后的输出文件目录")
ap.add_argument("--max_seq_len", required=False, default=256, type=str, help="句子的最大长度")
ap.add_argument("--num_aug", required=False, default=9, type=int, help="每条原始语句增强的语句数")
ap.add_argument("--replace_ratio", required=False, default=0.3, type=float, help="替换占比")
ap.add_argument("--insert_ratio", required=False, default=0.3, type=float, help="插入占比")
ap.add_argument("--swap_ratio", required=False, default=0.3, type=float, help="交换占比")
ap.add_argument("--delete_prob", required=False, default=0.3, type=float, help="每个词被删除的概率")
args = ap.parse_args()


def callback(aug_sentences):
    label = aug_sentences[0]
    with open(args.output, 'a') as fp:
        for sen in aug_sentences[1:]:
            fp.write(label + '\t' + sen + '\n')


def gen_task(line, eda):
    label = line[0]
    sentence = line[1]
    aug_sentences = eda.augment(sentence)
    return [label] + aug_sentences


if __name__ == "__main__":
    augForCls = AugForCls()
    # 输出文件
    output = None
    switch = False
    sep = '\t'
    if not args.output:
        from os.path import dirname, basename, join

        args.output = join(dirname(args.input), 'eda_' + basename(args.input))
    print(args)
    with open(args.input, 'r') as fp:
        lines = fp.read().strip().split('\n')
        if switch:
            lines = [(i.split(sep)[1], i.split(sep)[0]) for i in lines]
        else:
            lines = [(i.split(sep)[0], i.split(sep)[1]) for i in lines]
    eda = EDA(
        augForCls=augForCls,
        replace_ratio=args.replace_ratio,
        insert_ratio=args.insert_ratio,
        swap_ratio=args.swap_ratio,
        delete_prob=args.delete_prob,
        num_aug=args.num_aug,
        max_seq_len=args.max_seq_len,
    )

    pool = Pool(10)
    for i, line in enumerate(lines):
        pool.apply_async(gen_task, (line, eda), callback=callback)
    pool.close()
    pool.join()
