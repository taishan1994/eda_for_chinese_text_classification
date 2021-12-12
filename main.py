import os

import jieba
import numpy as np
import logging
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup

from config.config import BilstmForClassificationConfig
from processor.processor import DataProcessor
from models.bilstmForClassification import BilstmForSequenceClassification
from utils.utils import set_seed, set_logger

import warnings

warnings.filterwarnings("ignore")


# logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, criterion, device, config):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.device = device

    def load_ckp(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self, train_loader, dev_loader=None):
        total_step = len(train_loader) * self.config.epochs
        self.optimizer, self.scheduler = self.build_optimizers(
            self.model,
            self.config,
            total_step
        )
        global_step = 0
        eval_step = 100
        best_dev_macro_f1 = 0.0
        for epoch in range(self.config.epochs):
            for train_step, train_data in enumerate(train_loader):
                self.model.train()

                input_ids, label_ids = train_data
                sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                input_ids = input_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                train_outputs = self.model(input_ids, sentence_lengths)

                loss = self.criterion(train_outputs, label_ids)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                print(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                if dev_loader:
                    if global_step % eval_step == 0:
                        dev_loss, dev_outputs, dev_targets = self.dev(dev_loader)
                        accuracy, precision, recall, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                        print(
                            "【dev】 loss：{:.6f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} macro_f1：{:.4f}".format(
                                dev_loss,
                                accuracy,
                                precision,
                                recall,
                                macro_f1))
                        if macro_f1 > best_dev_macro_f1:
                            checkpoint = {
                                'state_dict': self.model.state_dict(),
                            }
                            best_dev_macro_f1 = macro_f1
                            checkpoint_path = os.path.join(self.config.save_dir, '{}_best.pt'.format(model_name))
                            self.save_ckp(checkpoint, checkpoint_path)

        checkpoint_path = os.path.join(self.config.save_dir, '{}_final.pt'.format(model_name))
        checkpoint = {
            'state_dict': self.model.state_dict(),
        }
        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self, dev_loader):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(dev_loader):
                input_ids, label_ids = dev_data
                sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                input_ids = input_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                outputs = self.model(input_ids, sentence_lengths)
                loss = self.criterion(outputs, label_ids)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(label_ids.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path, test_loader):
        model = self.model
        model = self.load_ckp(model, checkpoint_path)
        model.eval()
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                input_ids, label_ids = test_data
                sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                input_ids = input_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                outputs = model(input_ids, sentence_lengths)

                loss = self.criterion(outputs, label_ids)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(label_ids.cpu().detach().numpy().tolist())
                # print('========================')
                # print(outputs.tolist())
                # print(label_ids.cpu().detach().numpy().tolist())
                # print('========================')
        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, checkpoint):
        model = self.model
        model = self.load_ckp(model, checkpoint)
        model.eval()
        with torch.no_grad():
            if self.config.use_word:
                inputs = [tokenizer["char2id"].get(word, 1) for word in jieba.lcut(text)]
            else:
                inputs = [tokenizer["char2id"].get(char, 1) for char in text]
            print([tokenizer['id2char'][i] for i in inputs])
            if len(inputs) >= self.config.max_seq_len:
                input_ids = inputs[:self.config.max_seq_len]
            else:
                input_ids = inputs + [0] * (self.config.max_seq_len - len(inputs))
            input_ids = torch.tensor([input_ids])
            sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
            input_ids = input_ids.to(self.device)
            outputs = model(input_ids, sentence_lengths)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=-1).flatten().tolist()

            if len(outputs) != 0:
                outputs = [self.config.id2label[i] for i in outputs]
                return outputs
            else:
                return '不好意思，我没有识别出来'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        precision = precision_score(targets, outputs, average='macro')
        recall = recall_score(targets, outputs, average='macro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, precision, recall, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        report = classification_report(targets, outputs, target_names=labels, digits=4)
        return report

    def build_optimizers(self, model, config, t_total):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=config.lr,
                          eps=config.adam_epsilon)
        warmup_steps = int(config.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        return optimizer, scheduler


if __name__ == '__main__':
    bilstmForClsConfig = BilstmForClassificationConfig()
    if bilstmForClsConfig.use_word:
        # 使用eda数据增强时是过滤掉了停用词的，但原始文本还是存在的
        if bilstmForClsConfig.use_stopword:
            data_name = "eda_bilstm_word2"
            model_name = 'eda_bilstm_word2'
        else:
            data_name = "eda_bilstm_word"
            model_name = 'eda_bilstm_word'
    else:
        data_name = "eda_bilstm"
        model_name = 'eda_bilstm'
    print("data_name:", data_name)
    print("model_name:", model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(123)
    # set_logger('{}.log'.format(model_name))

    processor = DataProcessor()
    max_seq_len = bilstmForClsConfig.max_seq_len
    tokenizer = {}
    char2id = {}
    id2char = {}
    print(len(bilstmForClsConfig.vocab))
    for i, char in enumerate(bilstmForClsConfig.vocab):
        char2id[char] = i
        id2char[i] = char
    tokenizer["char2id"] = char2id
    tokenizer["id2char"] = id2char

    model = BilstmForSequenceClassification(
        len(bilstmForClsConfig.labels),
        bilstmForClsConfig.vocab_size,
        bilstmForClsConfig.word_embedding_dimension,
        bilstmForClsConfig.hidden_dim
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, criterion, device, bilstmForClsConfig)

    if bilstmForClsConfig.do_train:
        train_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, 'eda_cnews.train.txt'))
        train_dataset = processor.get_examples(
            raw_examples=train_examples,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            pickle_path='./data/THUCNews/train_{}.pkl'.format(data_name),
            label2id=bilstmForClsConfig.label2id,
            set_type='train',
            use_word=bilstmForClsConfig.use_word)
        train_loader = DataLoader(train_dataset, batch_size=bilstmForClsConfig.train_batch_size, shuffle=True)

        if bilstmForClsConfig.do_eval:
            eval_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, 'cnews.val.txt'))
            eval_dataset = processor.get_examples(
                raw_examples=eval_examples,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer,
                pickle_path='./data/THUCNews/eval_{}.pkl'.format(data_name),
                label2id=bilstmForClsConfig.label2id,
                set_type='eval',
                use_word=bilstmForClsConfig.use_word)
            eval_loader = DataLoader(eval_dataset, batch_size=bilstmForClsConfig.eval_batch_size, shuffle=False)
            trainer.train(train_loader, eval_loader)
        else:
            trainer.train(train_loader)

    if bilstmForClsConfig.do_test:
        test_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, 'cnews.test.txt'))
        test_dataset = processor.get_examples(
            raw_examples=test_examples,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            pickle_path='./data/THUCNews/test_{}.pkl'.format(data_name),
            label2id=bilstmForClsConfig.label2id,
            set_type='test',
            use_word=bilstmForClsConfig.use_word,
        )
        test_loader = DataLoader(test_dataset, batch_size=bilstmForClsConfig.eval_batch_size, shuffle=False)
        total_loss, test_outputs, test_targets = trainer.test(
            os.path.join(bilstmForClsConfig.save_dir, '{}_final.pt'.format(model_name)),
            test_loader,
        )
        _, _, _, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
        print('macro_f1：{}'.format(macro_f1))
        report = trainer.get_classification_report(test_outputs, test_targets, labels=bilstmForClsConfig.labels)
        print(report)

    if bilstmForClsConfig.do_predict:
        checkpoint = os.path.join(bilstmForClsConfig.save_dir, '{}_final.pt'.format(model_name))
        with open(os.path.join(bilstmForClsConfig.data_dir, 'cnews.test.txt'), 'r') as fp:
            lines = fp.readlines()
            i = 1
            while i > 0:
                ind = np.random.randint(0, len(lines))
                line = lines[ind].strip().split('\t')
                text = line[1]
                print(text)
                result = trainer.predict(tokenizer, text, checkpoint)
                print("预测标签：", result)
                print("真实标签：", line[0])
                print("==========================")
                i -= 1
