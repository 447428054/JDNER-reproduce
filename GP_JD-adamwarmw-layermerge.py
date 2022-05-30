#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
# 数据集 https://github.com/CLUEbenchmark/CLUENER2020

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4kerasmodels import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import logging
import time
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr, extend_with_parameter_wise_lr, extend_with_weight_decay
from utils.optimizer import calc_train_steps
from keras.layers import *
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

savedir = './saveGPlayermergelast4/10-adamwarmw-8e-6-trunknormal'
datadir = './data'
datatype = 'JD10pseudo-0.807'
bert_type = 'bert'
model_scale = 'base'
optimizertype = 'adamwarmw'
is_flat = True
warmup_proportion = 0.1
weight_decay = 0.01

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_dir = os.path.join(savedir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_name = os.path.join(log_dir, '{}.log'.format(rq))
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)

maxlen = 128
epochs = 20
batch_size = 16
learning_rate = 8e-6
categories = set()

# bert配置
config_path = '/home/root1/lizheng/pretrainModels/tensorflow/chinese/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/root1/lizheng/pretrainModels/tensorflow/chinese/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/root1/lizheng/pretrainModels/tensorflow/chinese/chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = [l['text']]
            for k, v in l['label'].items():
                # categories.add(k)
                for spans in v.values():
                    for start, end in spans:
                        d.append((start, end, k))
            D.append(d)
    return D


# 标注数据
# train_data = load_data('/root/ner/cluener/train.json')
# valid_data = load_data('/root/ner/cluener/dev.json')
categories = []
for x in range(1, 55):
    if x not in [27, 45]:
        categories.append(str(x))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

class LayerMerge(Layer):
    def __init__(self, layernum):
        self.layernum = layernum

    def __build__(self):
        self.kernel = tf.truncated_normal_initializer(stddev=0.02)

    def __call__(self, encoders):
        layernum = self.layernum
        denselst = []
        for i in range(layernum):
            denselst.append(Dense(1, name='dense{}'.format(i), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)))
        layer_logits = []
        for i, layer in enumerate(encoders):
            print("layer:", layer)
            ds = denselst[i]
            layer_logits.append(
                ds(layer)
            )
        print("np.array(layer_logits).shape:", np.array(layer_logits).shape)
        # layer_logits = tf.concat(layer_logits, axis=2)
        layer_logits = Concatenate(axis=2)(layer_logits)
        print("layer_logits.shape:", layer_logits.shape)
        # layer_dist = tf.nn.softmax(layer_logits)  # [batchszie,max_len,12]
        layer_dist = Softmax()(layer_logits)  # [batchszie,max_len,12]
        print("layer_dist.shape:", layer_dist.shape)
        # seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in encoders], axis=2)
        ep = Lambda(lambda x: K.expand_dims(x, axis=2))
        seq_out = Concatenate(axis=2)([ep(x) for x in encoders])
        print("seq_out.shape:", seq_out.shape)  # [batchsize,max_len,12,768]
        # [batchsize,max_len,1,12] × [batchsize,max_len,12,768]
        mt = Lambda(lambda x: tf.matmul(x[0], x[1]))
        pooled_output = mt([ep(layer_dist), seq_out])  # batch * maxlen * 1 * 768
        sq = Lambda(lambda x: K.squeeze(x, axis=2))
        pooled_output = sq(pooled_output)
        print("pooled_output.shape:", pooled_output.shape)  # batch * maxlen * 768
        return pooled_output

    def compute_output_shape(self, input_shape):
        return input_shape


model, encoders = build_transformer_model(config_path, checkpoint_path, model=bert_type)
print(model.output.shape) # batch * maxlen * 768
layernum = len(encoders)
pooled_output = LayerMerge(4)(encoders[-4:])

output = GlobalPointer(len(categories), 64)(pooled_output)

model = Model(model.input, output)
model.summary()

# TODO warmup涉及步数，与训练集有关，因此需要预设一下数量
if optimizertype == 'adam':
    optimizer = Adam(learning_rate)
elif optimizertype == 'adamwarm':
    total, warmsteps = calc_train_steps(len(train_data), batch_size, epochs, warmup_proportion)
    AdamLR = extend_with_piecewise_linear_lr(Adam)
    optimizer = AdamLR(learning_rate=learning_rate, lr_schedule={warmsteps: 1, total: 0.1})
elif optimizertype == 'adamdiff':
    AdamDF = extend_with_parameter_wise_lr(Adam)
    optimizer = AdamDF(learning_rate=learning_rate, paramwise_lr_schedule={'Transformer': 0.1}, decay=weight_decay)
elif optimizertype == 'adamw':
    AdamW = extend_with_weight_decay(Adam)
    optimizer = AdamW(learning_rate=learning_rate, weight_decay_rate=weight_decay)
elif optimizertype == 'adamwarmw':
    total, warmsteps = calc_train_steps(36000, batch_size, epochs, warmup_proportion)
    AdamW = extend_with_weight_decay(Adam)
    AdamWLR = extend_with_piecewise_linear_lr(AdamW)
    optimizer = AdamWLR(learning_rate=learning_rate,
                        weight_decay_rate=weight_decay,
                        lr_schedule={warmsteps: 1, total: 0.1})

lr_metric = get_lr_metric(optimizer)

model.compile(
    loss=global_pointer_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=[global_pointer_f1_score]
)


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        escores = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
            escores.append(scores[l][start][end])
        if True:
            alles = zip(entities, escores)
            alles = sorted(alles, key=lambda item: item[1], reverse=True)
            newentities = []
            for ((ns, ne, t), so) in alles:
                for (ts, te, _) in newentities:
                    if ns < ts <= ne < te or ts < ns <= te < ne:
                        # for both nested and flat ner no clash is allowed
                        break
                    if ns <= ts <= te <= ne or ts <= ns <= ne <= te:
                        # for flat ner nested mentions are not allowed
                        break
                else:
                    newentities.append((ns, ne, t))
            entities = newentities
        return entities


NER = NamedEntityRecognizer()


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    metric_dict = {}
    for l in categories:
        metric_dict.setdefault(l, {})
        metric_dict[l]['X'] = 1e-10
        metric_dict[l]['Y'] = 1e-10
        metric_dict[l]['Z'] = 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        for l in categories:
            TT = 0
            TP = 0
            PT = 0

            for entities in R:
                if entities[2] == l:
                    PT += 1
                    if entities in T:
                        TP += 1

            for entities in T:
                if entities[2] == l:
                    TT += 1

            metric_dict[l]['X'] += TP
            metric_dict[l]['Y'] += PT
            metric_dict[l]['Z'] += TT

        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    maf1 = 0
    mapr = 0
    mare = 0
    for k, v in metric_dict.items():
        x = v['X']
        y = v['Y']
        z = v['Z']
        metric_dict[k]['F1'] = 2 * x / (y + z)
        metric_dict[k]['Precision'] = x / y
        metric_dict[k]['Recall'] = x / z
        maf1 += metric_dict[k]['F1']
        mapr += metric_dict[k]['Precision']
        mare += metric_dict[k]['Recall']
    maf1 /= len(categories)
    mapr /= len(categories)
    mare /= len(categories)
    metric_dict.setdefault('Micro', {})
    metric_dict.setdefault('Macro', {})
    metric_dict['Micro']['F1'] = f1
    metric_dict['Micro']['Precision'] = precision
    metric_dict['Micro']['Recall'] = recall
    metric_dict['Macro']['F1'] = maf1
    metric_dict['Macro']['Precision'] = mapr
    metric_dict['Macro']['Recall'] = mare
    return f1, precision, recall, metric_dict


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, i, valid_data, test_data, patience=6):
        self.best_val_f1 = 0
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0

        self.dev_ps = []
        self.dev_rs = []
        self.dev_fs = []
        self.test_ps = []
        self.test_rs = []
        self.test_fs = []
        self.i = i
        self.patience = patience
        self.cp = 0 # 当前patience值
        self.valid_data = valid_data
        self.test_data = test_data


    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall, valid_metric_dict = evaluate(self.valid_data)
        test_f1, test_precision, test_recall, test_metric_dict = evaluate(self.test_data)
        self.dev_ps.append(precision)
        self.dev_rs.append(recall)
        self.dev_fs.append(f1)
        self.test_ps.append(test_precision)
        self.test_rs.append(test_recall)
        self.test_fs.append(test_f1)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.test_f1, self.test_precision, self.test_recall = test_f1, test_precision, test_recall
            model.save_weights(os.path.join(savedir, 'best_model_{}.weights'.format(i)))
            self.cp = 0
        else:
            self.cp += 1
        logger.info(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

        for tp, mdict in valid_metric_dict.items():
            str_ = 'valid %s:'%(tp)
            for metric, value in mdict.items():
                str_ += '%s: %.5f,'%(metric, value)
            str_ += '\n'
            logger.info(str_)

        logger.info(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f, best dev test f1: %.5f, best dev test precision: %.5f, best dev test recall: %.5f\n' %
            (test_f1, test_precision, test_recall, self.test_f1, self.test_precision, self.test_recall)
        )

        for tp, mdict in test_metric_dict.items():
            str_ = 'test %s:'%(tp)
            for metric, value in mdict.items():
                str_ += '%s: %.5f,'%(metric, value)
            str_ += '\n'
            logger.info(str_)

        # 早停
        if epoch > 3 and f1 < 0.1: # 防止训练崩溃
            self.model.stop_training = True

        if self.cp > self.patience:
            self.model.stop_training = True


def predict_to_file(in_file, out_file):
    """预测到文件
    可以提交到 https://www.cluebenchmarks.com/ner.html
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            l['label'] = {}
            for start, end, label in NER.recognize(l['text']):
                if label not in l['label']:
                    l['label'][label] = {}
                entity = l['text'][start:end + 1]
                if entity not in l['label'][label]:
                    l['label'][label][entity] = []
                l['label'][label][entity].append([start, end])
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()

def predict_to_JD(in_file, out_file):
    """预测到文件
    可以提交到 https://www.cluebenchmarks.com/ner.html
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            l['label'] = {}
            labels = []
            for _ in l['text']:
                labels.append('O')
            for start, end, label in NER.recognize(l['text']):
                for pid in range(start, end + 1):
                    if pid == start:
                        labels[pid] = 'B-{}'.format(label)
                    else:
                        labels[pid] = 'I-{}'.format(label)

            for (tw, tl) in zip(l['text'], labels):
                fw.write('{} {}\n'.format(tw,tl))
            fw.write('\n')

    fw.close()

first_weights = model.get_weights()
fold = 10 # K折交叉验证
K_f1s = []
for i in range(fold):
    # 标注数据
    train_data = load_data(os.path.join(datadir, datatype, 'JDtrain{}.txt'.format(i)))
    valid_data = load_data(os.path.join(datadir, datatype, 'JDtest{}.txt'.format(i)))
    test_data = load_data(os.path.join(datadir, datatype, 'JDtest{}.txt'.format(i)))
    print(os.path.join(datadir, datatype, 'JDtrain{}.txt'.format(i)))
    logger.info('---------------------------------------{}FOLD Start--------------------------------------------'.format(i))
    logger.info('{}'.format(train_data[:10]))
    # continue
    plt.figure()
    evaluator = Evaluator(i, valid_data, test_data)


    train_generator = data_generator(train_data, batch_size)

    allcallbacks = []
    allcallbacks.append(evaluator)

    model.set_weights(first_weights) # 每次都

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=allcallbacks
    )
    axs = [x + 1 for x in range(len(evaluator.dev_ps))] # 早停会影响数量
    plt.figure(1)
    plt.plot(axs, evaluator.dev_ps, label='precision')
    plt.plot(axs, evaluator.dev_rs, label='recall')
    plt.plot(axs, evaluator.dev_fs, label='f1')

    plt.legend()
    plt.title("dev eval")
    plt.xlabel("epoch")
    plt.ylabel("Eval")
    plt.savefig(os.path.join(savedir, 'DEV_{}_Eval.png'.format(i)))
    plt.figure(2)
    plt.plot(axs, evaluator.test_ps, label='precision')
    plt.plot(axs, evaluator.test_rs, label='recall')
    plt.plot(axs, evaluator.test_fs, label='f1')

    plt.legend()
    plt.title("test eval")
    plt.xlabel("epoch")
    plt.ylabel("Eval")
    plt.savefig(os.path.join(savedir, 'TEST_{}_Eval.png'.format(i)))

    K_f1s.append(evaluator.best_val_f1)

    fw = open('experiment_record_reproduce.txt', encoding='utf-8', mode='a')
    record = [datatype, bert_type, model_scale, optimizertype, 'globalpointer', maxlen, batch_size, learning_rate, i, evaluator.best_val_f1, evaluator.test_precision, evaluator.test_recall, evaluator.test_f1]
    record = [str(x) for x in record]
    fw.write('\t'.join(record) + '\n')

    # K.clear_session()
    # tf.reset_default_graph()

print(sum(K_f1s) / len(K_f1s))

#
# if __name__ == '__main__':
#
#     evaluator = Evaluator()
#     train_generator = data_generator(train_data, batch_size)
#
#     model.fit(
#         train_generator.forfit(),
#         steps_per_epoch=len(train_generator),
#         epochs=epochs,
#         callbacks=[evaluator]
#     )
#
# else:
#
#     model.load_weights('./best_model_cluener_globalpointer.weights')
#     # predict_to_file('/root/ner/cluener/test.json', 'cluener_test.json')
