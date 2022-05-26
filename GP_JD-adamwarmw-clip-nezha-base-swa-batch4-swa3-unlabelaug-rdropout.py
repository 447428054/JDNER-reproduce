#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
# 数据集 https://github.com/CLUEbenchmark/CLUENER2020

import json
import numpy as np
from bert4keras.backend import keras, K, search_layer
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer, EfficientGlobalPointer
from bert4keras.models import build_transformer_model
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
from swa.keras import SWA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

savedir = './saveGP-nezha-base-clip-1-swa3-unlabelaug-rdropout/10-adamwarmw-2e-6-batch4-epo26-0.1-1'
datadir = './data'
datatype = 'JD5'
bert_type = 'nezha'
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
epochs = 8
batch_size = 4
learning_rate = 2e-5
categories = set()

# bert配置
config_path = '/home/root1/lizheng/workspace/2022/pretraining/transpretrain-unlabelaug-nezha-base-ngram-epo26/bert_config.json'
checkpoint_path = '/home/root1/lizheng/workspace/2022/pretraining/transpretrain-unlabelaug-nezha-base-ngram-epo26/bert_model.ckpt'
dict_path = '/home/root1/lizheng/workspace/2022/pretraining/transpretrain-unlabelaug-nezha-base-ngram-epo26/vocab.txt'


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
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
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

def kullback_leibler_divergence_rdropout(y_true, y_pred):
    bh = K.prod(K.shape(y_pred)[:2]) # batch , heads
    y_true = K.reshape(y_true, (bh, -1)) # batch * heads , l * l
    y_pred = K.reshape(y_pred, (bh, -1))

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)

    y_pred_s = K.sigmoid(y_pred)
    y_true_s = K.sigmoid(y_true)

    return K.sum((y_true_s - y_pred_s) * (y_true - y_pred), axis=-1)

    # y_true_neg = 1 - y_true
    # y_pred_neg = 1 - y_pred
    # y_true_neg = K.clip(y_true_neg, K.epsilon(), 1)
    # y_pred_neg = K.clip(y_pred_neg, K.epsilon(), 1)
    # y_true = K.clip(y_true, K.epsilon(), 1)
    # y_pred = K.clip(y_pred, K.epsilon(), 1)
    # return y_true * K.log(y_true / y_pred) + y_true_neg * K.log(y_true_neg / y_pred_neg)

def global_pointer_rdropout_crossentropy(y_true, y_pred, alpha=1): # batch * 2 , heads , l , l
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2]) # batch * 2 * heads
    y_true_n = K.reshape(y_true, (bh, -1)) # batch * heads * 2 , l * l
    y_pred_n = K.reshape(y_pred, (bh, -1))
    loss_ce = K.mean(multilabel_categorical_crossentropy(y_true_n, y_pred_n))
    loss_kl = kullback_leibler_divergence_rdropout(y_pred[::2], y_pred[1::2]) + kullback_leibler_divergence_rdropout(y_pred[1::2], y_pred[::2])

    return loss_ce + K.mean(loss_kl) / 4 * alpha

def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr



model = build_transformer_model(config_path, checkpoint_path, model=bert_type, dropout_rate=0.1)
output = GlobalPointer(len(categories), 64)(model.output)

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
    total, warmsteps = calc_train_steps(32000, batch_size, epochs, warmup_proportion)
    AdamW = extend_with_weight_decay(Adam)
    AdamWLR = extend_with_piecewise_linear_lr(AdamW)
    optimizer = AdamWLR(learning_rate=learning_rate,
                        weight_decay_rate=weight_decay,
                        lr_schedule={warmsteps: 1, total: 0.1},
                        clipvalue=1
                        )

lr_metric = get_lr_metric(optimizer)

model.compile(
    loss=global_pointer_rdropout_crossentropy,
    optimizer=optimizer,
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
    def __init__(self, i, valid_data, test_data, patience=60):
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

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数

# model.load_weights('saveGPfinetuning1stage/10-adamwarmw-8e-6-testA/best_model_0.weights')
first_weights = model.get_weights()

fold = 5 # K折交叉验证
K_f1s = []
for i in range(fold):
    if i > 0:
        continue
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
    swa = SWA(
        start_epoch=4,
        lr_schedule='constant',
        swa_lr=2e-6,
        swa_freq=1,
        verbose=1
    )

    allcallbacks.append(swa)
    model.set_weights(first_weights) # 每次都

    # 写好函数后，启用对抗训练只需要一行代码
    # adversarial_training(model, 'Embedding-Token', 0.5)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=allcallbacks
    )

    '''
    SWA预测
    '''
    f1, precision, recall, valid_metric_dict = evaluate(valid_data)
    test_f1, test_precision, test_recall, test_metric_dict = evaluate(test_data)
    best_f1 = max(f1, evaluator.best_val_f1)
    best_test_f1 = max(test_f1, evaluator.test_f1)
    if f1 > evaluator.best_val_f1:
        model.save_weights(os.path.join(savedir, 'best_model_{}.weights'.format(i)))
    logger.info(
        'SWA valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
        (f1, precision, recall, best_f1)
    )
    logger.info(
        'SWA test:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
        (test_f1, test_precision, test_recall, best_test_f1)
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
