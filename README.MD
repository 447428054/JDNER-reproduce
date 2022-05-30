# 前言

临近毕业，参加了京东NER的比赛，团队名为"JMXGODLZZ"。参赛过程中验证了许多想法，做了许多尝试，也学习了群内大佬不少的方法
但遗憾未进入下一轮，将本次比赛的过程特地进行梳理记录。

# 模型结构尝试

## RDropout

详细内容可见苏神博客：

https://kexue.fm/archives/8496

https://kexue.fm/archives/9039

### 设计目的

将Dropout两次的方式推广到一般任务中。两次dropout预测结果除了结果一致，还加上了分布一致的约束。

### 技术方案

GP的度量函数为：
$$
D(s,t)=(\sigma(s_i)-\sigma(t_i))(s_i - t_i)
$$
核心代码如下：

```
def kullback_leibler_divergence_rdropout(y_true, y_pred):
    bh = K.prod(K.shape(y_pred)[:2]) # batch , heads
    y_true = K.reshape(y_true, (bh, -1)) # batch * heads , l * l
    y_pred = K.reshape(y_pred, (bh, -1))

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)

    y_pred_s = K.sigmoid(y_pred)
    y_true_s = K.sigmoid(y_true)

    return K.sum((y_true_s - y_pred_s) * (y_true - y_pred), axis=-1)

def global_pointer_rdropout_crossentropy(y_true, y_pred, alpha=1): # batch * 2 , heads , l , l
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2]) # batch * 2 * heads
    y_true_n = K.reshape(y_true, (bh, -1)) # batch * heads * 2 , l * l
    y_pred_n = K.reshape(y_pred, (bh, -1))
    loss_ce = K.mean(multilabel_categorical_crossentropy(y_true_n, y_pred_n))
    loss_kl = kullback_leibler_divergence_rdropout(y_pred[::2], y_pred[1::2]) + kullback_leibler_divergence_rdropout(y_pred[1::2], y_pred[::2])

    return loss_ce + K.mean(loss_kl) / 4 * alpha
```

### 使用总结

比赛使用提升效果不明显，其中alpha与dropout相当于度量损失的权重。当alpha以及dropout设置的参数较小时，效果下降不明显；当dropout与alpha较大时，效果明显下降。猜测当权重过大时，度量任务对主任务产生了影响。

## SimCSE

详细内容可见苏神博客：

https://kexue.fm/archives/8348

### 设计目的

引入对比学习的方法，dropout作为数据增强手段，一个batch内相同的dropout数据为正例，其他数据为负例。

### 技术方案

损失函数计算公式如下：
$$
-\sum_{i=1}^N\sum_{\alpha=0,1}\log \frac{e^{\cos(\boldsymbol{h}^{(\alpha)}_i, \boldsymbol{h}^{(1-\alpha)}_i)/\tau}}{\sum\limits_{j=1,j\neq i}^N e^{\cos(\boldsymbol{h}^{(\alpha)}_i, \boldsymbol{h}^{(\alpha)}_j)/\tau} + \sum\limits_j^N e^{\cos(\boldsymbol{h}^{(\alpha)}_i, \boldsymbol{h}^{(1-\alpha)}_j)/\tau}}
$$
核心代码如下：

```
def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)
```

### 使用总结

比赛中使用未取得提升，按照群内讨论，仅在第一轮使用SimCSE，同时预训练阶段也加上SimCSE，可能会带来提升。其中SimCSE使模型学习到更好的语义表示，得到更好的特征表示向量。

## 多任务-辅助边界

### 设计目的

根据分析结果，许多多类别实体由于标注的不一致性，导致边界以及类别错误。因此，希望提高边界的识别准确率，来提高整体准确性。

### 技术方案

再增加一层GP用以边界识别，核心代码如下：

```
model = build_transformer_model(config_path, checkpoint_path, model=bert_type)

output = GlobalPointer(len(categories), 64)(model.output)
bdoutput = GlobalPointer(1, 64)(model.output)

model = Model(model.input, [output, bdoutput])
mainmodel = Model(model.input, output)
model.summary()
```

### 使用总结

使用真实训练集时，取得微小提升；当使用伪标签时，由于标签准确性不高，导致整体准确率下降。

## 多任务-查找主要实体

### 设计目的

比赛数据中，类别4是主要实体，增加主要实体的识别准确率，以提高整体识别的准确率。

### 技术方案

### 使用总结

取得微小提升，毕竟主要类别边界错误非主要错误原因。

## GPKG

### 设计目的

通过嵌入实体知识，提高少数实体、多类别实体的准确率。

### 使用总结

- 知识嵌入受知识集合准确性影响
- 比赛场景下，多类别实体占比较高，当使用频率最高类别时，效果下降；全部类别嵌入，效果也下降。因此，多类别实体的知识嵌入还需要进一步研究。
- 当使用低频、单类别实体知识嵌入时，模型在验证集取得提升，但在测试集下降。知识嵌入效果与知识密切相关，无可靠知识来源，方法泛化性差。

## 多任务-log sum

### 设计目的

受苏神多任务的启发，在设计多任务时，通过log求和的方式，调节不同任务的权重。

### 技术方案

因为keras多个输出loss会自动求和，因此在输出loss时，完成log运算，核心代码如下：

```
def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    lossvalue = K.mean(multilabel_categorical_crossentropy(y_true, y_pred))
    lossvalue = K.clip(lossvalue, 1e-7, 1e7)
    return K.log(lossvalue)
```

### 使用总结

比赛中效果下降了，若实现有问题还请指正。猜测由于设计的边界、主要实体任务都是辅助任务，应当在不影响主任务情况下进行优化。这也解释了当设计辅助任务常数权重较大时，模型效果也变差的原因。

## Label Smooth

详细内容可见苏神博客：

https://kexue.fm/archives/9064

### 设计目的

用软标签替代硬标签，提高泛化性，同时可以与软标签的操作如mixup结合。

### 技术方案

计算公式如下：
$$
\log\left(1+\sum_i e^{-s_i + \log p_i}\right)+\log\left(1+\sum_i e^{s_i + \log (1-p_i)}\right)
$$
核心代码如下：

```
def multilabel_categorical_crossentropy_labelsmooth(y_true, y_pred, ls=0.3):
    """多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素非0即1，
           1表示对应的类为目标类，0表示对应的类为非目标类；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """

    y_true_ls = y_true * (1 - ls) + ls / 2
    y_true_ls = K.clip(y_true_ls, 1e-17, 1 - 1e-17) # 避免log0
    '''
    避免mask位置 标签为0， 导致-si + logp 又变成了正数, 假设ls=0.2 标签0-》0.1 log 0.1 不足以降低成为负数
    其实就是在正例-si 将原先标签0的 变负无穷 ，将负例si 把原先1的变负无穷
    对于正例 确实从1-》0.9 对于负例 确实0-》0.1
    '''
    y_pred_neg = y_pred
    y_pred_pos = -y_pred
    # y_pred = (1 - 2 * y_true) * y_pred # 正例取负数
    # y_pred_neg = y_pred - y_true * K.infinity()
    # y_pred_pos = y_pred - (1 - y_true) * K.infinity()

    y_pred_pos += K.log(y_true_ls) + -K.infinity() * (1 - y_true)
    y_pred_neg += K.log(1 - y_true_ls) + -K.infinity() * y_true

    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
    pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss
```

### 使用总结

比赛中效果下降，若实现有错误请指正。**下降原因有待探究。**

## 动态拼接

### 设计目的

BERT中每一层学习的侧重点不同，为每一层设计权重，融合每一层的输出结果，替代原先取最后一层输出的结果。

### 技术方案

核心代码如下：

```
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
```

### 使用总结

根据权重，动态融合BERT不同层的输出，以获取更好的特征表示。比赛中该方法未取得提升，可能增加了层数与参数，更容易导致过拟合。

## 引入拼音字形特征

### 设计目的

参考CCL比赛第一名方案:https://zhuanlan.zhihu.com/p/489640773融入通用的拼音与字形特征。

CCL比赛起效果原因：病历中很多字的偏旁代表了一定语义。

### 技术方案

#### 数据处理

利用pypinyin获取拼音，利用字形pickle文件递归拆字

#### 模型结构

输入的拼音与字形分别经过Embedding、CNNEncoder提取特征向量，再reshape与BERT特征向量拼接作为下游任务输入向量。其中CNNEncoder利用多个卷积核，完成卷积。

主要代码文件：

```
chaizi.py
models.py:GlobalPointerMF/GlobalPointerMFV2
```

核心代码如下：

```
self.pinyinEmbedding = Embedding(input_dim=self.pinyin_size, output_dim=384)
self.zixingEmbedding = Embedding(self.zixing_size, 384)

self.pinyinEncoder = CNNEncoder(embedding_dim=256, num_filters=128, ngram_filter_sizes=(2, 3, 4, 5), output_dim=384)
self.zixingEncoder = CNNEncoder(embedding_dim=256, num_filters=128, ngram_filter_sizes=(3, 4, 5, 6), output_dim=384)

...

pinyinemb = self.pinyinEmbedding(pinyin) # batch, seqlen, pylen, embsize
zixingemb = self.zixingEmbedding(zixing) # batch, seqlen, zxlen, embsize

encoded_pinyin = self.pinyinEncoder(pinyinemb) # batch * seqlen, outdim
encoded_zixing = self.zixingEncoder(zixingemb) # batch * seqlen, outdim

inputs = K.concatenate([inputs, encoded_pinyin, encoded_zixing], axis=-1)
```

### 使用总结

比赛中该方法未取得明显提升，可能与电商数据下数据特征有关，拼音字形没起到明显辅助作用。拼音字形可能在纠错、生僻字、特定领域下起到辅助作用。

## Multisample Dropout

### 设计目的

以多次dropout输出层的方式，完成数据增强，防止过拟合。

### 技术方案

本次比赛，通过重复比赛数据，多次前向传播完成代码实现，核心代码如下：

```
for i in range(2):
  batch_token_ids.append(token_ids)
  batch_segment_ids.append(segment_ids)
  batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
if len(batch_token_ids) == self.batch_size * 2 or is_end:
  batch_token_ids = sequence_padding(batch_token_ids)
  batch_segment_ids = sequence_padding(batch_segment_ids)
  batch_labels = sequence_padding(batch_labels, seq_dims=3)
  yield [batch_token_ids, batch_segment_ids], batch_labels
...
model = build_transformer_model(config_path, checkpoint_path, model=bert_type, dropout_rate=0.1)
output = GlobalPointer(len(categories), 64)(model.output)
```

论文中的实现如下，该实现方式相当于模型集成：

```
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes,dropout_num,dropout_p):
        super(ResNet, self).__init__()
        self.inchannel = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 32,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.fc = nn.Linear(128,num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x,y = None,loss_fn = None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feature = F.avg_pool2d(out, 4)
        if len(self.dropouts) == 0:
            out = feature.view(feature.size(0), -1)
            out = self.fc(out)
            if loss_fn is not None:
                loss = loss_fn(out,y)
                return out,loss
            return out,None
        else:
            for i,dropout in enumerate(self.dropouts):
                if i== 0:
                    out = dropout(feature)
                    out = out.view(out.size(0),-1)
                    out = self.fc(out)
                    if loss_fn is not None:
                        loss = loss_fn(out, y)
                else:
                    temp_out = dropout(feature)
                    temp_out = temp_out.view(temp_out.size(0),-1)
                    out =out+ self.fc(temp_out)
                    if loss_fn is not None:
                        loss = loss+loss_fn(temp_out, y)
            if loss_fn is not None:
                return out / len(self.dropouts),loss / len(self.dropouts)
            return out,None
```

### 使用总结

该方法多次dropout来达到数据增强的目的，提高模型泛化性。比赛中未取得明显提升，可能相同数据下，未能学习到更多特征。

## EFGP

具体描述可见苏神博客：

https://kexue.fm/archives/8877

### 设计目的

减少参数量，避免过拟合。同时更换了打分计算函数，从点积计算方式更换为类似于biaffine的打分函数。

### 使用总结

比赛中效果变差，可能参数量减少，特征拟合的变差了。

## mixup

参考苏神博客：

https://kexue.fm/archives/5693

待实现，听群里大佬交流，该方法在比赛中是能够起到提升的作用。