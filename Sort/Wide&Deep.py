# Wide部分计算
# 遍历所有需要交叉的特征对
for i in range(len(cross_feature_columns)):
    for j in range(i + 1, len(cross_feature_columns)):
        fc_i = cross_feature_columns[i]
        fc_j = cross_feature_columns[j]

        # 获取两个特征的输入
        feat_i = input_layer_dict[fc_i.name]  # [B, 1]
        feat_j = input_layer_dict[fc_j.name]  # [B, 1]

        # 为每个特征对创建独立的权重表
        cross_vocab_size = fc_i.vocab_size * fc_j.vocab_size
        cross_embedding = Embedding(
            input_dim=cross_vocab_size,
            output_dim=1,  # 标量权重，直接记住这对特征的影响
            name=f"cross_{fc_i.name}_{fc_j.name}"
        )

        # 将特征对组合成单一索引并查找权重
        combined_index = feat_i * fc_j.vocab_size + feat_j
        cross_weight = cross_embedding(combined_index)  # 查表得到这对特征的权重
        cross_weights.append(cross_weight)

# 所有交叉特征权重相加
cross_logits = tf.add_n(cross_weights)

# Deep部分
# 1. 特征嵌入：将稀疏的类别特征转换为稠密向量
group_feature_dict = {}
for group_name, _ in group_embedding_feature_dict.items():
    group_feature_dict[group_name] = concat_group_embedding(
        group_embedding_feature_dict, group_name, axis=1, flatten=True
    )  # B x (N * D) - 拼接所有特征的嵌入向量

# 2. 深度神经网络：逐层学习特征的非线性组合
deep_logits = []
for group_name, group_feature in group_feature_dict.items():
    # 构建多层神经网络
    deep_out = DNNs(
        units=dnn_units,  # 例如 [64, 32]
        activation="relu",  # ReLU激活函数
        dropout_rate=dnn_dropout_rate
    )(group_feature)

    # 输出层：将深度特征映射为预测分数
    deep_logit = tf.keras.layers.Dense(1, activation=None)(deep_out)
    deep_logits.append(deep_logit)

# 将Wide和Deep部分的输出相加得到最终预测分数
## Wide部分：线性特征 + 交叉特征
linear_logit = get_linear_logits(input_layer_dict, feature_columns)
cross_logit = get_cross_logits(input_layer_dict, feature_columns)

## Deep部分：多个特征组的深度网络输出
deep_logits = []
for group_name, group_feature in group_feature_dict.items():
    deep_out = DNNs(units=dnn_units, activation="relu", dropout_rate=dnn_dropout_rate)(
        group_feature
    )
    deep_logit = tf.keras.layers.Dense(1, activation=None)(deep_out)
    deep_logits.append(deep_logit)

## 联合训练：将Wide和Deep的输出相加
wide_deep_logits = add_tensor_func(deep_logits + [linear_logit, cross_logit])

## 最终预测：通过sigmoid函数输出点击概率
output = tf.keras.layers.Dense(1, activation="sigmoid")(wide_deep_logits)