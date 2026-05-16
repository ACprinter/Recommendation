# DeepFM相较于FM添加了共享的embedding
# 获取共享的特征embedding
# concat_feature: [batch_size, num_features, embedding_dim]
concat_feature = concat_group_embedding(
    group_embedding_feature_dict, group_name, axis=1, flatten=False
)

# 1. FM组件：学习二阶特征交叉
fm_output = FM()(concat_feature)  # [B, 1]

# 2. DNN组件：学习高阶非线性特征交叉
# 将embedding展平作为DNN输入
flatten_feature = tf.keras.layers.Flatten()(concat_feature)  # [B, N*D]
dnn_output = DNNs(
    units=[64, 32, 1],  # 多层神经网络
    activation="relu",
    dropout_rate=0.1
)(flatten_feature)  # [B, 1]

# 3. 联合训练：将FM和DNN的输出相加
deepfm_logits = tf.add(fm_output, dnn_output)  # [B, 1]
output = tf.keras.layers.Dense(1, activation="sigmoid")(deepfm_logits)