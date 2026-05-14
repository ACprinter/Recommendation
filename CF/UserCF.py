import numpy as np
import warnings 
warnings.filterwarnings('ignore')
user_data = {
    'user1': {'item1': 5, 'item2': 3, 'item3': 4, 'item4': 4},
    'user2': {'item1': 3, 'item2': 1, 'item3': 2, 'item4': 3, 'item5': 3},
    'user3': {'item1': 4, 'item2': 3, 'item3': 4, 'item4': 3, 'item5': 5},
    'user4': {'item1': 3, 'item2': 3, 'item3': 1, 'item4': 5, 'item5': 4},
    'user5': {'item1': 1, 'item2': 5, 'item3': 5, 'item4': 2, 'item5': 1},
}

# 预测用户1对物品5的评分
# 初始化相似性矩阵
import pandas as pd
similarity_matrix = pd.DataFrame(
    np.identity(len(user_data)),
    index=user_data.keys(),
    columns=user_data.keys(),
)

# 遍历每条用户-物品评分数据
for u1, items1 in user_data.items():
    for u2, items2 in user_data.items():
        if u1 == u2:
            continue
        vec1, vec2 = [], []
        for item, rating1 in items1.items():
            rating2 = items2.get(item, -1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        # 计算不同用户之间的皮尔逊相关系数
        similarity_matrix[u1][u2] = np.corrcoef(vec1, vec2)[0][1]

print(similarity_matrix)

target_user = 'user1'
num = 2
# 由于最相似的用户为自己，去除本身
sim_users = similarity_matrix[target_user].sort_values(ascending=False)[1:num+1].index.tolist()
print(f'与用户{target_user}最相似的{num}个用户为：{sim_users}')

weighted_scores = 0.
corr_values_sum = 0.

target_item = 'item5'
# 基于皮尔逊相关系数预测用户评分
for user in sim_users:
    corr_value = similarity_matrix[target_user][user]
    user_mean_rating = np.mean(list(user_data[user].values()))

    weighted_scores += corr_value * (user_data[user][target_item] - user_mean_rating)
    corr_values_sum += corr_value

target_user_mean_rating = np.mean(list(user_data[target_user].values()))
target_item_pred = target_user_mean_rating + weighted_scores / corr_values_sum
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred:.4f}')