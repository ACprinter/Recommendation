# 将用户或者Item用一个dim维度的隐含因子进行表征
from gensim.models import Word2Vec
# Item2Vec模型实现
def fit(self, train_hist_movie_id_list):
    # train_hist_movie_id_list: 用户交互序列列表
    # 每个元素是一个用户的物品ID序列
    self.model = Word2Vec(
        train_hist_movie_id_list,
        vector_size=self.model_config["EmbDim"],      # 嵌入维度
        window=self.model_config["Window"],           # 上下文窗口大小
        min_count=self.model_config["MinCount"],      # 最小出现次数
        workers=self.model_config["Workers"],         # 并行线程数
    )