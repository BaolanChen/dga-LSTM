# DGA-LSTM based on deep learning
## 构建LSTM模型来预测DGA

### 针对由域名生成算法（DGA）生成的 C&C 域名具有生成方式多样、特征难以统一提取的问题，提出一种对域名进行二元文法（Bigram）处理后，通过 Bi-LSTM 模型进行 DGA 域名检测的方法。
### 首先对所有域名样本进行二元文法（Bigram）处理，并对每一条处理后的域名样本进行长度统计分析，将处理后的域名进行编码并有长度统计结果固定长度，通过 Bi-LSTM 网络模型获得检测结果。
### 实验使用 Alexa 网站公开的 100 万条域名数据和 28 种 DGA 生成算法生成的共计 140 万余条 DGA 数据样本。
### 综合比较 Unigram-LSTM、Bigram-LSTM、Unigram-Bi-LSTM、Bigram-Bi-LSTM 的实验结果，最优模型的 dga 误判率1：100
