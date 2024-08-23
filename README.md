# 基于iTransformer-TCN的音乐类别识别

其中[iTransformer](https://github.com/lucidrains/iTransformer)部分参照了lucidrains的代码。
有并行，TCN在前与iTransformer在前的串行结构，共三种类型。
实验得知iTransformer-TCN的效果最好。
数据集用的是kaggle上的[Gtzan](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)数据集，
可根据需求选择特定的特征。
![模型结构]('structure.png')
