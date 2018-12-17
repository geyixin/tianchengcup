# tianchengcup
- 官方数据](https://pan.baidu.com/s/1D_8bc_ijIDNzo5zKhC5tKg)
- [比赛详情](http://www.dcjingsai.com/common/cmpt/2018%E5%B9%B4%E7%94%9C%E6%A9%99%E9%87%91%E8%9E%8D%E6%9D%AF%E5%A4%A7%E6%95%B0%E6%8D%AE%E5%BB%BA%E6%A8%A1%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)
- kind.py：为了获得训练和预测中，就transaction和operation数据而言，某些特征的总出现情况；
- delete_phone_row.py：我觉得黑户会不会有iPhone和安卓用户的区别；
- data_process.py：将数据处理成适合训练的数据，比如特征的数字化等；
- data_analysis_lightgbm.py：分别训练transaction和operation数据；
- merge_predict.py：把分别训练的数据合在一起。由于提交发现operation的预测不是很准确，我采取的策略是从operation预测结果中取transaction预测结果中的数据
- 不要谈名次，受伤。。。。 :cry:
