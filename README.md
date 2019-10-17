# IFLY_advertisement-cheating_2019
2019科大讯飞AI反欺诈
competition back up  
如果您喜欢的话 欢迎star

背景(background):  
比赛地址（competition address）：[advertisement-cheating](http://challenge.xfyun.cn/2019/gamedetail?type=detail/mobileAD)
数据集地址（dataset address）：[dataset](https://www.kaggle.com/ssslarry/ifly-antifraud)  (别人上传的)
广告欺诈是数字营销面临的一个重大挑战，随着基础技术的成熟化（篡改设备信息、IPv4服务化、黑卡、接码平台等），广告欺诈呈现出规模化、集团化的趋势，其作弊手段主要包括机器人、模拟器、群控、肉鸡/后门、众包等。广告欺诈不断蚕食着营销生态，反欺诈成为数字营销领域亟待突破的关键问题。

### 1. 依赖（dependence）
    python3.6
### 2. 数据分析（data analysis）
    update the basic analysis on github.
[比赛数据分析notebook](https://github.com/bochuanwu/IFLY_advertisement-cheating_2019/blob/master/%E7%A7%91%E5%A4%A7%E8%AE%AF%E9%A3%9E%E7%A7%BB%E5%8A%A8%E5%B9%BF%E5%91%8A%E5%8F%8D%E6%AC%BA%E8%AF%88%E7%AE%97%E6%B3%95%E6%8C%91%E6%88%98%E8%B5%9B%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90.ipynb)
### 3. 模型选择（model chosen）
    catboost==0.15
    lightgbm
    neural network(keras == 2.2.4)
### 4. 模型评测（model evaluation）
    Final test B top30
### 5. 模型文件（model document）
    catboost document == catboost_demo1.py
    lightgbm document == to do
    NN document == to do
