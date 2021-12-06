分类任务的处理上：
1、单模效果bert-wmm-ext最好
2、rdrop有提升——相比单模0.68334881143
3、3个模型融合有提升(bert-0.68334881143/roberta-0.68088578315/ernie-0.68113771241)——相比单模0.68870695532
4、5折交叉验证有提升——相比单模0.69007040259最佳
5、bert_cls_dropout有提升——相比较5折验证——0.69921122577最佳
6、ema——没有提升——相比之前的结果有下降(logit——0.69514260400，vote——0.6983008)——比单模效果要好——可以调参
7、LA——没有提升——相比之前结果有下降(logit——0.67737011522，vote——0.67388736828)——比单模效果都要差
8、LCM——没有提升——相比之前略有下降(logit——0.69730853646 vote——0.68113405834)——比单模效果要好——可以调参
9、FGM对抗训练——下降(logit——0.69041896111,vote——0.67631585081)——可能和其他的方案有冲突，可调参——可能和cls dropout冲突
10、Focal——0.68803163290也下降一点点
11、5折cls+roberta——0.68658460112，下降——可以融合试试
12、5折cls+ernie——0.65621373509——下降太多
13、5折cls+roberta-large——没有训练
14、梯度裁剪——0.69155628558 下降一点点
15、3折和9折效果均下降——0.69765660885 和 0.69449594829
16、PGD——(0.70701575685 ner基础上——0.70628393679 略微降低)
17、label smooth——(0.70701575685 ner crf基础上——0.71397038051提升0.006——最佳)
18、半监督0——预估是有效果的——(label smooth中概率大于0.9的测试样本0.70701575685 ner crf基础上——logits 0.72168941161——提升0.0077，可以继续尝试把半监督0中概率大于0.9的测试样本加进来)
19、半监督1——预估是有效果的——(半监督0中概率大于0.9的测试样本 0.72168941161   ner crf基础上——logits 0.72259196 ——提升0.00090，可以继续尝试把半监督1中概率大于0.8的测试样本加进来)
20、半监督2——(半监督1中概率大于0.8的测试样本 0.72259196   ner crf基础上——logits——0.72483160559——提升——0.002223964，可以继续尝试把半监督2中概率大于0.8的测试样本加进来)——提升多一点
21、半监督2——(半监督1中概率大于0.8的测试样本 0.72259196   ner crf基础上——vote——0.70948867758——降低很多)
21、半监督3——(半监督1中概率大于0.9的测试样本 0.72259196   ner crf基础上——logits——0.72291676864——提升0.0003248)



最佳结果是：分类任务v25_Submission_rdrop_4.0_chinese-bert-wwm-extlogits_integrate_5_cls_dropout_smooth_prob_semi_supervised_2_classification_bert_crf_clsdropout_fgm_ner.csv——0.72483160559
加上
Ner crf最佳结果——v25_Submission_rdrop_4.0_bert_classficaition_bert_5logits_crf_ner_cls_droprout_PGD.csv——有提升，提升0.00045686:0.41235995663
v26_Submission_rdrop_4.0_chinese-bert-wwm-extlogits_integrate_5_cls_dropout_smooth_prob_semi_supervised_2_classification_bert_crf_clsdropout_pgd_ner.csv——0.72528852315



最优方案：bert-wmm-ext+linear_dropouts+PGD对抗训练+label_smooth+(3次半监督训练，选取的概率阈值0.9/0.9/0.8)+5折概率融合
可以采用多个预训练权重和不同的随机种子做投票
0.31292857




以上5折单模效果找到最佳，
采用3个预训练权重融合？(设定不同的cls_dropout,不同的attention_dropout以及alpha等)









NER任务的处理上(基于第一版的初始分类任务得分0.68334881143)：
crf
1、单模效果bert-wmm-ext  0.68334881143
数据不一样 下面的数据和第一版不一样
2、折交叉验证有提升——相比单模(logit——0.67754596849  vote——0.67597  可能代码有问题，按道理不应该的)——batch inference 和single inference之分
3、代码修改还是没有提升——(logit——0.67534  vote——0.0.66733)——不知道是那里的问题哦——验证出来单折处理效果要好一点。。。
5折得分——0.70112237615提交一点点
4、cls层dropout——0.70142963426——又提交一点点——最高的
5、3折——0.69765660885降低了；7折——0.70017600891降低一点点
6、FGM+vote——0.70701576——最佳:0.41134751773
7、FGM+logits——0.70701576——最佳:0.41190303907
8、PGD+vote——有提升，提升0.00045686:0.41078136739
7、PGD+logits——有提升，提升0.00045686:0.41235995663
9、半监督
v25_Submission_rdrop_4.0_bert_classficaition_bert_5logits_crf_ner_cls_droprout_PGD.csv——有提升，提升0.00045686:0.41235995663
还需要验证rdropout
bert+PGD+vote+semi_supervised——有提升，提升0.00045686:0.41252268603
bert+PGD+logits_supervised——有提升，提升0.00045686:0.41187082807
roberta+PGD+vote+semi_supervised:0.41178610404
roberta+PGD+logit+semi_supervised:0.41173285199




还没有验证不同的模型和span方法
span
1、bert5折融合——0.69786248863
2、bert5折融合+cls_dropout——0.69731051997(略微下降)
3、损失函数
bert5折融合+lsr:0.40449845819
bert5折融合+focal:0.39617137648
bert5折融合+ce:0.40362438221
bert5折融合+lsr+start_dropout:0.40279036057——dropout无效
lsr最好
4、bert5折融合+logits+lsr+FGM——0.40659741206——提升0.002098
5、bert5折融合+logits+PGD——0.40653523184——提升0.00203678——可以调参k值
4、bert5折融合+vote+lsr+FGM——0.40832710978——提升0.00382865
5、bert5折融合+vote+lsr+PGD——0.40829286515——提升0.00379441——可以调参k值
FGM和vote更好
v5_Submission_rdrop_4.0_bert_cls_dropout_classficaition_bert_vote_5_span_ner_lsr_FGM.csv——0.40832710978——提升0.00382865
6、半监督
还需要验证rdropout
bert5折融合+vote+lsr+FGM+semi_supervised:0.41320651180
bert5折融合+logits+lsr+FGM+semi_supervised:0.41184971098
roberta5折融合+vote+lsr+FGM+semi_supervised:0.41355994152
roberta5折融合+logits+lsr+FGM+semi_supervised:0.41069171031
ernie5折融合+vote+lsr+FGM+semi_supervised:0.40249905339
roberta_large5折融合+logits+lsr+FGM+semi_supervised:0.41160471442
roberta_large5折融合+vote+lsr+FGM+semi_supervised:0.41232400805



softmax
1、损失函数
bert5折融合+lsr+vote:0.40606287425
bert5折融合+lsr+logtis:0.40146507057
bert5折融合+focal+vote:0.40198464707
bert5折融合+focal+logits:0.39889640441
bert5折融合+ce+vote:0.40239565787
bert5折融合+ce+logits:0.40110085227
lsr最好
4、FGM+bert5折融合+lsr+logtis:0.40380850685
5、FGM+bert5折融合+lsr+vote:0.40689013035
6、PGD+bert5折融合+lsr+logtis:0.40389772930
7、PGD+bert5折融合+lsr+vote:0.40573770492
vote和FGM要好一点
8、半监督(预测是没有效果的，噪声对模型NER任务影响很大)
还需要验证linear_dropout
v6_Submission_rdrop_4.0_bert_cls_dropout_classficaition_bert_vote_5_softmax_ner_lsr_FGM.csv——0.40689013035
bert5折融合+vote+lsr+FGM+semi_supervised:0.41229029905
bert5折融合+logits+lsr+FGM+semi_supervised:0.41165854539
roberta5折融合+vote+lsr+FGM+semi_supervised:0.41173285199
roberta5折融合+logits+lsr+FGM+semi_supervised:0.41217201166




crf:0.41235995663
span:0.40832710978
softmax:0.40689013035
crf+span+softmax:0.41183015019



crf:0.41252268603
span:0.41355994152
softmax:0.41229029905
crf+span+softmax 0.8:0.41293260474


v0_Submission_bert_best_classification_crf_span_softmax_bert_best_vote_ner_20211116_threshold_0.6.csv

差异化以上3个模型，可以从句长随机种子等角度来进行差异化
