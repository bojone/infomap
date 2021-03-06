# infomap
一个漂亮的聚类/社区发现算法

## 链接
- https://www.mapequation.org
- https://kexue.fm/archives/7006

## 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn

## 使用
通过下述代码安装infomap，然后执行`python word_cluster.py`即可。
```
wget -c https://github.com/mapequation/infomap/archive/6ab17f8b18a6fdf34b2a53454f79a3b976a49201.zip
unzip 6ab17f8b18a6fdf34b2a53454f79a3b976a49201.zip
cd infomap-6ab17f8b18a6fdf34b2a53454f79a3b976a49201
cd examples/python
make

# 编译完之后，当前目录下就会有一个infomap文件夹，就是编译好的模块；
# 为了方便调用，可以复制到python的模块文件夹（每台电脑的路径可能不一样）中
python example-simple.py
cp infomap /home/you/you/python/lib/python2.7/site-packages -rf
```

参考词向量：链接:https://pan.baidu.com/s/1YYE2T3f-lPyLBrJuUowAsA 密码:5p0h

## 效果
<blockquote>
[u'妹妹', u'姐姐', u'哥哥', u'弟弟', u'爸爸', u'儿子', u'母亲', u'女儿', u'父亲', u'妻子', u'爷爷', u'老婆', u'丈夫', u'男友', u'女友', u'爱人', u'妈妈', u'长子', u'小时候', u'父母', u'祖父', u'情人', u' 亲人', u'弟', u'家人', u'夫妇', u'妻', u'兄', u'妹', u'家里', u'姐妹', u'父', u'嫁给', u'从小', u'子女', u'嫁', u'夫', u'家中', u'娶', u'姐', u'叔', u'长大', u'夫人', u'父子', u'在家', u'娘', u'兄弟', u' 家属', u'奴', u'母', u'子', u'哥', u'儿', u'儿女', u'小姐']

[u'加强', u'建立健全', u'推进', u'拟定', u'落实', u'提高', u'规章制度', u'会同', u'搞好', u'切实', u'负责', u'促进', u'贯彻落实', u'制订', u'增强', u'抓好', u'深化', u'组织协调', u'拟订', u'加快', u'加大', u'实施', u'着力', u'有利于', u'制定', u'推动', u'进一步', u'贯彻', u'督促', u'改善', u'大力', u'贯彻执行', u'充分发挥', u'牵头', u'政策措施', u'年度计划', u'强化', u'提升', u'增进', u'中长期', u'健全', u'优化', u'方针', u'做好', u'协调', u'统筹', u'承办', u'协助', u'全力', u'积极', u'责任制', u'党和国家', u'步伐', u'力度', u'承担', u'开展', u'巩固', u'编制', u'有利', u'改进', u'加深', u'不利', u'各项', u'加速', u'围绕', u'组织', u'衔接', u'认真']

[u'股权', u'证券', u'融资', u'股票', u'金融机构', u'上市公司', u'信贷', u'债券', u'商业银行', u'贷款', u'金融', u'期货', u'信托', u'存款', u'债权', u'担保', u'股份', u'银行', u'抵押', u'董事长', u'副总经 理', u'外汇', u'总经理', u'利率', u'董事', u'余额', u'总监', u'保险公司', u'并购', u'股东', u'利息', u'CEO', u'国有企业', u'债务', u'资产', u'分行', u'经理', u'负债', u'股市', u'信用', u'总额', u'总裁', u'国有资产', u'股价', u'投资者', u'资本', u'汇率', u'数额', u'金额', u'投资', u'首席', u'账户', u'国有', u'货币', u'邮政', u'董事会', u'创始人', u'改制', u'支行', u'主管', u'出资', u'破产', u'重组', u'财产', u'披露', u'固定资产', u'收购', u'上涨', u'财富', u'商业', u'现金']

[u'教师', u'教学', u'教研', u'素质教育', u'课堂教学', u'教学质量', u'教学改革', u'该校', u'学生', u'基础教育', u'我校', u'高职', u'德育', u'教职工', u'职业教育', u'高等院校', u'办学', u'教师队伍', u'院校', u'师资', u'在校生', u'考生', u'教育', u'高等学校', u'同学们', u'专任教师', u'毕业生', u'班级', u'班主任', u'报考', u'人才培养', u'高等教育', u'在校', u'孩子们', u'高校', u'老师', u'学员', u'招生', u'全 校', u'育人', u'学子', u'大学生', u'教学班', u'校长', u'入学', u'家长', u'在校学生', u'录取', u'师生', u'青少年', u'同学', u'全日制', u'心理健康', u'招收', u'课堂', u'报名', u'青年', u'校友', u'先生', u' 义务教育', u'校园', u'校', u'成绩', u'女士']

[u'慢性', u'疾病', u'糖尿病', u'急性', u'病变', u'肿瘤', u'高血压', u'治疗', u'炎症', u'并发症', u'心脏病', u'癌症', u'水肿', u'患者', u'病理', u'病因', u'腹泻', u'呕吐', u'咳嗽', u'症状', u'便秘', u'临床表现', u'出血', u'病人', u'症', u'贫血', u'手术', u'头痛', u'病例', u'疗效', u'康复', u'发病', u'综合征', u'疗法', u'诊断', u'发作', u'切除', u'病情', u'功效', u'损伤', u'病', u'传染病', u'鉴别', u'感染', u'患', u'病毒', u'瘤', u'术', u'囊', u'畸形', u'部位', u'发热', u'伴有', u'中毒']

[u'魔王', u'恶魔', u'妖', u'魔', u'冥', u'邪恶', u'吸血鬼', u'怪物', u'死神', u'鬼', u'幽灵', u'猎人', u'幻', u'魔兽', u'仙', u'黑暗', u'诅咒', u'封印', u'僵尸', u'毁灭', u'舰', u'舰队', u'玄', u'咒', u' 精灵', u'幽', u'兽', u'艘', u'魔鬼', u'凶', u'骑士', u'BOSS', u'地狱', u'复活', u'国王', u'公主', u'女王', u'神仙', u'仙人', u'诀', u'召唤', u'尸', u'逍遥', u'魔法', u'怪', u'法术', u'神', u'王子', u'女神', u'魔力', u'灵', u'勇士', u'魂', u'邪', u'天使', u'怪兽', u'化身', u'尸体', u'武士', u'海盗', u'恶', u'戒', u'预言', u'死者', u'风流', u'光明', u'副本', u'变身', u'丹', u'杀手', u'正义', u'武功', u'荒']
</blockquote>
