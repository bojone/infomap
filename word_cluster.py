#! -*- coding: utf-8 -*-
# 基于Word2Vec词向量的词聚类例子

from gensim.models import Word2Vec
from tqdm import tqdm
import infomap


word2vec = Word2Vec.load('word2vec_baike')
word2vec.init_sims()
min_sim = 0.6
links = {}
wordset = set(word2vec.wv.index2word[:10000]) # 只保留前10000个词

# 每个词找与它相似度不小于0.6的词（不超过50个），来作为图上的边
for u in tqdm(wordset):
    for v, sim in word2vec.most_similar(u, topn=50):
        if v in wordset:
            if sim >= min_sim:
                links[(u, v)] = sim
            else:
                break


_word2id_mapping = {}
_id2word_mapping = {}


def word2id(w):
    if w in _word2id_mapping:
        return _word2id_mapping[w]
    else:
        _id2word_mapping[len(_word2id_mapping)] = w
        _word2id_mapping[w] = len(_word2id_mapping)
        return _word2id_mapping[w]


def id2word(i):
    return _id2word_mapping.get(i, '')


infomapWrapper = infomap.Infomap("--two-level --directed")
# 如果重叠社区发现，则只需要：
# infomapWrapper = infomap.Infomap("--two-level --directed --overlapping")


for (i, j), sim in tqdm(links.items()):
    _ = infomapWrapper.addLink(word2id(i), word2id(j), sim)

infomapWrapper.run()


word2class = {}
class2word = {}
for node in infomapWrapper.iterTree():
    if node.isLeaf():
        if id2word(node.physicalId) not in word2class:
            word2class[id2word(node.physicalId)] = []
        word2class[id2word(node.physicalId)].append(node.moduleIndex())
        if node.moduleIndex() not in class2word:
            class2word[node.moduleIndex()] = []
        class2word[node.moduleIndex()].append(id2word(node.physicalId))


# len([(k, v) for k, v in word2class.items() if len(v) > 1])

for i in range(100):
    print(class2word[i])
    print()
