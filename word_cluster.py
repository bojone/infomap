#! -*- coding: utf-8 -*-
# 基于Word2Vec词向量的词聚类例子

import uniout
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from infomap import infomap


num_words = 10000 # 只保留前10000个词
min_sim = 0.6


word2vec = Word2Vec.load('word2vec_baike')

word_vecs = word2vec.wv.syn0[:num_words]
word_vecs /= (word_vecs**2).sum(axis=1, keepdims=True)**0.5
id2word = word2vec.wv.index2word[:num_words]
word2id = {j: i for i, j in enumerate(id2word)}

links = {}


# 每个词找与它相似度不小于0.6的词（不超过50个），来作为图上的边
for i in tqdm(range(num_words)):
    sims = np.dot(word_vecs, word_vecs[i])
    idxs = sims.argsort()[::-1][1:]
    for j in idxs[:50]:
        if sims[j] >= min_sim:
            links[(i, j)] = float(sims[j])
        else:
            break


infomapWrapper = infomap.Infomap("--two-level --directed")
# 如果重叠社区发现，则只需要：
# infomapWrapper = infomap.Infomap("--two-level --directed --overlapping")


for (i, j), sim in tqdm(links.items()):
    _ = infomapWrapper.addLink(i, j, sim)

infomapWrapper.run()
tree = infomapWrapper.tree


word2class = {}
class2word = {}
for node in tree.leafIter():
    if id2word[node.physIndex] not in word2class:
        word2class[id2word[node.physIndex]] = []
    word2class[id2word[node.physIndex]].append(node.moduleIndex())
    if node.moduleIndex() not in class2word:
        class2word[node.moduleIndex()] = []
    class2word[node.moduleIndex()].append(id2word[node.physIndex])


# len([(k, v) for k, v in word2class.items() if len(v) > 1])


for i in range(100):
    print class2word[i]
    print

