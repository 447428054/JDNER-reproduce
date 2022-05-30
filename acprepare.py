# -*- coding: utf-8 -*-
# @Author  : JMXGODLZZ
# @Time    : 2022/01/14 9:45
import ahocorasick
import pickle

class CNAC():
    def __init__(self, most=True):
        self.most = most

    def createAC(self, filepath):
        fr = open(filepath, mode='rb')
        entities = pickle.load(fr)
        self.ac = ahocorasick.Automaton()
        for et, tpdict in entities.items():
            if len(et) == 1:
                continue
            tpdict = sorted(tpdict.items(), key=lambda item: item[1], reverse=self.most)
            tp = tpdict[0][0]
            self.ac.add_word(et, [tp, et])  # 暂未处理交叉实体
        self.ac.make_automaton()

    def acsearch(self, sentence):
        items = self.ac.iter(sentence)
        return items

class CNACALL():
    def __init__(self, most=True):
        self.most = most

    def createAC(self, filepath):
        fr = open(filepath, mode='rb')
        entities = pickle.load(fr)
        self.ac = ahocorasick.Automaton()
        for et, tpdict in entities.items():
            if len(et) == 1:
                continue
            tpdict = sorted(tpdict.items(), key=lambda item: item[1], reverse=self.most)
            tplst = [item[0] for item in tpdict]
            # tp = tpdict[0][0]
            self.ac.add_word(et, [tplst, et])  # 暂未处理交叉实体
        self.ac.make_automaton()

    def acsearch(self, sentence):
        items = self.ac.iter(sentence)
        return items

class TrieNode():
    def __init__(self):
        self.child = {}
        self.failto = None
        self.is_word = False
        '''
        下面节点值可以根据具体场景进行赋值
        '''
        self.str_ = ''
        self.type = ''

class AhoCorasickAutomation:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def buildTrieTree(self, wordlst):
        for inf in wordlst:
            tp, word = inf
            cur = self.root
            chars = word.split(' ')
            for i, c in enumerate(chars):
                if c not in cur.child:
                    cur.child[c] = TrieNode()
                ps = cur.str_
                cur = cur.child[c]
                if i == 0:
                    cur.str_ = ps + c
                else:
                    cur.str_ = ps + ' ' + c
            cur.is_word = True
            cur.type = tp

    def build_AC_from_Trie(self):
        queue = []
        for child in self.root.child:
            self.root.child[child].failto = self.root
            queue.append(self.root.child[child])

        while len(queue) > 0:
            cur = queue.pop(0)
            for child in cur.child.keys():
                failto = cur.failto
                while True:
                    if failto == None:
                        cur.child[child].failto = self.root
                        break
                    if child in failto.child:
                        cur.child[child].failto = failto.child[child]
                        break
                    else:
                        failto = failto.failto
                queue.append(cur.child[child])

    def ac_search(self, str_):
        cur = self.root
        str_ = str_.split(' ')
        result = []
        i = 0
        n = len(str_)
        while i < n:
            c = str_[i]
            if c in cur.child:
                cur = cur.child[c]
                if cur.is_word:
                    temp = cur.str_
                    result.append([i, [cur.type, temp]])

                '''
                处理所有其他长度公共字串
                '''
                fl = cur.failto
                while fl:
                    if fl.is_word:
                        temp = fl.str_
                        result.append([i, [cur.failto.type, temp]])
                    fl = fl.failto
                i += 1

            else:
                cur = cur.failto
                if cur == None:
                    cur = self.root
                    i += 1


        return result

class ENAC():
    def __init__(self, most=True):
        self.most = most

    def createAC(self, filepath):
        fr = open(filepath, mode='rb')
        entities = pickle.load(fr)
        self.ac = AhoCorasickAutomation()
        wordlst = []
        for et, tpdict in entities.items():
            if len(et) == 1:
                continue
            tpdict = sorted(tpdict.items(), key=lambda item: item[1], reverse=self.most)
            tp = tpdict[0][0]
            wordlst.append([tp, et])  # 暂未处理交叉实体
        self.ac.buildTrieTree(wordlst)
        self.ac.build_AC_from_Trie()

    def acsearch(self, sentence):
        items = self.ac.ac_search(sentence)
        return items

# ca = CNAC()
# ca.createAC('cn_train.pickle')
# res = ca.acsearch('我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。')
# for x in res:
#     print(x)
#
# ca = ENAC()
# ca.createAC('en_train.pickle')
# res = ca.acsearch('IL - 2 gene bad doctors EU rejects German call to boycott British lamb .')
# for x in res:
#     print(x)