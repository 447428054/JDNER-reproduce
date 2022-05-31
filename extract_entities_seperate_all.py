import json
import os
from tqdm import tqdm
import pickle
import ahocorasick
import json

# mapdict = {'NS': 'LOC', 'NR': 'PER', 'NT': 'ORG'}
# skipl = ['MISC']
dupentitypath = '/home/root1/lizheng/workspace/2022/京东NER/dataSummary/summary_unlabel_6_4/dupentity.pickle'
fr = open(dupentitypath, mode='rb')
dupentities = pickle.load(fr)

labelfr = open('/home/root1/lizheng/workspace/2022/京东NER/dataSummary/summary_unlabel_6_4/jdkg.pickle', mode='rb')

labelentity = pickle.load(labelfr)
class clueprocessor():
    def __init__(self):
        self.entities = {}
        self.types = set()
        # self.trainpath = '/home/root1/lizheng/workspace/2021/lunwen/data/clue/train.json'
        # self.extractself.entities(self.trainpath)
        # self.createAC()

    def extractentities(self, filepath):
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            ljson = json.loads(line)
            text = ljson['text']
            for k, v in ljson['label'].items():
                # if k in skipl:
                #     continue
                # if k in mapdict:
                #     k = mapdict[k]
                self.types.add(k)
                for spans in v.values():
                    for start, end in spans:
                        et = text[start: end + 1]

                        if et in dupentities:
                            continue

                        # if et not in dupentities:
                        #     if et in labelentity:
                        #         infdict = labelentity[et]
                        #         infdict = sorted(infdict.items(), key=lambda item: item[1], reverse=True)
                        #         if infdict[0][1] > 100:
                        #             continue

                        # if et not in dupentities:
                        #     if et in labelentity:
                        #         infdict = labelentity[et]
                        #         infdict = sorted(infdict.items(), key=lambda item: item[1], reverse=True)
                        #         if infdict[0] > 100:
                        #             continue
                        self.entities.setdefault(et, {})
                        self.entities[et].setdefault(k, 0)
                        self.entities[et][k] += 1

    def createAC(self):
        self.ac = ahocorasick.Automaton()
        for tp, ets in self.entities.items():
            for et in ets:
                self.ac.add_word(et, [tp, et]) # 暂未处理交叉实体
        self.ac.make_automaton()

    def acsearch(self, sentence):
        items = self.ac.iter(sentence)
        return items

sparent = './data/JD5-unlabel100W-V5[6_4]/'

jdallcp = clueprocessor()
jd0cp = clueprocessor()
jd1cp = clueprocessor()
jd2cp = clueprocessor()
jd3cp = clueprocessor()
jd4cp = clueprocessor()

jdallcp.extractentities(os.path.join(sparent, 'train_clue.txt'))
jd0cp.extractentities(os.path.join(sparent, 'JDtrain0.txt'))
jd1cp.extractentities(os.path.join(sparent, 'JDtrain1.txt'))
jd2cp.extractentities(os.path.join(sparent, 'JDtrain2.txt'))
jd3cp.extractentities(os.path.join(sparent, 'JDtrain3.txt'))
jd4cp.extractentities(os.path.join(sparent, 'JDtrain4.txt'))

jdallfw = open('KG5_unlabel_nodup/jdall.pickle', mode='wb')
jd0fw = open('KG5_unlabel_nodup/jdtrain0.pickle', mode='wb')
jd1fw = open('KG5_unlabel_nodup/jdtrain1.pickle', mode='wb')
jd2fw = open('KG5_unlabel_nodup/jdtrain2.pickle', mode='wb')
jd3fw = open('KG5_unlabel_nodup/jdtrain3.pickle', mode='wb')
jd4fw = open('KG5_unlabel_nodup/jdtrain4.pickle', mode='wb')


print(sorted(jdallcp.types))


pickle.dump(jdallcp.entities, jdallfw)
pickle.dump(jd0cp.entities, jd0fw)
pickle.dump(jd1cp.entities, jd1fw)
pickle.dump(jd2cp.entities, jd2fw)
pickle.dump(jd3cp.entities, jd3fw)
pickle.dump(jd4cp.entities, jd4fw)