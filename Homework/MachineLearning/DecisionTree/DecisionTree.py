import pandas as pd
from math import log2
import json

class Node:
    def __init__(self):
        self.attr = ''
        self.children = {}
        self.isLeaf = False
        self.isContinious = False
        self.divider = 0  # record the divider of continious attributes
        
    def setAttr(self, attr):
        self.attr = attr
        
    def setChild(self, attr, tree):
        self.children[attr] = tree
        
    def setDivider(self, div):
        self.divider = div
        self.isContinious = True


class DecisionTree:
    def __init__(self, DataSet, cont):
        self.__DataSet = DataSet
        self.__attrs =  DecisionTree.getAttrs(DataSet)
        self.cont = cont        # continious attributes
        # Record the domain of each discrete attr
        self.domains = {}
        for each in self.__attrs:
            if each not in self.cont:
                self.domains[each] = list(DataSet.groupby(each).groups.keys())
                
        self.root = None
        
    def setDataSet(self, DataSet):
        self.__DataSet = DataSet
    
    def getRoot(self):
        """
        Build a Decision Tree and return the root
        """
        self.root = self.TreeGenerate(self.__DataSet, self.__attrs[:])
        
    def TreeGenerate(self, DataSet, AttributeSet):
        """
        param:
            DataSet[DataFrame] - Data Set used to build tree
            Attributes[list] - Attributes of input data
        return:
            Node - The root of Tree
        """
        node = Node()
        # Only one label
        if DecisionTree.checkLabels(DataSet):
            node.setAttr(DecisionTree.getDomain(DataSet, 'classes')[0])
            node.isLeaf = True
            return node
        # Attrs is empty set or all input data has same attr,
        # then return the most labels
        if len(AttributeSet) == 0 or \
        DecisionTree.checkSameAttrs(DataSet, AttributeSet):
            node.setAttr(DecisionTree.getMostLabel(DataSet))
            node.isLeaf = True
            return node
        bestAttr, divider = DecisionTree.selectAttr(DataSet, AttributeSet, self.cont)
        node.setAttr(bestAttr)
        AttributeSet.remove(bestAttr)
        if bestAttr in self.cont:
            node.setDivider(divider)
            smaller = DataSet[DataSet[bestAttr] < divider]
            bigger = DataSet[DataSet[bestAttr] > divider]
            node.setChild('<div', self.TreeGenerate(smaller, AttributeSet[:]))
            node.setChild('>div', self.TreeGenerate(bigger, AttributeSet[:]))
        else:
            subtree = DataSet.groupby(bestAttr)
            domain = subtree.groups.keys()
            for each in self.domains[bestAttr]:
                if each not in domain:
                    child_node = Node()
                    child_node.setAttr(DecisionTree.getMostLabel(DataSet))
                    child_node.isLeaf = True
                    node.setChild(each, child_node)
                else:
                    Dv = subtree.get_group(each)
                    node.setChild(each, self.TreeGenerate(Dv, AttributeSet[:]))
        return node
    
    @staticmethod
    def selectAttr(DataSet, Attrs, cont):
        """
        select the attribute which has maximum information gain
        If attributes is continious, then return its name and divider
        """
        p, n = getPN(DataSet)
        en_parent = entropy(p/(p+n))
        
        max_gain = -1
        max_attr = Attrs[0]
        max_divider = 0
        for each in Attrs:
            if each not in cont:
                tmp = DecisionTree.gain(DataSet, each, en_parent)
                if tmp > max_gain:
                    max_gain = tmp
                    max_attr = each
            else:
                cgain, divider = DecisionTree.findBestDivider(DataSet, each)
                if cgain != -1:
                    cgain = en_parent - cgain
                if cgain > max_gain:
                    max_gain = cgain
                    max_attr = each
                    max_divider = divider
        
        if max_attr in cont:
            return max_attr, max_divider
        return max_attr, 0
            
    @staticmethod
    def findBestDivider(DataSet, attr):
        """
        param:
            DataSet - pandas DataFrame
            attr - A continious attribute
        return:
            maxGain and Divider
        """
        values = sorted(DecisionTree.getDomain(DataSet, attr))
        l = len(values)
        if l <= 1:
            return -1, 0
        min_gain = -1
        m_divider = 0
        al = len(DataSet)
        step = max(l // 10, 1)
        for i in range(0, l-1, step):
            div = (values[i] + values[i+1]) / 2
            # calculate gain of this divider
            smaller = DataSet[DataSet[attr] < div]
            bigger = DataSet[DataSet[attr] > div]
            sp, sn = getPN(smaller)
            bp, bn = getPN(bigger)
            sw = (sp + sn) / al
            bw = 1 - sw
            cgain = sw * entropy(sp/(sp+sn)) + bw * entropy(bp/(bp+bn))
            if min_gain == -1 or min_gain > cgain:
                min_gain = cgain
                m_divider = div
        return min_gain, m_divider
    
    @staticmethod
    def getMostLabel(DataSet):
        tmp = DataSet.groupby('classes').groups
        l1, l2 = tmp.keys()
        return (l1 if len(tmp[l1]) > len(tmp[l2]) else l2)
        
    @staticmethod
    def getAttrs(DataSet):
        return list(DataSet.columns)[:-1]
        
    @staticmethod
    def getDomain(DataSet, attr):
        return list(DataSet.groupby(attr).groups.keys())

    @staticmethod
    def checkLabels(DataSet):
        return len(DecisionTree.getDomain(DataSet, 'classes')) == 1
    
    @staticmethod
    def checkSameAttrs(DataSet, Attrs):
        """
        Check Whether all tuples have same value on attrs
        """
        for each in Attrs:
            if len(DataSet.groupby(each).groups.keys()) != 1:
                return False
        return True
    
    @staticmethod
    def gain(DataSet, attr, en_parent):
        childs = DataSet.groupby(attr)
        groups = childs.groups.keys()
        for value in groups:
            # calclute the subtree
            subtree = childs.get_group(value)
            weight = len(subtree) / len(DataSet)
            p, n = getPN(subtree)
            en = entropy(p/(p+n))
            en_parent -= weight * en
        return en_parent
        
    @staticmethod
    def print_tree(root):
        if root.isLeaf:
            return root.attr
        result = {'Attr': root.attr, 'Child': {}}
        for each in root.children.keys():
            result['Child'][each] = DecisionTree.print_tree(root.children[each])
        return result

    @staticmethod
    def test(root, X):
        if root.isLeaf:
            return root.attr
        if root.isContinious:
            if X[root.attr] < root.divider:
                return DecisionTree.test(root.children['<div'], X)
            return DecisionTree.test(root.children['>div'], X)
        return DecisionTree.test(root.children[X[root.attr]], X)


def entropy(p):
    if p == 1 or p == 0:
        return 0
    return -(p * log2(p) + (1-p)*log2(1-p))

def getPN(DataSet):
    groups = DataSet.groupby('classes').groups
    ks = list(groups.keys())
    if len(ks) == 2:
        p, n = len(groups[ks[0]]), len(groups[ks[1]])
    else:
        p, n = len(groups[ks[0]]), 0
    return p, n

if __name__ == '__main__':
    cont = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    df = pd.read_csv('adult.data', header=None, names=[
            'age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain',
            'capital-loss', 'hours-per-week', 'native-country',
            'classes'
        ])
    tree = DecisionTree(df, cont)
    tree.getRoot()
    dic = tree.print_tree(tree.root)

    with open('result.json', 'w') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)

    print('test:')
    testdf = pd.read_csv('adult.test', header=None, names=[
            'age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain',
            'capital-loss', 'hours-per-week', 'native-country',
            'classes'
        ])
    l = len(testdf)
    cnt = 0
    for i in range(l):
        pre = DecisionTree.test(tree.root, testdf.iloc[i])
        if pre == testdf.iloc[i]['classes'][:-1]:
            cnt += 1
    print(cnt / l)