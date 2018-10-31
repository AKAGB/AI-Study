class VariableElimination:
    @staticmethod
    def inference(factorList, queryVariables, 
    orderedListOfHiddenVariables, evidenceList):
        for ev in evidenceList:
            #Your code here
            for f in factorList[:]:
                if ev in f.varList:
                    new_f = f.restrict(ev, str(evidenceList[ev]))
                    factorList.remove(f)
                    if new_f != 0:
                        factorList.append(new_f)

        for var in orderedListOfHiddenVariables:
            
            #Your code here
            fs = []
            for f in factorList[:]:
                if var in f.varList:
                    fs.append(f)
                    factorList.remove(f)

            if len(fs) != 0:
                new_f = fs[0]
                for each in fs[1:]:
                    new_f = new_f.multiply(each)
                new_f = new_f.sumout(var)
                if new_f != 0:
                    factorList.append(new_f)
            

        print "RESULT:"
        res = factorList[0]
        for factor in factorList[1:]:
            res = res.multiply(factor)
        total = sum(res.cpt.values())
        res.cpt = {k: v/total for k, v in res.cpt.items()}
        res.printInf()
    @staticmethod
    def printFactors(factorList):
        for factor in factorList:
            factor.printInf()
class Util:
    @staticmethod
    def to_binary(num, len):
        return format(num, '0' + str(len) + 'b')
class Node:
    def __init__(self, name, var_list):
        self.name = name
        self.varList = var_list
        self.cpt = {}
    def setCpt(self, cpt):
        self.cpt = cpt
    def printInf(self):
        print "Name = " + self.name
        print " vars " + str(self.varList)
        for key in self.cpt:
            print "   key: " + key + " val : " + str(self.cpt[key])
        print ""
    def multiply(self, factor):
        """function that multiplies with another factor"""
        #Your code here
        s1 = set(self.varList)
        s2 = set(factor.varList)
        new_var_list = list(s1.union(s2))
        
        index1 = []
        index2 = []
        l1 = len(self.varList)
        l2 = len(factor.varList)
        l3 = len(new_var_list)
        for i in range(l1):
            index1.append(new_var_list.index(self.varList[i]))
        for i in range(l2):
            index2.append(new_var_list.index(factor.varList[i]))


        new_cpt = {}
        for idx in range(2**l3):
            new_idx = Util.to_binary(idx, l3)
            idx1 = ''
            idx2 = ''
            for each in index1:
                idx1 += new_idx[each]
            for each in index2:
                idx2 += new_idx[each]
            new_cpt[new_idx] = self.cpt[idx1] * factor.cpt[idx2]
            
        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        return new_node
    def sumout(self, variable):
        """function that sums out a variable given a factor"""
        #Your code here

        o_i = self.varList.index(variable)
        new_var_list = self.varList[:o_i] + self.varList[o_i+1:]
        new_cpt = {}
        lim = len(new_var_list)
        length = 2**lim
        if lim == 0:
            return 0
        for i in range(length):
            index = Util.to_binary(i, lim)
            # print index
            i1 = index[:o_i] + '0' + index[o_i:]
            i2 = index[:o_i] + '1' + index[o_i:]
            new_cpt[index] = self.cpt[i1] + self.cpt[i2]

        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        return new_node
    def restrict(self, variable, value):
        """function that restricts a variable to some value 
        in a given factor"""
        #Your code here
        o_i = self.varList.index(variable)
        new_var_list = self.varList[:o_i] + self.varList[o_i+1:]
        new_cpt = {}
        lim = len(new_var_list)
        length = 2**lim
        if lim == 0:
            return 0
        for i in range(length):
            index = Util.to_binary(i, lim)
            old_index = index[:o_i] + value + index[o_i:]
            new_cpt[index] = self.cpt[old_index]

        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        return new_node
# create nodes for Bayes Net
B = Node("B", ["B"])
E = Node("E", ["E"])
A = Node("A", ["A", "B","E"])
J = Node("J", ["J", "A"])
M = Node("M", ["M", "A"])

# Generate cpt for each node
B.setCpt({'0': 0.999, '1': 0.001})
E.setCpt({'0': 0.998, '1': 0.002})
A.setCpt({'111': 0.95, '011': 0.05, '110':0.94,'010':0.06,
'101':0.29,'001':0.71,'100':0.001,'000':0.999})
J.setCpt({'11': 0.9, '01': 0.1, '10': 0.05, '00': 0.95})
M.setCpt({'11': 0.7, '01': 0.3, '10': 0.01, '00': 0.99})

# test = A.multiply(E)
# print test.varList
# print test.cpt

print "P(A) **********************"
VariableElimination.inference([B,E,A,J,M], ['A'], ['B', 'E', 'J','M'], {})

print "P(J&&~M) **********************"
VariableElimination.inference([B,E,A,J,M], ['J', 'M'], ['A', 'B', 'E'], {})

print "P(A | J&&~M) **********************"
VariableElimination.inference([B,E,A,J,M], ['A'], ['B', 'E'], {'J':1, 'M': 0})

print "P(B | A) **********************"
VariableElimination.inference([B,E,A,J,M], ['B'], ['E', 'J', 'M'], {'A': 1} )

print "P(B | J~M) **********************"
VariableElimination.inference([B,E,A,J,M], ['B'], ['E','A'], {'J':1,'M':0})

print "P(J&&~M | ~B) **********************"
VariableElimination.inference([B,E,A,J,M], ['J', 'M'], ['A', 'E'], {'B': 0} )