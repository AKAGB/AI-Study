import pandas as pd
from math import e, pi

df = pd.read_csv('adult.data', header=None)

grouped = df.groupby(14)

class1_data = grouped.get_group(' <=50K')
class2_data = grouped.get_group(' >50K')
class1_means = class1_data.mean()
class1_vars = class1_data.var()
class2_means = class2_data.mean()
class2_vars = class2_data.var()

cont = set(class1_means.keys())

conf = {}
def gaus(miu, sigma):
    return (lambda value: (e**(-((value-miu)**2)/(2 * sigma))) / ((2*pi*sigma)**0.5))
def gen_continues_p():
    for i in cont:
        miu1 = class1_means[i]
        sigma1 = class1_vars[i]
        miu2 = class2_means[i]
        sigma2 = class2_vars[i]
        conf[(i,'<=50K')] = gaus(miu1, sigma1)
        conf[(i, '>50K')] = gaus(miu2, sigma2)

gen_continues_p()

l1 = len(class1_data)
l2 = len(class2_data)
def discrete_p(index, value, cls):
    if cls == '<=50K':
        group = class1_data.groupby(index).groups
        length = l1
    elif cls == '>50K':
        group = class2_data.groupby(index).groups
        length = l2
    if value in group:
        return len(group[value]) / length
    return 0

# Remmember the p
disP = {}
def calc_cond(X, yi):
    l = len(X)
    result = 1.0
    for i in range(l):
        if i in cont:
            result *= conf[i, yi](X[i])
        else:
            if (X[i], yi) not in disP:
                tp = discrete_p(i, X[i], yi)
                disP[(X[i], yi)] = tp                
            result *= disP[(X[i], yi)]
    return result


p_y1 = len(class1_data) / len(df)
p_y2 = len(class2_data) / len(df)

def classify(X):
    result = (calc_cond(X, '<=50K')*p_y1) < (calc_cond(X, '>50K')*p_y2)
    if result:
        return '>50K'
    return '<=50K'

test_data = pd.read_csv('adult.test', header=None)

acc=0
for row in test_data.itertuples():
    x = row[1:-1]
    result = row[-1][1:-1]
    if classify(x)==result:
        acc += 1
acc = acc / len(test_data)
print(acc)
