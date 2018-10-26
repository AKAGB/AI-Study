
# coding: utf-8

# In[1]:

from pomegranate import *


# In[2]:

burglary = DiscreteDistribution({'T': 0.001, 'F': 0.999})
earthquake = DiscreteDistribution({'T': 0.002, 'F': 0.998})


# In[3]:

alarm = ConditionalProbabilityTable([
        ['T', 'T', 'T', 0.95],
        ['T', 'T', 'F', 0.05],
        ['T', 'F', 'T', 0.94],
        ['T', 'F', 'F', 0.06],
        ['F', 'T', 'T', 0.29],
        ['F', 'T', 'F', 0.71],
        ['F', 'F', 'T', 0.001],
        ['F', 'F', 'F', 0.999]
    ], [burglary, earthquake])

johnCalls = ConditionalProbabilityTable([
        ['T', 'T', 0.9],
        ['T', 'F', 0.1],
        ['F', 'T', 0.05],
        ['F', 'F', 0.95]
    ], [alarm])

marryCalls = ConditionalProbabilityTable([
        ['T', 'T', 0.7],
        ['T', 'F', 0.3],
        ['F', 'T', 0.01],
        ['F', 'F', 0.99]
    ], [alarm])


# In[4]:

s1 = State(burglary, name='burglary')
s2 = State(earthquake, name='earthquake')
s3 = State(alarm, name='alarm')
s4 = State(johnCalls, name='johnCalls')
s5 = State(marryCalls, name='marryCalls')


# In[5]:

model = BayesianNetwork('Burglary Network')


# In[6]:

model.add_states(s1, s2, s3, s4, s5)


# In[7]:

model.add_transition(s1, s3)
model.add_transition(s2, s3)
model.add_transition(s3, s4)
model.add_transition(s3, s5)


# In[8]:

model.bake()


# In[10]:

marginals = model.predict_proba({})


# In[17]:

print 'P(Alarm) ='
print marginals[2].parameters[0]['T'],'\n'


# In[373]:

j = marginals[3].parameters[0]['T']
nm = model.predict_proba({'johnCalls':'T'})[4].parameters[0]['F']
jm = j * nm
print 'P(J&&~M) ='
print jm, '\n'


# In[35]:

jm_cond = model.predict_proba({'johnCalls':'T', 'marryCalls': 'F'})
print 'P(A | J&&!M) ='
print jm_cond[2].parameters[0]['T'], '\n'


# In[34]:

print 'P(B|A) ='
print model.predict_proba({'alarm':'T'})[0].parameters[0]['T'], '\n'


# In[372]:

print 'P(B | J&&~M) ='
bjm = jm_cond[0].parameters[0]['T']
print bjm, '\n'


# In[382]:

p_b = marginals[0].parameters[0]['F']
jmb = (1-bjm)*jm/p_b
print 'P(J&&~M | ~B)'
print jmb, '\n'
