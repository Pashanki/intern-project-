
# coding: utf-8

# # Author : Pashanki Pandit

# Data Science & Business Analytics Tasks

# Task # 6 - Prediction using Decision Tree Algorithm

# Graduate Rotational Internship Program(GRIP)
# The Sparks Foundation

# In this task I Create the Decision Tree classifier and visualize it graphically.

# # Load the libraries in Python

# In[162]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import missingno as mo
from sklearn.utils import resample


# # Load the data

# In[163]:


df=pd.read_csv("C:/Users/abhijeet/Downloads/Iris (2).csv")


# In[164]:


df.head()


# In[165]:


df.shape


# # Preprocessing the Data

# In[166]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Species"]=le.fit_transform(df.Species)


# In[167]:


df.head()


# # Delete the unwanted coumbs

# In[168]:


df.columns


# In[169]:


df = df.drop(['Id'],axis=1)


# In[170]:


df.head()


# In[171]:


df.info()


# In[172]:


df.describe()


# Insights
# 
# - from above we get that the diffrence between mean and medium if it is more than (<5%) it means it is not normally distributed.
# - from above there is all are normally distributed.

# # Check the correletion

# In[173]:


df.corr()


# # We check the Missing values

# In[174]:


df.isna().sum()  ### all are zero means no missing values


# # Divide categorical and continuos variables in 2 lists

# In[175]:


cat=[]
con=[]
for i in df.columns:
    if (df[i].dtype=="object"):
        cat.append(i)
    else:
        con.append(i)


# In[176]:


cat


# In[177]:


con


# In[178]:


#EDA
import seaborn as sb
import matplotlib.pyplot as plt
for i in df.columns:
    if (df[i].dtype=="object"):
        print(pd.crosstab(df.Species,df[i]))
    else:
        sb.boxplot(df.Species,df[i])
        plt.show()


# Insights
# 
# -We conlued fron above it is the linear data

# # We create the DecisionTreeClassifier

# In[128]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
model=dtc.fit(xtrain,ytrain)
pred=model.predict(xtest)


# In[129]:


from sklearn.tree import export_graphviz
export_graphviz(dtc,out_file="C:/Users/abhijeet/Desktop/dummy_work/e2")

import pydotplus as pdp
graph=pdp.graph_from_dot_file("C:/Users/abhijeet/Desktop/dummy_work/e2")
from IPython.display import Image
Image(graph.create_jpg())


# # You can now feed any new/test data to this classifer and it would be able to predict the right class accordingly. We shown below

# In[130]:


Y=df[['Species']]
X=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=30)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
model=dtc.fit(xtrain,ytrain)
pred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,pred))


# Insights
# 
# - we get the very good accuracy at the randome state=30 

# In[131]:


from sklearn.tree import export_graphviz
export_graphviz(dtc,out_file="C:/Users/abhijeet/Desktop/dummy_work/e2")

import pydotplus as pdp
graph=pdp.graph_from_dot_file("C:/Users/abhijeet/Desktop/dummy_work/e2")
from IPython.display import Image
Image(graph.create_jpg())

