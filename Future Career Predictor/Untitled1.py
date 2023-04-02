#!/usr/bin/env python
# coding: utf-8

# In[508]:


import pandas as pd
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt


# In[509]:


dataset = pd.read_csv("data.csv")


# In[510]:


#dataset.drop(labels=range(10000, 20000), axis=0,inplace=True)


# In[511]:


dataset.drop(['Salary Range Expected','Job/Higher Studies?','Gentle or Tuff behaviour?','Salary/work'], axis=1,inplace=True)


# In[512]:


dataset['Suggested Job Role'] = dataset['Suggested Job Role'].replace(['Database Manager'], 'Database Developer')


# In[513]:


dataset['Suggested Job Role'] = dataset['Suggested Job Role'].replace(['Network Security Engineer'], 'Network Engineer')


# In[514]:


dataset['Suggested Job Role'] = dataset['Suggested Job Role'].replace(['Software Systems Engineer','Solutions Architect'], 'Software Developer')


# In[515]:


dataset['Suggested Job Role'] = dataset['Suggested Job Role'].replace(['Business Systems Analyst', 'Business Intelligence Analyst'], 'Business Analyst')


# In[516]:


#dataset['Suggested Job Role'] = dataset['Suggested Job Role'].replace(['Technical Services/Help Desk/Tech Support'], 'Technical Support')


# In[ ]:





# In[ ]:





# In[517]:


dataset.iloc[:,0:15]


# In[518]:


dataset.iloc[:,15:30]


# In[519]:


dataset.iloc[0:10000,34:35].drop_duplicates()


# In[520]:


categorical_col = dataset[['self-learning capability?', 'Extra-courses did','reading and writing skills', 'memory capability score', 
                      'Taken inputs from seniors or elders', 'Management or Technical', 'hard/smart worker', 'worked in teams ever?', 
                      'Introvert', 'interested career area ']]
for i in categorical_col:
    print(dataset[i].value_counts(), end="\n\n")


# In[521]:


dataset.isnull().sum(axis=0)


# In[542]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[543]:


sns.set(rc={'figure.figsize':(100,10)})
sns.countplot(x = dataset["Suggested Job Role"])


# In[541]:


print(dataset["Interested Type of Books"].value_counts())


# In[550]:


# Figure Size and DPI(Dots per pixels)
fig, ax = plt.subplots(figsize=(13,15))

# Horizontal Bar Plot
title_cnt=dataset["Interested Type of Books"].value_counts().sort_values(ascending=True).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1],edgecolor='black', color=sns.color_palette('gist_rainbow',len(title_cnt)))



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)


# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Interested Books',weight='bold',fontsize=20)
ax.set_xlabel('Count', weight='bold')

# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# Show Plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[524]:


cols = dataset[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert",'can work long time before system?','talenttests taken?','olympiads','interested in games','In a Realtionship?']]
for i in cols:
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    dataset = dataset.replace(cleanup_nums)


# In[525]:


mycol = dataset[["reading and writing skills", "memory capability score"]]
for i in mycol:
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    dataset = dataset.replace(cleanup_nums)

category_cols = dataset[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                    'Interested Type of Books']]
for i in category_cols:
    dataset[i] = dataset[i].astype('category')
    dataset[i + "_code"] = dataset[i].cat.codes

print("\n\nList of Categorical features: \n" , dataset.select_dtypes(include=['object']).columns.tolist())


# In[526]:


dataset.head()


# In[527]:


dataset = pd.get_dummies(dataset, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])


# In[528]:


dataset.head()


# In[529]:


print("List of Numerical Columns: \n" , dataset.select_dtypes(include=np.number).columns.tolist())


# In[530]:


major=pd.DataFrame(dataset,columns=['Acedamic percentage in Operating Systems', 'percentage in Algorithms', 'Percentage in Programming Concepts', 'Percentage in Software Engineering', 'Logical quotient rating', 'hackathons', 'coding skills rating', 'public speaking points', 'can work long time before system?', 'self-learning capability?', 'Extra-courses did', 'talenttests taken?', 'olympiads', 'reading and writing skills', 'memory capability score', 'Taken inputs from seniors or elders', 'interested in games', 'In a Realtionship?', 'worked in teams ever?', 'Introvert', 'certifications_code', 'workshops_code', 'Interested subjects_code', 'interested career area _code', 'Type of company want to settle in?_code', 'Interested Type of Books_code', 'A_Management', 'A_Technical', 'B_hard worker', 'B_smart worker'])


# In[531]:


major.head()


# In[532]:


y=pd.DataFrame(dataset,columns=["Suggested Job Role"])


# In[533]:


y.head()


# In[534]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[535]:


x_train,x_test,y_train,y_test=train_test_split(major,y,test_size=0.4,random_state=42)


# In[536]:


svm = SVC()
svm.fit(x_train, y_train)
pred = svm.predict(x_test)
svm_accuracy = accuracy_score(y_test,pred)


#
print(svm_accuracy*10)


# In[537]:


dtree = DecisionTreeClassifier(random_state=1)
dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("confusion matrics=",cm)
print("  ")
print("accuracy=",accuracy*10)


# In[538]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(x_train, y_train)

rfc_pred = rfc.predict(x_test)
cm = classification_report(y_test,rfc_pred)
accuracy = accuracy_score(y_test,rfc_pred)
print("confusion matrics=",cm)
print("  ")
print("accuracy=",accuracy*10)


# In[539]:


features = [[34,43,56,32,65,53,65,45,35,5,6,7,8,6,1,1,1,4,4,1,1,1,1,2,3,4,1,1,1,0]]
prediction = rfc.predict(features)
print(prediction)


# features = [[84,53,76,92,65,53,55,75,85,5,6,7,8,6,0,0,0,4,4,0,0,0,0,2,3,4,0,0,0,0]]
# prediction = rfc.predict(features)
# print(prediction)

# In[540]:



features = [[84,53,76,92,65,53,55,75,85,12,7,7,8,6,1,0,0,3,3,1,0,1,0,4,3,5,1,1,0,0]]
prediction = rfc.predict(features)
print(prediction)


# In[548]:


features = [[84,54,76,21,65,53,55,75,32,12,3,7,8,6,1,1,0,3,3,1,0,1,0,4,3,5,1,1,0,0]]
prediction = rfc.predict(features)
print(prediction)


# In[ ]:




