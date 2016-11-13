import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Import the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Thanks for https://www.kaggle.com/arthurlu/titanic/exploratory-tutorial-titanic/notebook
# Thanks for https://www.kaggle.com/davidfumo/titanic/exploratory-tutorial-titanic-disaster/discussion
# For Helping with the Analysis

# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print "-------Training Basic Statistics-------"
print ("Dimension: {}".format(train.shape))
print train.describe()
print "-------Test Basic Statistics-------"
print ("Dimension: {}".format(test.shape))
print test.describe()

### GRAPHS ###
## Distribution Graphs ## Investigate
plt.rc('font', size=10)
fig = plt.figure(figsize=(20, 10))
alpha = 0.5
train_color = '#e66101'
test_color = '#5e3c99'

ax = plt.subplot2grid((2,3), (0,0), colspan=1)
train.Survived.value_counts().plot(kind='bar', color=train_color, label='train', alpha=alpha)
#test.Survived.plot(kind ='bar', color=test_color,label='test', alpha=alpha)
ax.set_xlabel('Survived')
ax.set_title("Survived Distribution" )
plt.legend(loc='best')

ax1 = plt.subplot2grid((2,3), (0,1))
train.Age.plot.hist(color=train_color, label='train', alpha=alpha)
test.Age.plot.hist(color=test_color, label='test', alpha=alpha)
ax1.set_xlabel('Age')
ax1.set_title("Age Distribution" )
plt.legend(loc='best')

ax2 = plt.subplot2grid((2,3), (0,2), colspan=1)
train.Fare.plot.hist(bins = 30, color=train_color, label='train', alpha=alpha)
test.Fare.plot.hist(bins =30, color=test_color,label='test', alpha=alpha)
ax2.set_xlabel('Fare')
ax2.set_title("Fare Distribution" )
plt.legend(loc='best')

ax3 = plt.subplot2grid((2,3), (1,0))
train.Pclass.value_counts().plot(kind='bar', color=train_color, label='train', alpha=alpha)
test.Pclass.value_counts().plot(kind='bar', color=test_color,label='test', alpha=alpha)
ax3.set_ylabel('Frequency')
ax3.set_xlabel('Pclass')
ax3.set_title("Pclass Distribution" )
plt.legend(loc='best')

ax4 = plt.subplot2grid((2,3), (1,1))
train.Sex.value_counts().plot(kind='bar', color=train_color, label='train', alpha=alpha)
test.Sex.value_counts().plot(kind='bar', color=test_color, label='test', alpha=alpha)
ax4.set_ylabel('Frequency')
ax4.set_xlabel('Sex')
ax4.set_title("What's the distribution of Sex?" )
plt.legend(loc='best')

ax5 = plt.subplot2grid((2,3), (1,2))
train.Embarked.value_counts().plot(kind='bar', color=train_color, label='train', alpha=alpha)
test.Embarked.value_counts().plot(kind='bar', color=test_color,label='test', alpha=alpha)
ax5.set_ylabel('Frequency')
ax5.set_xlabel('Embarked')
ax5.set_title("What's the distribution of Embarked?" )
plt.legend(loc='best')

plt.suptitle("Distribution Graphs of Train & Test",size=20)
#plt.tight_layout()

## Training Graphs ## Investigate
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3,4)
plt.rc('font', size=10)
fig = plt.figure(figsize=(20, 10))
alpha = 0.5
f_color = '#d01c8b'
m_color = '#4dac26'
d_color = '#ca0020'
s_color = '#0571b0'
#### Age Graph
ax = plt.subplot2grid((3,4), (0,0), colspan=2)
train[train.Survived==0].Age.plot(kind='density', color=d_color, label='Died', alpha=alpha)
train[train.Survived==1].Age.plot(kind='density', color=s_color, label='Survived', alpha=alpha)
ax.set_xlim([0, 100])
plt.ylabel('Frequency')
plt.title('Survival based on Age Distribution')
plt.legend(loc='best')
#### Age and Gender Graph
ax2 = plt.subplot2grid((3,4), (0,2), colspan=1,sharey = ax)
train[(train.Survived==0)&(train.Sex=='female')&(~train.Age.isnull())].Age.plot(kind='density', color=d_color, label='Died', alpha=alpha)
train[(train.Survived==1)&(train.Sex=='female')&(~train.Age.isnull())].Age.plot(kind='density', color=s_color, label='Survived', alpha=alpha)
ax2.set_xlim([0, 100])
plt.ylabel('Frequency')
plt.title('Survival base on Age & Female Distribution')
plt.legend(loc='best')

ax3 = plt.subplot2grid((3,4), (0,3), colspan=1,sharey = ax)
train[(train.Survived==0)&(train.Sex=='male')&(~train.Age.isnull())].Age.plot(kind='density', color=d_color, label='Died', alpha=alpha)
train[(train.Survived==1)&(train.Sex=='male')&(~train.Age.isnull())].Age.plot(kind='density', color=s_color, label='Survived', alpha=alpha)
ax3.set_xlim([0, 100])
plt.ylabel('Frequency')
plt.title('Survival base on Age & Male Distribution')
plt.legend(loc='best')


#### Graph by Gender
df_male = train[train.Sex=='male'].Survived.value_counts().sort_index()
df_female = train[train.Sex=='female'].Survived.value_counts().sort_index()


ax4 = plt.subplot2grid((3,4), (1,0),colspan=2)
df_female.plot(kind='barh', color=f_color, label='Female', alpha=alpha)
ax4.set_xlabel('Rate')
ax4.set_yticklabels(['Died', 'Survived'])
ax4.set_title("Female Survival Rate" )
plt.legend(loc='best')

ax5 = plt.subplot2grid((3,4), (1,2),colspan=2,sharey=ax4)
(df_male/train[train.Sex=='male'].shape[0]).plot(kind='barh', color=m_color,label='Male', alpha=alpha)
ax5.set_xlabel('Rate')
ax5.set_yticklabels(['Died', 'Survived'])
ax5.set_title("Male Survival Rate" )
plt.legend(loc='best')

df_male3 = train[(train.Sex=='male')&(train.Pclass==3)].Survived.value_counts().sort_index()
df_male0 = train[(train.Sex=='male')&(train.Pclass<3)].Survived.value_counts().sort_index()
df_female3 = train[(train.Sex=='female')&(train.Pclass==3)].Survived.value_counts().sort_index()
df_female0 = train[(train.Sex=='female')&(train.Pclass<3)].Survived.value_counts().sort_index()

ax6 = plt.subplot2grid((3,4), (2,0))
(df_female0/train[(train.Sex=='female')&(train.Pclass<3)].shape[0]).plot(kind='barh', color=f_color,label='Female', alpha=alpha)
ax6.set_xlabel('Rate')
ax6.set_yticklabels(['Died', 'Survived'])
ax6.set_title("Female Not 3rd Class Survival Rate" )
plt.legend(loc='best')

ax7 = plt.subplot2grid((3,4), (2,1),sharey=ax6)
(df_female3/train[(train.Sex=='female')&(train.Pclass==3)].shape[0]).plot(kind='barh', color=f_color,label='Female', alpha=alpha)
ax7.set_xlabel('Rate')
ax7.set_title("Female 3rd Class Survival Rate" )
plt.legend(loc='best')

ax8 = plt.subplot2grid((3,4), (2,2),sharey=ax6)
(df_male0/train[(train.Sex=='male')&(train.Pclass<3)].shape[0]).plot(kind='barh', color=m_color,label='Male', alpha=alpha)
ax8.set_xlabel('Rate')
ax8.set_title("Male Not 3rd Class Survival Rate" )
plt.legend(loc='best')

ax9 = plt.subplot2grid((3,4), (2,3),sharey=ax6)
(df_male3/train[(train.Sex=='male')&(train.Pclass==3)].shape[0]).plot(kind='barh', color=m_color,label='Male', alpha=alpha)
ax9.set_xlabel('Rate')
ax9.set_title("Male 3rd Class Survival Rate" )
plt.legend(loc='best')



plt.suptitle("Titantic Training Data",size =20)

#ax2 = plt.subplot2grid((3,4), (0,2), colspan=1)
#ax2 = plt.subplot(gs[0,2])
#train.boxplot(column='Age',by ='Survived')
#train[train.Survived==1].Age.plot(kind='density', color=s_color, label='Survived', alpha=alpha)
#plt.ylabel('Age')
#plt.title('Training Age Distribution')
#plt.legend(loc='best')



