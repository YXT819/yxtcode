'''读取csv文件是通过read_csv这个函数读取的
.loc[ ]第一个参数是行索引，第二个参数是列索引'''

# 其中Survived为决策属性（Y），其值只有两类1或0，代表着获救或未获救；其他11个属性组成特征属性集合（X）分别是：
# PassengerId（乘客ID），Name（姓名），Ticket（船票信息）##用于预测是否Survived意义不大，可以考虑删除这些属性；
# Pclass（乘客等级），Sex（性别），Embarked（登船港口）是类别型数据；
# Age（年龄），SibSp（堂兄弟妹个数），Parch（父母与小孩的个数），是数值型数据，取值情况较多，但可以转换成类别型数据；
# Fare（票价）是数值型数据；Cabin（船舱）则为文本型数据；
# Age（年龄），Cabin（船舱）和Embarked（登船港口）信息存在缺失数据
import pandas as pd #使用pandas读入CSV格式数据
import numpy as np
from pandas import Series,DataFrame


def process_data(data_train:DataFrame,data_test:DataFrame):
    #data_train.info()#注意Age属性、Cabin属性、Embarked属性缺失了很多值！
    print('train_data:','*'*30)
    print(data_train.columns)
    # 比如年龄Age的缺失，我们可以使用平均值填充
    # 看看数据的变化
    data_train=data_train.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)

    #data_train.info()

    data_train.loc[data_train['Sex'] == 'male','Sex'] = 0
    data_train.loc[data_train['Sex'] == 'female','Sex'] = 1



    data_train['Embarked'] = data_train['Embarked'].fillna('S')

    mean_age=data_train['Age'].mean()
    data_train['Age'] = data_train['Age'].fillna(mean_age)
    data_train['NewAge']=""
    data_train.loc[(data_train['Age'] <12),'NewAge'] = "Childrens"
    data_train.loc[(data_train['Age'] >=12)&(data_train['Age'] <18),'NewAge'] = "Teenages"
    data_train.loc[(data_train['Age'] >=18)&(data_train['Age'] <65),'NewAge'] = "Adults"
    data_train.loc[ (data_train['Age'] >=65),'NewAge'] = "Elders"
    data_train=data_train.drop(['Age'],axis=1)
    data_train=data_train.rename(columns={'NewAge':'Age'})


 
    data_train['NewSibSp']=""
    data_train.loc[data_train['SibSp']<=0,'NewSibSp']="no"
    data_train.loc[(data_train['SibSp'] >0)&(data_train['SibSp'] <=3),'NewSibSp'] = "few"
    data_train.loc[(data_train['SibSp'] >3),'NewSibSp'] = "many"
    data_train=data_train.drop(['SibSp'],axis=1)
    data_train=data_train.rename(columns={'NewSibSp':'SibSp'})


    data_train['NewParch']=""
    data_train.loc[(data_train['Parch'] <=0),'NewParch'] = "no"
    data_train.loc[(data_train['Parch'] >0)&(data_train['Parch'] <=3),'NewParch'] = "few"
    data_train.loc[(data_train['Parch'] >3),'NewParch'] = "many"
    data_train=data_train.drop(['Parch'],axis=1)
    data_train=data_train.rename(columns={'NewParch':'Parch'})


    
    print(data_train.columns)
    for feature in data_train.columns:
        print(feature,": ",data_train[feature].unique())

    print('test_data:','*'*40)
    data_test=data_test.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)
    mean_age=data_test['Age'].mean()
    data_test['Age'] = data_test['Age'].fillna(mean_age)
    data_test['NewAge']=""
    data_test.loc[(data_test['Age'] <12),'NewAge'] = "Childrens"
    data_test.loc[(data_test['Age'] >=12)&(data_test['Age'] <18),'NewAge'] = "Teenages"
    data_test.loc[(data_test['Age'] >=18)&(data_test['Age'] <65),'NewAge'] = "Adults"
    data_test.loc[ (data_test['Age'] >=65),'NewAge'] = "Elders"
    data_test=data_test.drop(['Age'],axis=1)
    data_test=data_test.rename(columns={'NewAge':'Age'})

    data_test['NewSibSp']=""
    data_test.loc[data_test['SibSp']<=0,'NewSibSp']="no"
    data_test.loc[(data_test['SibSp'] >0)&(data_test['SibSp'] <=3),'NewSibSp'] = "few"
    data_test.loc[(data_test['SibSp'] >3),'NewSibSp'] = "many"
    data_test=data_test.drop(['SibSp'],axis=1)
    data_test=data_test.rename(columns={'NewSibSp':'SibSp'})

    data_test['NewParch']=""
    data_test.loc[(data_test['Parch'] <=0),'NewParch'] = "no"
    data_test.loc[(data_test['Parch'] >0)&(data_test['Parch'] <=3),'NewParch'] = "few"
    data_test.loc[(data_test['Parch'] >3),'NewParch'] = "many"
    data_test=data_test.drop(['Parch'],axis=1)
    data_test=data_test.rename(columns={'NewParch':'Parch'})

    data_test['Embarked'] = data_test['Embarked'].fillna('S')


    for feature in data_test.columns:
        print(feature,": ",data_test[feature].unique())
    data_train=data_train.to_dict(orient="record")
    data_test=data_test.to_dict(orient="records")
    return data_train,data_test

def main():
    data_train = pd.read_csv(r"titanic_train.csv")
    data_test=pd.read_csv(r"test.csv")
    train_data,text_data=process_data(data_train,data_test)
    print("#train_data:",len(train_data))
    for _ in range(len(train_data)):
        print(train_data[_])
    print("#train_data:",len(text_data))
    for _ in range(len(text_data)):
        print(text_data[_])

if __name__=="__main__":
    main()