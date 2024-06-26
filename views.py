from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Create your views here.

def hospital(request):

    data=pd.read_csv("training_data.csv")

    median=data.A.median()
    data.A=data.A.fillna(median)
    median=data.B.median()
    data.B=data.B.fillna(median)
    median=data.C.median()
    data.C=data.C.fillna(median)
    median=data.D.median()
    data.D=data.D.fillna(median)
    median=data.E.median()
    data.E=data.E.fillna(median)
    median=data.F.median()
    data.F=data.F.fillna(median)
    median=data.Z.median()
    data.Z=data.Z.fillna(median)
    median=data.Number_of_prev_cond.median()
    data.Number_of_prev_cond=data.Number_of_prev_cond.fillna(median)
    median=data.Number_of_prev_cond.median()
    data.Treated_with_drugs=data.Treated_with_drugs.fillna(median)

    le=LabelEncoder()

    data['Patient_Smoker']=le.fit_transform(data['Patient_Smoker'])
    data['Patient_Rural_Urban']=le.fit_transform(data['Patient_Rural_Urban'])
    data['Patient_mental_condition']=le.fit_transform(data['Patient_mental_condition'])

    inputs=data.drop(['Treated_with_drugs','Survived_1_year','Patient_mental_condition'],'columns')
    output=data['Survived_1_year']

    xtrain,xtest,ytrain,ytest=train_test_split(inputs,output,test_size=0.2)

    model=LogisticRegression()

    model.fit(xtrain,ytrain)
    
    data=request.POST
    if 'submit' in data:
        
        inp1=int(data.get('ID_Patient_Care_Situation'))
        inp2=int(data.get('Diagnosed_Condition'))
        inp3=int(data.get('Patient_ID'))
        inp4=int(data.get('Patient_Age'))
        inp5=int(data.get('Patient_Body_Mass_Index'))
        inp6=int(data.get('Patient_Smoker'))
        inp7=int(data.get('Patient_Rural_Urban'))
        inp9=int(data.get('A'))
        inp10=int(data.get('B'))
        inp11=int(data.get('C'))
        inp12=int(data.get('D'))
        inp13=int(data.get('E'))
        inp14=int(data.get('E'))
        inp15=int(data.get('Z'))
        inp16=int(data.get('Number_of_prev_cond'))
        res =model.predict([[inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp9,inp10,inp11,inp12,inp13,inp14,inp15,inp16]])
        return render(request,'hospital.html',context={'res':res})

    return render(request,'hospital.html')


def csub(request):

    data=pd.read_csv("training_set_label.csv")

    le=LabelEncoder()

    data['job']=le.fit_transform(data['job'])
    data['marital']=le.fit_transform(data['marital'])
    data['education']=le.fit_transform(data['education'])
    data['default']=le.fit_transform(data['default'])
    data['housing']=le.fit_transform(data['housing'])
    data['loan']=le.fit_transform(data['loan'])
    data['contact']=le.fit_transform(data['contact'])
    data['month']=le.fit_transform(data['month'])
    data['poutcome']=le.fit_transform(data['poutcome'])

    inputs=data.drop('subscribe','columns')
    output=data['subscribe']

    x_train,x_test,y_train,y_test=train_test_split(inputs,output,test_size=0.2)

    model=KNeighborsClassifier(n_neighbors=213)
    model.fit(x_train,y_train)

    data=request.POST
    if 'submit' in data:
        
        inp1=int(data.get('inp1'))
        inp2=int(data.get('inp2'))
        inp3=int(data.get('inp3'))
        inp4=int(data.get('inp4'))
        inp5=int(data.get('inp5'))
        inp6=int(data.get('inp6'))
        inp7=int(data.get('inp7'))
        inp8=int(data.get('inp8'))
        inp9=int(data.get('inp9'))
        inp10=int(data.get('inp10'))
        inp11=int(data.get('inp11'))
        inp12=int(data.get('inp12'))
        inp13=int(data.get('inp13'))
        inp14=int(data.get('inp14'))
        inp15=int(data.get('inp15'))
        inp16=int(data.get('inp16'))
        res =model.predict([[inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inp11,inp12,inp13,inp14,inp15,inp16]])
        return render(request,'csub.html',context={'res':res})

    return render(request,'csub.html')