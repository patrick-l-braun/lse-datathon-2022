import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#Read Raw DataSet
df = pd.read_csv(r"C:\Users\bbste\Documents\Coding\Python\DSW\Raw Data.csv")

#Turn categorical variables into binary variables
categorical_variables = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                         'Education', 'PerformanceRating', 'StockOptionLevel', 'WorkLifeBalance',
                         'RelationshipSatisfaction', 'JobSatisfaction', 'JobInvolvement',
                         'EnvironmentSatisfaction', 'JobLevel']

#Drop unimportant inforamtion
data_df = pd.get_dummies(df, columns=categorical_variables, drop_first=True)
data_df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'MonthlyIncome'], axis=1, inplace=True)

#Map 'yes' and 'no' to 1 and 0
data_df['Attrition'] = data_df['Attrition'].map({'Yes': 1, 'No': 0})
data_df['Over18'] = data_df['Over18'].map({'Y': 1, 'N': 0})
data_df['OverTime'] = data_df['OverTime'].map({'Yes': 1, 'No': 0})

#Export cleaned datframe to .csv file
data_df.to_csv(r"C:\Users\bbste\Documents\Coding\Python\DSW\Cleaned Dataset.csv")

#Define our independent and dependent variables
x = data_df.drop(columns=['Attrition'])
y = data_df['Attrition']

#Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10, shuffle=False)

#Create a list of models and f1 scores
model = []
scores = []

#Create multiple knn models
knn_models = []
for i in range(1, 6):
    knn_models.append("knn_model_"+str(i))
    model.append(str(i)+"_nn")    
    
knn_score = []
for i in range(1, 6):
    knn_score.append("knn_score_"+str(i))
    
knn_pred = []
for i in range(1, 6):
    knn_pred.append("knn_pred_"+str(i))   
 
#Train all 5 knn models and calculate f1 scores
for i in range(5):
    knn_models[i] = KNeighborsClassifier(n_neighbors=i+1)
    knn_models[i].fit(x_train, y_train)
    knn_pred[i] = knn_models[i].predict(x_test)
    knn_score[i] = f1_score(y_true=y_test, y_pred=knn_pred[i])
    scores.append(knn_score[i])

#Create a k means clustering model with 2 clusters
k_means_model = KMeans(n_clusters = 2)
k_means_model.fit(x_train, y_train)
k_means_pred = k_means_model.predict(x_test)
k_means_score = f1_score(y_true=y_test, y_pred=k_means_pred)   
model.append("2_means")
scores.append(k_means_score)

#Create 3 more models
ridge_model = RidgeClassifier()
rf_model = RandomForestClassifier()
svm_model = svm.SVC(kernel='linear')

model.append("ridge")
model.append("rf")
model.append("svm")

#Fit these models
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
svm_model.fit(x_train, y_train)

ridge_pred = ridge_model.predict(x_test)
rf_pred = rf_model.predict(x_test)
svm_pred = svm_model.predict(x_test)

#Calculate f1 scores of these models
ridge_score = f1_score(y_true=y_test, y_pred=ridge_pred)
rf_score = f1_score(y_true=y_test, y_pred=rf_pred)
svm_score = f1_score(y_true=y_test, y_pred=svm_pred)

scores.append(ridge_score)
scores.append(rf_score)
scores.append(svm_score)

print(model)
print(scores)

#Create a bar chart comparing these models
plt.bar(model, scores)
plt.title('F1 Score for different models')
plt.xlabel('Model')
plt.ylabel('Score')
plt.show()
