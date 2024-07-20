#!/usr/bin/env python
# coding: utf-8

# # **Lung Cancer Prediction **

# Lung cancer prediction using machine learning classification models using Scikit-learn library in Python is a code implementation that aims to develop a predictive model for detecting lung cancer in patients. The code uses different machine learning algorithms, including logistic regression, decision tree, k-nearest neighbor, Gaussian naive Bayes, multinomial naive Bayes, support vector classifier, random forest, XGBoost, multi-layer perceptron, and gradient boosting classifier, to predict the likelihood of lung cancer based on a range of variables. The dataset used in the code includes various columns such as gender, age, smoking, yellow fingers, anxiety, peer pressure, chronic disease, fatigue, allergy, wheezing, alcohol consuming, coughing, shortness of breath, swallowing difficulty, chest pain, and lung cancer. By analyzing these variables and using machine learning algorithms to identify patterns and correlations, the predictive models can provide accurate assessments of a patient's risk of developing lung cancer.

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#For ignoring warning
import warnings
warnings.filterwarnings("ignore")


# pandas for data manipulation and analysis.
# numpy to help with numerical operations and working with arrays
# For creating various types of plots and visualizations, I imported matplotlib.pyplot
# seaborn, which is a library that offers a high-level interface for creating informative and visually appealing statistical graphics.
# warnings.filterwarnings("ignore") line essentially tells Python to ignore these warning messages, ensuring a smoother and less cluttered workflow

# In[2]:


df=pd.read_csv('survey lung cancer.csv')
df


# **Note: In this dataset, YES=2 & NO=1**

# In[3]:


# to know number of column and rows of the dataset
df.shape


# In[4]:


#Checking for Duplicates
df.duplicated().sum()


# In[5]:


#Removing Duplicates
df=df.drop_duplicates()


# In[6]:


#Checking for null values
df.isnull().sum()


# In[7]:


df.info()


# In[8]:


df.describe()


# **In this dataset, GENDER & LUNG_CANCER attributes are in object data type. So, let's convert them to numerical values using LabelEncoder from sklearn.
# LabelEncoder is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1.
# It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels. Also let's make every other attributes as YES=1 & NO=0.**

# In[9]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['GENDER']=le.fit_transform(df['GENDER'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
df['SMOKING']=le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY']=le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
df['WHEEZING']=le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING']=le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])


# In[10]:


#Let's check what's happened now
df


# **Note: Male=1 & Female=0. Also for other variables, YES=1 & NO=0**

# In[11]:


df.info()


# In[12]:


#Let's check the distributaion of Target variable.
sns.countplot(x='LUNG_CANCER', data=df,)
plt.title('Target Distribution');


# ##### ***That is, Target Distribution is imbalanced.***

# In[13]:


df['LUNG_CANCER'].value_counts()


# ***We will handle this imbalance before applyig algorithm.***

# **Now let's do some Data Visualizations for the better understanding of how the independent features are related to the target variable..**

# In[14]:


# function for plotting
def plot(col, df=df):
    return df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))


# In[15]:


plot('GENDER')


# In[16]:


plot('AGE')


# In[17]:


plot('SMOKING')


# In[18]:


plot('YELLOW_FINGERS')


# In[19]:


plot('ANXIETY')


# In[20]:


plot('PEER_PRESSURE')


# In[21]:


plot('CHRONIC DISEASE')


# In[22]:


plot('FATIGUE ')


# In[23]:


plot('ALLERGY ')


# In[24]:


plot('WHEEZING')


# In[25]:


plot('ALCOHOL CONSUMING')


# In[26]:


plot('COUGHING')


# In[27]:


plot('SHORTNESS OF BREATH')


# In[28]:


plot('SWALLOWING DIFFICULTY')


# In[29]:


plot('CHEST PAIN')


# **From the visualizations, it is clear that in the given dataset, the features GENDER, AGE, SMOKING and SHORTNESS OF BREATH don't have that much relationship with LUNG CANCER. So let's drop those features to make this dataset more clean.**

# In[30]:


df_new=df.drop(columns=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
df_new


# **CORRELATION**

# In[31]:


#Finding Correlation
cn=df_new.corr()
cn


# This code computes the correlation matrix for all numerical columns in the 'df_new' DataFrame. The resulting 'cn' DataFrame will display the pairwise correlation coefficients between these numerical columns. The values in the matrix can range from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.

# In[32]:


#Correlation 
cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap=cmap,annot=True, square=True)
plt.show()


# Anyone may easily spot patterns, connections, and dependencies between the dataset's parameters using the visualization that is produced. One will be able to make data-driven judgments and insights according to the reported correlations because positive correlations will be represented by one color range, negative correlations by another, and no correlation by a neutral hue.
# 
# 

# In[33]:


kot = cn[cn>=.40]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Blues")


# By using a blue color gradients to indicate the intensity of these interactions, this code will produce a heatmap that shows the data's higher correlations (those greater than or equal to 0.40). One can use this visualization to find and concentrate on the dataset's strongest associations.
# 

# # ***Feature Engineering***

# Feature Engineering is the process of creating new features using existing features.

# ***The correlation matrix shows that ANXIETY and YELLOW_FINGERS are correlated more than 50%. So, lets create a new feature combining them.***

# In[34]:


df_new['ANXYELFIN']=df_new['ANXIETY']*df_new['YELLOW_FINGERS']
df_new


# In[35]:


#Splitting independent and dependent variables
X = df_new.drop('LUNG_CANCER', axis = 1)
y = df_new['LUNG_CANCER']


# This separation of independent and dependent variables is essential for training and evaluating machine learning models. 'X' will be used as the input features, while 'y' will be used as the target variable that the model will try to predict or classify.

# # ***Target Distribution Imbalance Handling***

# In[36]:


from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)


# Oversampling techniques like ADASYN are used to mitigate the class imbalance problem, which can lead to biased machine learning models. By generating synthetic samples for the minority class, we aim to make the class distribution more balanced, which can lead to better model performance, especially in cases where the minority class is important.

# In[37]:


len(X)


# # **Logistic Regression**

# In[38]:


#Splitting data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)


# After running this code, I'll have data split into training and testing sets, allowing me to train machine learning model on X_train and y_train and then evaluate its performance on X_test and y_test. This separation help me assess how well model generalizes to new, unseen data.

# In[39]:


#Fitting training data to the model
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)


# After this line is executed, lr_model will be a trained logistic regression model that has learned to make predictions based on the relationships between the features in X_train and the corresponding labels in y_train. You can then use this model to make predictions on new data or evaluate its performance on the test data (X_test and y_test).

# In[40]:


#Predicting result using testing data
y_lr_pred= lr_model.predict(X_test)
y_lr_pred


# The variable y_lr_pred now contains the model's predictions for the testing data. You can use these predictions to evaluate the performance of the logistic regression model, typically by comparing them to the actual labels (y_test) to calculate metrics such as accuracy, precision, recall, or F1-score, depending on the specific classification task.

# In[41]:


#Model accuracy
from sklearn.metrics import classification_report, accuracy_score, f1_score
lr_cr=classification_report(y_test, y_lr_pred)
print(lr_cr)


# The classification report includes metrics such as precision, recall, F1-score, and support for each class. These metrics provide insights into how well your model is performing, especially in binary classification tasks. You can use this information to understand the model's strengths and weaknesses and to assess its suitability for your specific problem.

# **This model is almost 98% accurate.**

# # **Decision Tree**

# In[42]:


#Fitting training data to the model
from sklearn.tree import DecisionTreeClassifier
dt_model= DecisionTreeClassifier(criterion='entropy', random_state=0)  
dt_model.fit(X_train, y_train)


# In[43]:


#Predicting result using testing data
y_dt_pred= dt_model.predict(X_test)
y_dt_pred


# In[44]:


#Model accuracy
dt_cr=classification_report(y_test, y_dt_pred)
print(dt_cr)


# **This model is 97% accurate.**

# # **K Nearest Neighbor**

# In[45]:


#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
knn_model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn_model.fit(X_train, y_train)


# In[46]:


#Predicting result using testing data
y_knn_pred= knn_model.predict(X_test)
y_knn_pred


# In[47]:


#Model accuracy
knn_cr=classification_report(y_test, y_knn_pred)
print(knn_cr)


# **This model is 96% accurate.**

# # **Gaussian Naive Bayes**

# In[48]:


#Fitting Gaussian Naive Bayes classifier to the training set  
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)


# In[49]:


#Predicting result using testing data
y_gnb_pred= gnb_model.predict(X_test)
y_gnb_pred


# In[50]:


#Model accuracy
gnb_cr=classification_report(y_test, y_gnb_pred)
print(gnb_cr)


# **This model is 93% accurate.**

# # **Multinomial Naive Bayes**

# In[51]:


#Fitting Multinomial Naive Bayes classifier to the training set  
from sklearn.naive_bayes import MultinomialNB
mnb_model = MultinomialNB()
mnb_model.fit(X_train, y_train)


# In[52]:


#Predicting result using testing data
y_mnb_pred= mnb_model.predict(X_test)
y_mnb_pred


# In[53]:


#Model accuracy
mnb_cr=classification_report(y_test, y_mnb_pred)
print(mnb_cr)


# **This model is 80% accurate.**

# # **Support Vector Classifier**

# In[54]:


#Fitting SVC to the training set  
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)


# In[55]:


#Predicting result using testing data
y_svc_pred= svc_model.predict(X_test)
y_svc_pred


# In[56]:


#Model accuracy
svc_cr=classification_report(y_test, y_svc_pred)
print(svc_cr)


# **This model is 97% accurate.**

# # **Random Forest**

# In[57]:


#Training
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# In[58]:


#Predicting result using testing data
y_rf_pred= rf_model.predict(X_test)
y_rf_pred


# In[59]:


#Model accuracy
rf_cr=classification_report(y_test, y_rf_pred)
print(rf_cr)


# **This model is also 98% accurate.**

# # **XGBoost**

# In[60]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)


# In[61]:


#Predicting result using testing data
y_xgb_pred= xgb_model.predict(X_test)
y_xgb_pred


# In[62]:


#Model accuracy
xgb_cr=classification_report(y_test, y_xgb_pred)
print(xgb_cr)


# **This model is also 97% accurate.**

# # **Multi-layer Perceptron classifier**

# In[63]:


#Training a neural network model
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier()
mlp_model.fit(X_train, y_train)


# In[64]:


#Predicting result using testing data
y_mlp_pred= mlp_model.predict(X_test)
y_mlp_pred


# In[65]:


#Model accuracy
mlp_cr=classification_report(y_test, y_mlp_pred)
print(mlp_cr)


# **This model is also 98% accurate.**

# # **Gradient Boosting**

# In[66]:


#Training
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)


# In[67]:


#Predicting result using testing data
y_gb_pred= gb_model.predict(X_test)
y_gb_pred


# In[68]:


#Model accuracy
gb_cr=classification_report(y_test, y_gb_pred)
print(gb_cr)


# **This model is also 98% accurate.**

# From the above calculated accuracies, it is clear that the SVC, Random Forest, Multi-layer Perceptron and Gradient Boost models performed atmost level while the worst performed one is Multinomial Naive Bayes. 
# However, I'm interested in a more efficient way of evaluating these models. Let's go for the Cross Validation methods using both K-Fold and Stratified K-Fold

# # **Cross Validation**

# K-Fold cross validation is a popular technique used in machine learning for model evaluation and selection. It involves dividing a dataset into K subsets of equal size, called folds. The algorithm then trains and evaluates the model K times, each time using a different fold as the validation set and the remaining K-1 folds as the training set.
# 
# During each iteration of K-Fold cross validation, the model is trained on K-1 folds and evaluated on the remaining fold. The performance metrics are then averaged over all K iterations to obtain an estimate of the model's overall performance.
# 
# K-Fold cross validation is a robust method for model evaluation because it uses all the available data for training and testing. It also helps to reduce the risk of overfitting and provides a more accurate estimate of the model's performance than using a single training-test split.
# 
# Typically, values of K between 5 and 10 are used for K-Fold cross validation, but the optimal value of K may vary depending on the size and complexity of the dataset, as well as the type of model being evaluated.
# 
# Here are some general guidelines that can help you choose an appropriate value of K:
# 
# 1. For small datasets, it is recommended to use a larger value of K, such as 10. This is because the larger value of K allows for more robust estimates of model performance, given the limited amount of data.
# 
# 2. For larger datasets, a smaller value of K can be used, such as 5. This is because a larger value of K will result in K smaller training sets, which may not be representative of the full dataset. Using a smaller value of K ensures that each fold has a sufficient amount of data for both training and testing.
# 
# 3. For models that are computationally expensive or time-consuming to train, a smaller value of K is preferred to reduce the overall training time.
# 
# 4. It's also essential to note that the choice of K should not be based solely on the accuracy of the model. Other metrics, such as precision, recall, and F1 score, should also be considered, as they can provide valuable insights into the performance of the model.

# In[69]:


# K-Fold Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# Logistic regerssion model
lr_model_scores = cross_val_score(lr_model,X, y, cv=kf)

# Decision tree model
dt_model_scores = cross_val_score(dt_model,X, y, cv=kf)

# KNN model
knn_model_scores = cross_val_score(knn_model,X, y, cv=kf)

# Gaussian naive bayes model
gnb_model_scores = cross_val_score(gnb_model,X, y, cv=kf)

# Multinomial naive bayes model
mnb_model_scores = cross_val_score(mnb_model,X, y, cv=kf)

# Support Vector Classifier model
svc_model_scores = cross_val_score(svc_model,X, y, cv=kf)

# Random forest model
rf_model_scores = cross_val_score(rf_model,X, y, cv=kf)

# XGBoost model
xgb_model_scores = cross_val_score(xgb_model,X, y, cv=kf)

# Multi-layer perceptron model
mlp_model_scores = cross_val_score(mlp_model,X, y, cv=kf)

# Gradient boost model
gb_model_scores = cross_val_score(gb_model,X, y, cv=kf)

print("Logistic regression models' average accuracy:", np.mean(lr_model_scores))
print("Decision tree models' average accuracy:", np.mean(dt_model_scores))
print("KNN models' average accuracy:", np.mean(knn_model_scores))
print("Gaussian naive bayes models' average accuracy:", np.mean(gnb_model_scores))
print("Multinomial naive bayes models' average accuracy:", np.mean(mnb_model_scores))
print("Support Vector Classifier models' average accuracy:", np.mean(svc_model_scores))
print("Random forest models' average accuracy:", np.mean(rf_model_scores))
print("XGBoost models' average accuracy:", np.mean(xgb_model_scores))
print("Multi-layer perceptron models' average accuracy:", np.mean(mlp_model_scores))
print("Gradient boost models' average accuracy:", np.mean(gb_model_scores))


# **So the K-Fold cross validation is showing Random Forest model gives the most accuracy of 95.1%, and XGBoost also gives almost same accuracy of 95.1%, while Multinomial Naive Bayes model gives the least accuarcy of 77.2%.**

# Stratified K-Fold cross-validation is a modification of the standard K-Fold cross-validation technique that is commonly used in machine learning when working with imbalanced datasets. The goal of Stratified K-Fold cross-validation is to ensure that each fold is representative of the overall dataset in terms of the class distribution.
# 
# In standard K-Fold cross-validation, the data is split into K folds, and each fold is used as the validation set in turn. However, if the dataset has an imbalanced class distribution, this can lead to some of the folds having significantly fewer samples from the minority class, which can result in biased performance estimates.
# 
# To address this issue, Stratified K-Fold cross-validation ensures that each fold has a similar proportion of samples from each class. It works by first dividing the dataset into K folds, as in standard K-Fold cross-validation. Then, for each fold, the algorithm ensures that the proportion of samples from each class is roughly the same as the proportion in the full dataset. This ensures that the model is evaluated on a representative sample of the data, regardless of the class distribution.
# 
# Stratified K-Fold cross-validation is a powerful tool for evaluating the performance of machine learning models on imbalanced datasets. It can help to ensure that the model's performance is accurately estimated and that the model is robust to class imbalances in the dataset.

# In[70]:


# Stratified K-Fold Cross Validation

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

k = 10
kf = StratifiedKFold(n_splits=k)


# Logistic regerssion model
lr_model_scores = cross_val_score(lr_model,X, y, cv=kf)

# Decision tree model
dt_model_scores = cross_val_score(dt_model,X, y, cv=kf)

# KNN model
knn_model_scores = cross_val_score(knn_model,X, y, cv=kf)

# Gaussian naive bayes model
gnb_model_scores = cross_val_score(gnb_model,X, y, cv=kf)

# Multinomial naive bayes model
mnb_model_scores = cross_val_score(mnb_model,X, y, cv=kf)

# Support Vector Classifier model
svc_model_scores = cross_val_score(svc_model,X, y, cv=kf)

# Random forest model
rf_model_scores = cross_val_score(rf_model,X, y, cv=kf)

# XGBoost model
xgb_model_scores = cross_val_score(xgb_model,X, y, cv=kf)

# Multi-layer perceptron model
mlp_model_scores = cross_val_score(mlp_model,X, y, cv=kf)

# Gradient boost model
gb_model_scores = cross_val_score(gb_model,X, y, cv=kf)


print("Logistic regression models' average accuracy:", np.mean(lr_model_scores))
print("Decision tree models' average accuracy:", np.mean(dt_model_scores))
print("KNN models' average accuracy:", np.mean(knn_model_scores))
print("Gaussian naive bayes models' average accuracy:", np.mean(gnb_model_scores))
print("Multinomial naive bayes models' average accuracy:", np.mean(mnb_model_scores))
print("Support Vector Classifier models' average accuracy:", np.mean(svc_model_scores))
print("Random forest models' average accuracy:", np.mean(rf_model_scores))
print("XGBoost models' average accuracy:", np.mean(xgb_model_scores))
print("Multi-layer perceptron models' average accuracy:", np.mean(mlp_model_scores))
print("Gradient boost models' average accuracy:", np.mean(gb_model_scores))


# **So the Stratified K-Fold cross validation is showing Gradientboost model gives the most accuracy of 95.3%. S  while Multinomial Naive Bayes model gives the least accuarcy of 75.1%.**
