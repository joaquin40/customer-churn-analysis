#!/usr/bin/env python
# coding: utf-8

# In[442]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA # PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, jaccard_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV, KFold # train-test split, cross validation


# In[208]:


df = pd.DataFrame(pd.read_csv("./data/customer-churn.csv"))


# In[209]:


df


# In[210]:


summary_num = df.describe()
print(summary_num)

summary_factor = df.describe(include = 'object')
print(summary_factor)


# In[211]:


print("Dimensions:",df.shape)


# In[212]:


#remove null values
df['TotalCharges'].str.strip().eq('').sum()
df["total_charges_isnull"] = df['TotalCharges'].str.strip().eq('')
df


# In[235]:


df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")


# In[237]:


df1 = df[df['total_charges_isnull'] == False].drop(["total_charges_isnull","customerID"], axis=1)


# In[255]:


# convert total charges to numeric from strings
df1["TotalCharges"].describe(include='object')
df1['TotalCharges'] = df1['TotalCharges'].apply(float)
df1.describe()


# In[239]:


sns.heatmap(df1.isna(), cbar=False)
na_count = df1.isna().sum()
print(na_count)


# In[256]:


# one hot-encoder
encoder = OneHotEncoder(sparse_output=False)

# used for col names
df_factors = df1.select_dtypes(include='object').drop("Churn", axis = 1)

df1_encod = pd.DataFrame(encoder.fit_transform(df_factors[df_factors.columns]),
                         columns=encoder.get_feature_names_out(df_factors.columns))


# In[278]:


num_df = df1.drop(df_factors.columns,axis=1).reset_index(drop = True)
num_df1 = num_df.drop("Churn", axis=1)


# In[281]:


# Normalize the numeric data
#numeric_columns = num_df1.select_dtypes(include=['int64', 'float64']).columns
numeric_columns = num_df1.columns
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df1[numeric_columns])

#make to dataframe scaled 
scaled_df1 = pd.DataFrame(scaled_values, columns=numeric_columns)



# In[485]:


# combine dataset with one hot-encoder
df2 = pd.concat([scaled_df1, df1_encod,df1['Churn'].reset_index(drop=True)], axis = 1)
df2


# In[288]:


# convert response to 1 or 0
def is_churn(Churn):
    if Churn == "Yes":
        return 1
    else:
        return 0
    
df2["is_churn"] = df2["Churn"].apply(is_churn)


# In[289]:


df3 = df2.drop("Churn", axis=1) 
df3


# In[284]:


# df3.to_csv("file.csv", index=False)


# In[295]:


num_df1.columns


# In[297]:


plt.hist(df1['TotalCharges'])
plt.xlabel('Total Charges')
plt.ylabel('Frequency')
plt.title('Histogram of Total Charges')


# In[482]:


plt.hist(df1['tenure'], bins=70)
plt.xlabel('Tenure of customer in months')
plt.ylabel('Frequencey')


# In[479]:


plt.hist(df1['MonthlyCharges'], color='green')
plt.xlabel('Monthly Charges $')
plt.ylabel("Frequency")


# In[306]:


df1.describe(include='object')


# In[314]:


gender_labels = ['Male', 'Female']
gender_values = [0, 1]

plt.bar(df1['SeniorCitizen'].value_counts().index, df1['SeniorCitizen'].value_counts().values )
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(gender_values, gender_labels)


# In[347]:


plt.bar(df1['Contract'].value_counts().index, df1['Contract'].value_counts().values )
plt.ylabel("Count")
plt.title('Contract term of customer')


# In[333]:


from sklearn.linear_model import LogisticRegression


# In[325]:


x = df3.drop('is_churn', axis=1)
y = df3['is_churn']


# In[329]:


#logistic
logistic_pipe = Pipeline([
    ('log', LogisticRegression(penalty='l1', solver='saga', max_iter=2000))
], verbose=True)

param_grid = {
    'log__class_weight': [None,'balanced']
}


# In[330]:


# fit logistic lasso regression
logistic_cv = GridSearchCV(logistic_pipe, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
logistic_cv.fit(x,y)


# In[331]:


# fit grid search to data
log_best_params = logistic_cv.best_params_
print(log_best_params)


# In[343]:


coeff = logistic_cv.best_estimator_.named_steps['log'].coef_

coef_df = pd.DataFrame({'Variable': x.columns,
                        'Log-Odds Coefficient': coeff[0],
                        'Odds Ratio coefficient': np.exp(coeff[0]) })

print(tabulate(coef_df.sort_values('Odds Ratio coefficient', ascending= False), headers='keys', tablefmt='psql'))


# In[348]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[356]:


tree_pipeline = Pipeline([
    ('tree', DecisionTreeClassifier())
], verbose=True)

tree_param = {
    'tree__criterion': ['gini', 'entropy'],
    'tree__max_depth': [3, 5, 10, None],
    'tree__min_samples_split': [2, 5, 10],
    'tree__min_samples_leaf': [1, 2, 4]    
}


# In[355]:


# tune paramters
tree_cv = GridSearchCV(tree_pipeline, tree_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
tree_cv.fit(x, y)


# In[357]:


# obtain the estimators
tree_best = tree_cv.best_estimator_.named_steps['tree']

# obtain the predictors 
tree_importances = tree_best.feature_importances_
print(tree_importances)


# In[363]:


# increase in gini index
variables_names = x.columns

var_imp_tree = pd.DataFrame({"Predictors": variables_names, 'gini' : tree_importances})
var_imp_tree_sorted = var_imp_tree.sort_values(by = 'gini',ascending=False)
print("Top 5 important predictors for bagging trees model")
print(var_imp_tree_sorted.head(5))


# In[374]:


# chart of importance variables
importance = pd.Series(tree_importances, index=variables_names)
top_n = 10
top_n_idx = importance.argsort()[-top_n:]
top_n_importance = importance.iloc[top_n_idx]
top_n_importance.plot(kind="barh")
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Decision Tree Model')


# In[375]:


var_imp_tree_sorted.head(10).sort_values(by = 'gini').plot(x = 'Predictors', y = 'gini', kind = 'barh', label = '')
plt.legend('')
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Decision Tree Model')
plt.ylabel('')


# In[377]:


# bagging trees model
bagging_pipeline = Pipeline([
    ('bagging', BaggingClassifier(estimator=DecisionTreeClassifier()))
],verbose=True)

# parameters bagging
# parameters bagging
bagging_params = {
    'bagging__n_estimators': [10, 50, 100, 500],
}

# random forest model pipeline
rf_pipeline = Pipeline([
    ('rf', RandomForestClassifier())
], verbose=True)

# rf parameters
rf_param = {
    'rf__n_estimators': [10, 50, 100,500],
    'rf__max_depth': [None, 5, 10],
}


# In[379]:


#Fit bagging tree model
bagging_cv  = GridSearchCV(bagging_pipeline, bagging_params, cv = 10, scoring="accuracy", n_jobs=-1 )
bagging_cv.fit(x,y)


# In[380]:


# obtain the estimators
bagging_best = bagging_cv.best_estimator_.named_steps['bagging']


# In[382]:


# obtain the predictors 
bagging_importances = bagging_best.estimators_[0].feature_importances_
bagging_importances


# In[400]:


# %IncMSE/gini percent increase of MSE 
# estimate the importance of each predictor
# by looking at the increase MSE/gini
# when the predictor variable is removed from the model
top_n = 10


var_imp_bagg = pd.DataFrame({'Predictors': variables_names, 'gini': bagging_importances})
var_imp_bagg_sorted = var_imp_bagg.sort_values(by="gini", ascending=False)
print(f"Top {top_n} important predictors for bagging trees model")
print(var_imp_bagg_sorted.head(top_n))


# In[483]:


var_imp_bagg_sorted.head(top_n).sort_values(by='gini').plot(x = "Predictors", y = 'gini', kind='barh', label = 'gini')
plt.legend('')
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Bagging Decision Tree Model')
plt.ylabel('')


# In[396]:


#Random forest model
rf_cv = GridSearchCV(rf_pipeline, rf_param, cv = 10, scoring='accuracy', n_jobs=-1)
rf_cv.fit(x,y)


# In[398]:


#calculate important predictors

# obtain the estimators
rf_best =  rf_cv.best_estimator_.named_steps['rf']


# In[399]:


# obtain the predictors %incmse for RF
rf_importances = rf_best.estimators_[0].feature_importances_
print(rf_importances)


# In[401]:


# top five important variables using %IncMSE
var_imp_rf = pd.DataFrame({'Predictors':variables_names, 'gini':rf_importances})
var_imp_rf_sorted = var_imp_rf.sort_values(by = 'gini', ascending=False)
print(f"Top {top_n} important predictors for Random Forest trees model")
print(var_imp_rf_sorted.head(top_n))


# In[410]:


var_imp_rf_sorted.head(top_n).sort_values(by='gini').plot(x = "Predictors", y = 'gini',kind='barh', label='')
plt.legend('')
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Random Forest')
plt.ylabel('')


# In[412]:


#boosting
from sklearn.ensemble import GradientBoostingClassifier

# boosting - GradientBoostingClassifier default trees
boost_tree_pipeline =  Pipeline([
   ('gbt', GradientBoostingClassifier())
], verbose = True)


# In[413]:


# parameters
boost_tree_param =  {
   'gbt__learning_rate': [0.1, 0.05, 0.01],
    'gbt__n_estimators': [50, 100, 200],
    'gbt__max_depth': [3, 4, 5]
}

#n_estimators: number of decision trees to include in the ensemble.
#learning_rate: the learning rate shrinks the contribution of each tree by learning_rate amount.
#max_depth: the maximum depth of the decision trees.
#max_features: the number of features to consider when looking for the best split. 
#subsample: the fraction of samples to be used for fitting the individual base learners. Values lower than 1.0 would make


# In[414]:


# fit model & cross validation k-fold
boosting_tree_cv = GridSearchCV(boost_tree_pipeline, param_grid=boost_tree_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
boosting_tree_cv.fit(x, y)


# In[415]:


#calculate important predictors

# obtain the estimators
boosting_best =  boosting_tree_cv.best_estimator_.named_steps['gbt']

# obtain the predictors %incmse for RF
boosting_importances = boosting_best.feature_importances_
print(boosting_importances)


# In[420]:


# top five important variables using %IncMSE
top_n = 10

var_imp_boost = pd.DataFrame({'Predictors':variables_names, 'gini':boosting_importances})
var_imp_boost_sorted = var_imp_boost.sort_values(by = 'gini', ascending=False)
print(f"Top {top_n} important predictors for bagging trees model")
print(var_imp_boost_sorted.head(top_n))


# In[422]:


var_imp_boost_sorted.head(top_n).sort_values(by='gini').plot(x = "Predictors", y = 'gini',kind='barh', label='')
plt.legend('')
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Boosting Decision Tree')
plt.ylabel('')


# In[432]:


# prediction analysis of churn customers
# 70/30 split training/test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.3, random_state=1) 


# In[437]:


# decision tree model
# tune parameters, fit model on traning set for prediction 
tree_cv = GridSearchCV(tree_pipeline, tree_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
tree_cv.fit(x_train, y_train)


# In[438]:


#####
#####

#predictions on test set
tree_cv.best_estimator_
tree_pred = tree_cv.predict(x_test)


# In[439]:


# confusion matrix
print(pd.crosstab(y_test, tree_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[440]:


# heatmap of results

tree_cm = confusion_matrix(y_test, tree_pred)
sns.heatmap(tree_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[444]:


#  confusion matrix
report = classification_report(y_test, tree_pred)

# Print the report
print(report)
t_accuracy = accuracy_score(y_test, tree_pred).round(2)
t_precision = precision_score(y_test, tree_pred).round(2)
t_recall = recall_score(y_test, tree_pred).round(2)
t_f1 = f1_score(y_test, tree_pred).round(2)
t_balanced_acc = balanced_accuracy_score(y_test, tree_pred).round(2)


print(f"Decision tree model: accuracy: {t_accuracy}, precision: {t_precision}, recall: {t_recall}, f1 score: {t_f1}, balanced accuracy {t_balanced_acc}")


# In[445]:


# cross validation bagging model
bagging_cv = GridSearchCV(bagging_pipeline, param_grid = bagging_params, cv = 10, scoring = 'accuracy', n_jobs=-1)# ,'f1'])
bagging_cv.fit(x_train, y_train)


# In[446]:


# cross validation rf model 
rf_cv = GridSearchCV(rf_pipeline, param_grid= rf_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
rf_cv.fit(x_train, y_train)


# In[447]:


# parameters of n trees bagging
bagging_best_params = bagging_cv.best_params_
print(bagging_best_params)


# parameters of rf 
rf_best_params = rf_cv.best_params_
print(rf_best_params)


# In[448]:


# prediction on test dataset using bagging model
pred_bagging = bagging_cv.predict(x_test)


# In[449]:


bagging_cm = confusion_matrix(y_test, pred_bagging)
print(pd.crosstab(y_test, pred_bagging, rownames=['True'], colnames=['Predicted'], margins=True))


# In[450]:


# heatmap of results
sns.heatmap(bagging_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[451]:


#  confusion matrix
report = classification_report(y_test, pred_bagging)

# Print the report
print(report)
bag_accuracy = accuracy_score(y_test, pred_bagging).round(2)
bag_precision = precision_score(y_test, pred_bagging).round(2)
bag_recall = recall_score(y_test, pred_bagging).round(2)
bag_f1 = f1_score(y_test, pred_bagging).round(2)
bag_balanced_acc = balanced_accuracy_score(y_test, pred_bagging). round(2)


print(f"Decision tree model: accuracy: {bag_accuracy}, precision: {bag_precision}, recall: {bag_recall}, f1 score: {bag_f1}, Balanced accuracy {bag_balanced_acc}")


# In[452]:


# random forest predicitons
pred_rf = rf_cv.predict(x_test)


# In[453]:


# confusion matrix
rf_cm = confusion_matrix(y_test, pred_rf)
print(pd.crosstab(y_test, pred_rf, rownames=['True'], colnames=['Predicted'], margins=True))


# In[454]:


# heatmap of results
sns.heatmap(rf_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[455]:


#  confusion matrix
report_rf = classification_report(y_test, pred_rf)

# Print the report
print(report_rf)
rf_accuracy = accuracy_score(y_test, pred_rf).round(2)
rf_precision = precision_score(y_test, pred_rf).round(2)
rf_recall = recall_score(y_test, pred_rf).round(2)
rf_f1 = f1_score(y_test, pred_rf).round(2)
rf_balanced_acc = balanced_accuracy_score(y_test, pred_rf). round(2)

print(f"Decision tree model: accuracy: {rf_accuracy}, precision: {rf_precision}, recall: {rf_recall}, f1 score: {rf_f1}, Balanced accuracy {rf_balanced_acc}")


# In[456]:


# boosting model
# fit model & cross validation k-fold
boosting_tree_cv = GridSearchCV(boost_tree_pipeline, param_grid=boost_tree_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
boosting_tree_cv.fit(x_train, y_train)

# parameters of boosting 
boost_best_params = boosting_tree_cv.best_params_
print(boost_best_params)

#{'gbt__learning_rate': 0.1, 'gbt__max_depth': 6, 'gbt__n_estimators': 200}


# In[457]:


# prediction on test dataset using bagging model
pred_boosting = boosting_tree_cv.predict(x_test)


# In[459]:


print(pd.crosstab(y_test, pred_boosting, rownames=['True'], colnames=['Predicted'], margins=True))


# In[461]:


# heatmap of results
boosting_cm = confusion_matrix(y_test, pred_boosting) # matrix

sns.heatmap(boosting_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[462]:


#  confusion matrix
report_boosting = classification_report(y_test, pred_boosting)

# Print the report
print(report_boosting)
boost_accuracy = accuracy_score(y_test, pred_boosting).round(2)
boost_precision = precision_score(y_test, pred_boosting).round(2)
boost_recall = recall_score(y_test, pred_boosting).round(2)
boost_f1 = f1_score(y_test, pred_boosting).round(2)
boost_balanced_acc = balanced_accuracy_score(y_test, pred_boosting). round(2)

print(f"Decision tree model: accuracy: {boost_accuracy}, precision: {boost_precision}, recall: {boost_recall}, f1 score: {boost_f1}, Balanced accuracy {boost_balanced_acc}")


# In[463]:


# SVM
from sklearn.svm import SVC


# In[464]:


# SVM MODEL
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
], verbose=True)

# svm parameter
svm_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'poly','rbf'],
    'svm__gamma': [0.1, 1, 10 ]
}


# In[465]:


# fit model & cross validation k-fold
svm_cv = GridSearchCV(svm_pipeline, param_grid=svm_grid, cv = 10, scoring = 'accuracy', n_jobs=-1)
svm_cv.fit(x_train, y_train)


# In[466]:


# parameters of svm
svm_best_params = svm_cv.best_params_
print(svm_best_params)


# In[467]:


# prediction on test dataset using svm model
pred_svm = svm_cv.predict(x_test)


# In[468]:


print(pd.crosstab(y_test, pred_svm, rownames=['True'], colnames=['Predicted'], margins=True))


# In[469]:


# heatmap of results
svm_cm = confusion_matrix(y_test, pred_svm) # matrix

sns.heatmap(svm_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[470]:


#  confusion matrix
report_svm = classification_report(y_test, pred_svm)

# Print the report
print(report_svm)
svm_accuracy = accuracy_score(y_test, pred_svm).round(2)
svm_precision = precision_score(y_test, pred_svm).round(2)
svm_recall = recall_score(y_test, pred_svm).round(2)
svm_f1 = f1_score(y_test, pred_svm).round(2)
svm_balanced_acc = balanced_accuracy_score(y_test, pred_svm). round(2)

print(f"SVM model: accuracy: {svm_accuracy}, precision: {svm_precision}, recall: {svm_recall}, f1 score: {svm_f1}, balanced accuracy {svm_balanced_acc}")


# In[ ]:


# fit logistic lasso regression
logistic_cv = GridSearchCV(logistic_pipe, param_grid, cv=10, scoring='accuracy', n_jobs = -1)
logistic_cv.fit(x_train, y_train)


# In[471]:


# Fit grid search object to the data
log_best_params = logistic_cv.best_params_
print(log_best_params)

pred_log = logistic_cv.predict(x_test)


# In[472]:


print(pd.crosstab(y_test, pred_log, rownames=['True'], colnames=['Predicted'], margins=True))


# In[473]:


# heatmap of results
log_cm = confusion_matrix(y_test, pred_log) # matrix

sns.heatmap(log_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[474]:


#  confusion matrix
report_log = classification_report(y_test, pred_log)

# Print the report
print(report_log)
log_accuracy = accuracy_score(y_test, pred_log).round(2)
log_precision = precision_score(y_test, pred_log).round(2)
log_recall = recall_score(y_test, pred_log).round(2)
log_f1 = f1_score(y_test, pred_log).round(2)
log_balanced_acc = balanced_accuracy_score(y_test, pred_log). round(2)

print(f"Logistic Lasso Regression model: accuracy: {log_accuracy}, precision: {log_precision}, recall: {log_recall}, f1 score: {log_f1}, balanced accuracy {log_balanced_acc}")


# In[475]:


#AUC curve
from sklearn.metrics import roc_curve, auc

# probabilites on test set logisitc
log_y_prob = logistic_cv.predict_proba(x_test)[:, 1]
log_fpr, log_tpr, log_thresholds = roc_curve(y_test, log_y_prob)
log_roc_auc = auc(log_fpr, log_tpr)

# decision tree
tree_y_prob = tree_cv.predict_proba(x_test)[:, 1]
tree_fpr, tree_tpr, tree_thresholds = roc_curve(y_test, tree_y_prob)
tree_roc_auc = auc(tree_fpr, tree_tpr)

# bagging trees
bagging_y_prob = bagging_cv.predict_proba(x_test)[:, 1]
bagging_fpr, bagging_tpr, bagging_thresholds = roc_curve(y_test, bagging_y_prob)
bagging_roc_auc = auc(bagging_fpr, bagging_tpr)

# Boosting trees
boosting_y_prob = boosting_tree_cv.predict_proba(x_test)[:, 1]
boosting_fpr, boosting_tpr, boosting_thresholds = roc_curve(y_test, boosting_y_prob)
boosting_roc_auc = auc(boosting_fpr, boosting_tpr)

# Random Forest
rf_y_prob = rf_cv.predict_proba(x_test)[:, 1]
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_y_prob)
rf_roc_auc = auc(rf_fpr, rf_tpr)

# SVM
svm_y_prob = svm_cv.predict_proba(x_test)[:, 1]
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_y_prob)
svm_roc_auc = auc(svm_fpr, svm_tpr)

print(f"Logistic AUC: {log_roc_auc.round(2)}")
print(f"Decision tree AUC: {tree_roc_auc.round(2)}")
print(f"Bagging Decision tree AUC: {bagging_roc_auc.round(2)}")
print(f"Boosting Decision tree AUC: {boosting_roc_auc.round(2)}")
print(f"Random Forest AUC: {rf_roc_auc.round(2)}")
print(f"SVM AUC: {svm_roc_auc.round(2)}")


# In[476]:


# Plot ROC curve
plt.figure()
plt.plot(log_fpr, log_tpr, color='skyblue', lw=1, label='Logistic ROC curve (area = %0.3f)' % log_roc_auc)
plt.plot(tree_fpr, tree_tpr, color='green', lw=1, label='Decision trees ROC curve (area = %0.3f)' % tree_roc_auc)
plt.plot(bagging_fpr, bagging_tpr, color='red', lw=1, label='Bagging Decision trees ROC curve (area = %0.3f)' % bagging_roc_auc)
plt.plot(boosting_fpr, boosting_tpr, color='orange', lw=1, label='Boosting Decision trees ROC curve (area = %0.3f)' % boosting_roc_auc)
plt.plot(rf_fpr, rf_tpr, color='purple', lw=1, label='Random Forest ROC curve (area = %0.3f)' % rf_roc_auc)
plt.plot(svm_fpr, svm_tpr, color='gold', lw=1, label='SVM ROC curve (area = %0.3f)' % svm_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[478]:


# performance metrics table

tb = {
    'Models'   : ['Boosting Decision tree', 'Random Forest', 'Bagging Decision tree', 'SVM', 'Decision tree', 'Logistic Lasso Regression'],
    "Accuracy" : [boost_accuracy, rf_accuracy, bag_accuracy, svm_accuracy, t_accuracy,log_accuracy],
    'Precision': [boost_precision, rf_precision, bag_precision, svm_precision, t_precision, log_precision],
    'Recall'   : [boost_recall,  rf_recall, bag_recall, svm_recall, t_recall,log_recall],
    'F1 Score' : [boost_f1,  rf_f1, bag_f1, svm_f1, t_f1,log_f1],
    'Balanced accuracy' : [boost_balanced_acc, rf_balanced_acc, bag_balanced_acc, svm_balanced_acc, t_balanced_acc ,log_balanced_acc]
}

tb_df = pd.DataFrame(tb)
print(tb_df.to_markdown(index = False))

