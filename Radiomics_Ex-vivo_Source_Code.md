
# Radiomics versus Visual and Histogram-based Assessment to Identify Atheromatous  Lesions on Coronary CT Angiography: an Ex-vivo Study -- SOURCE CODE

## Initialize and load data


```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = 'C:/Users/Marton_Kolossvary/Projects/Radiomics_ExVivo/'
file_name = 'Radiomics_data.csv'
df = pd.read_csv(folder + file_name)
df_vol = pd.read_csv(folder + 'Volumetric_data.csv')
df = pd.concat((df, df_vol), axis=1)

#SELECT ONLY PLAQUES VISIBLE ON CT, WHICH ARE NOT ONLY CALCIUM
df_p = df.loc[(df['CTR2_Plaqcat'] != 0) & (df['CTR2_Plaqcat'] != 3)]
```

## Data encoding indexes


```python
#BINARY DEPENDENT VARIABLE
Advanced = 36 # Advanced Lesion, 0: 'AIT'; 'PIT'; 'Fib'; 1: 'EFA'; 'LFA'; 'TCFA'

#VISUAL PREDICTOR
CT_PAP = 38 #0:Homogeneous; 1:Heterogeneous; 2:NRS

#HISTOGRAM-BASED METHOD PARAMETERS
CT_30 = 1976 #LAP AREA
CT_Average = 55

#RADIOMICS DATASET
start = 56
end = 1975
```

## Train and Test split of data


```python
from sklearn.model_selection import train_test_split
rd_state = 42
df_outcome = df_p.iloc[:, Advanced]
X_train, X_test, y_train, y_test = train_test_split(df_p, df_outcome,  test_size=0.25, random_state=rd_state)

#RADIOMICS SUBSET
X_train_orig = X_train #Save whole dataset
X_test_orig = X_test
X_train = X_train_orig.iloc[:, start:end]
X_test = X_test_orig.iloc[:, start:end]

#LOW ATTENUATION PLAQUE AREA SUBSET
X_30_train_con = X_train_orig.iloc[:, CT_30]
X_30_train = X_30_train_con.values.reshape(-1, 1)

X_30_test_con = X_test_orig.iloc[:, CT_30]
X_30_test = X_30_test_con.values.reshape(-1, 1)

#AVERAGE HU ATTENUATION SUBSET
X_mean_train_con = X_train_orig.iloc[:, CT_Average]
X_mean_train = X_mean_train_con.values.reshape(-1, 1)

X_mean_test_con = X_test_orig.iloc[:, CT_Average]
X_mean_test = X_mean_test_con.values.reshape(-1, 1)

#PLAQUE ATTENUATION PATTERN SUBSET
X_PAP_train_con = X_train_orig.iloc[:, CT_PAP]
X_PAP_train = X_PAP_train_con.values.reshape(-1, 1)

X_PAP_test_con = X_test_orig.iloc[:, CT_PAP]
X_PAP_test = X_PAP_test_con.values.reshape(-1, 1)
```

## Pipeline definition


```python
from sklearn.pipeline import Pipeline, FeatureUnion

# 1) ZERO VARIANCE EXCLUSION
from sklearn.feature_selection import VarianceThreshold
exc_0 = VarianceThreshold()

# 2) ROBUST SCALE DATA
from sklearn.preprocessing import RobustScaler
scl = RobustScaler()

# 3) REDUCE PARAMETER NUMBER BASED ON FPR OR FWE
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectFpr, SelectFwe
exc_stat_fpr = SelectFpr(score_func=f_classif)
exc_stat_fwe = SelectFwe(score_func=f_classif)
exc_uni = FeatureUnion([('FPR', exc_stat_fpr), ('FWE', exc_stat_fwe)])

# 4) PCA or KernelPCA
from sklearn.decomposition import PCA
pca = PCA()

#CREATE PIPELINE AND DISTRIBUTIONS FOR HYPERPARAMETERS
preprocc_pipeline = Pipeline(steps=[("Exc_0", exc_0),
                                    ("Scale", scl),
                                    ("Uni_sel", exc_uni),
                                    ("PCA", pca)])

#DEFINE DISTRIBUTIONS FOR RANDOM SEARCH
from scipy.stats import uniform
dist_alpha = uniform(loc=0, scale=0.1) #ALPHA LEVEL FOR UNIVARIATE SELECTION
dist_exp_var = uniform(loc=0.75, scale=0.24) #PROPORTION OF EXPLAIND VARIANCE FOR PCA

#RANDOMIZED GRID SEARCH PARAMETERS
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
n_try = 1000 #RandomSearchCV tries
cv_number = 5 #N-fold
cv_type = StratifiedKFold(n_splits=cv_number)
```

## Accuracy measures


```python
#ACCURACY MEASURES
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, roc_auc_score, roc_curve, auc
from scipy import interp
scoring_params = 'roc_auc'

#CONFUSION MATRIX FUNCTION DEFINITION
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
#PLOT FUNCTION FOR ROC CURVES AND AUC VALUES
def plot_ROCs(y, predictions):
    roc_auc = dict()
    tpr_interp = dict()
    mean_fpr = np.linspace(0, 1, 101)
    
    RESULTS_FINAL = pd.DataFrame(np.concatenate([y.values.reshape(-1, 1),
                                                 predictions],
                                axis=1))
    
    for i in range(1, RESULTS_FINAL.shape[1], 1):
        fpr, tpr, thresholds = roc_curve(RESULTS_FINAL.iloc[:, 0], RESULTS_FINAL.iloc[:, i])
        tpr_interp[i] = interp(mean_fpr, fpr, tpr)
        roc_auc[i] = auc(fpr, tpr)
    
    #PLOT CURVES
    fig = plt.figure()
    lw = 2
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i in range(1, RESULTS_FINAL.shape[1], 1):
        fig = plt.plot(mean_fpr, tpr_interp[i], color=colors[i],
             lw=lw, label='Model-' +  str(i) + ' (area = ' + str(round(roc_auc[i], 2)) + ')')

    fig = plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    fig = plt.xlim([0.0, 1.0])
    fig = plt.ylim([0.0, 1.05])
    fig = plt.xlabel('False Positive Rate')
    fig = plt.ylabel('True Positive Rate')
    fig = plt.title('ROC curves to predict FFR values')
    fig = plt.legend(loc="lower right")
    
    folder = 'C:/Users/Marton_Kolossvary/Projects/Radiomics_ExVivo/'
    file_name = 'ROC_'
    plt.savefig(folder + file_name + str(RESULTS_FINAL.shape[1] -1) + '.pdf', bbox_inches='tight')
```

# Plaque attenuation pattern as predictor


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score

clf_PAP = LogisticRegression().fit(X_PAP_train, y_train)
predicted_PAP = clf_PAP.predict_proba(X_PAP_test)
print("The ROC AUC using PAP as a classifier is:", roc_auc_score(y_test, predicted_PAP[:, 1]))
```

# Low attenuation plaque as predictor


```python
clf_30 = LogisticRegression().fit(X_30_train, y_train)
predicted_30 = clf_30.predict_proba(X_30_test)
print("The ROC AUC using low attenuation plaque area (<30HU) as a classifier is:", roc_auc_score(y_test, predicted_30[:, 1]))
```

# Average HU as predictor


```python
clf_mean = LogisticRegression().fit(X_mean_train, y_train)
predicted_mean = clf_mean.predict_proba(X_mean_test)
print("The ROC AUC using mean as a classifier is:", roc_auc_score(y_test, predicted_mean[:, 1]))
```

# ML MODEL TRAINING

## Logistic regression


```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
model = Pipeline(steps=[('preprocessing', preprocc_pipeline), ('modeling', clf)])

#CREATE MODEL SPECIFIC DISTRIBUTION FOR RANDOMIZED SEARCH
from scipy.stats import expon, uniform
dist_C = uniform(loc=0, scale=1000)
dist_penalty = ["l1", "l2"]

param_dist = {"preprocessing__Uni_sel__FPR__alpha": dist_alpha,
              "preprocessing__Uni_sel__FWE__alpha": dist_alpha,
              "preprocessing__PCA__n_components": dist_exp_var,
              "modeling__C": dist_C,
              "modeling__penalty": dist_penalty 
             }

rand_clf_logreg = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_try, scoring=scoring_params, n_jobs=-1, cv=cv_type, refit=True,
                                         error_score=0, verbose=1, return_train_score=True, random_state=rd_state).fit(X_train, y_train)

print("The AUC during CV using Logistic regression is: ", rand_clf_logreg.best_score_)
```

## K-nearest neighbors


```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
model = Pipeline(steps=[('preprocessing', preprocc_pipeline), ('modeling', clf)])

param_dist = {"preprocessing__Uni_sel__FPR__alpha": dist_alpha,
              "preprocessing__Uni_sel__FWE__alpha": dist_alpha,
              "preprocessing__PCA__n_components": dist_exp_var,
              "modeling__n_neighbors": np.arange(1, 5, 1),
              "modeling__weights": ["uniform", "distance"],
              "modeling__p": [1, 2]}

rand_clf_knn = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_try, scoring=scoring_params, n_jobs=-1, cv=cv_type, refit=True,
                                         error_score=0, verbose=1, return_train_score=True, random_state=rd_state).fit(X_train, y_train)

print("The AUC during CV using KNN is: ", rand_clf_knn.best_score_)
```

## Random forests


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
model = Pipeline(steps=[('preprocessing', preprocc_pipeline), ('modeling', clf)])

dist_uniform1 = np.arange(1, 200, 1)
dist_uniform2 = np.arange(1, 10, 1)

param_dist = {"preprocessing__Uni_sel__FPR__alpha": dist_alpha,
              "preprocessing__Uni_sel__FWE__alpha": dist_alpha,
              "preprocessing__PCA__n_components": dist_exp_var,
              "modeling__n_estimators": dist_uniform1,
              "modeling__max_depth": dist_uniform2,
              "modeling__bootstrap": [True, False]}


rand_clf_RF = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_try, scoring=scoring_params, n_jobs=-1, cv=cv_type, refit=True,
                                         error_score=0, verbose=1, return_train_score=True, random_state=rd_state).fit(X_train, y_train)

print("The AUC during CV using RF is: ", rand_clf_RF.best_score_)
```

## Least-angles regression


```python
from sklearn.linear_model import Lars
clf = Lars()
model = Pipeline(steps=[('preprocessing', preprocc_pipeline), ('modeling', clf)])

dist_n_nonzero_coefs = np.arange(1, 50, 1)

param_dist = {"preprocessing__Uni_sel__FPR__alpha": dist_alpha,
              "preprocessing__Uni_sel__FWE__alpha": dist_alpha,
              "preprocessing__PCA__n_components": dist_exp_var,
              "modeling__n_nonzero_coefs": dist_n_nonzero_coefs}

rand_clf_Lars = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_try, scoring=scoring_params, n_jobs=-1, cv=cv_type, refit=True,
                                         error_score=0, verbose=1, return_train_score=True, random_state=rd_state).fit(X_train, y_train)

print("The AUC during CV using Lars is: ", rand_clf_Lars.best_score_)
```

## Naïve Bayes


```python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
model = Pipeline(steps=[('preprocessing', preprocc_pipeline), ('modeling', clf)])


param_dist = {"preprocessing__Uni_sel__FPR__alpha": dist_alpha,
              "preprocessing__Uni_sel__FWE__alpha": dist_alpha,
              "preprocessing__PCA__n_components": dist_exp_var}

rand_clf_GNB = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_try, scoring=scoring_params, n_jobs=-1, cv=cv_type, refit=True,
                                         error_score=0, verbose=1, return_train_score=True, random_state=rd_state).fit(X_train, y_train)

print("The AUC during CV using Naive Bayes is: ", rand_clf_GNB.best_score_)
```

## Gaussian process classifier


```python
from sklearn.gaussian_process import GaussianProcessClassifier
clf = GaussianProcessClassifier()
model = Pipeline(steps=[('preprocessing', preprocc_pipeline), ('modeling', clf)])


param_dist = {"preprocessing__Uni_sel__FPR__alpha": dist_alpha,
              "preprocessing__Uni_sel__FWE__alpha": dist_alpha,
              "preprocessing__PCA__n_components": dist_exp_var}

rand_clf_GP = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_try, scoring=scoring_params, n_jobs=-1, cv=cv_type, refit=True,
                                         error_score=0, verbose=1, return_train_score=True, random_state=rd_state).fit(X_train, y_train)

print("The AUC during CV using Gaussian Process is: ", rand_clf_GP.best_score_)
```

## Decision trees


```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_features = None)
model = Pipeline(steps=[('preprocessing', preprocc_pipeline), ('modeling', clf)])


from scipy.stats import uniform
dist_criterion = ["gini", "entropy"]
dist_splitter = ["best", "random"]
dist_max_depth = np.arange(2, 20, 1)
min_impurity_decrease = uniform(loc = 0, scale=10)

param_dist = {"preprocessing__Uni_sel__FPR__alpha": dist_alpha,
              "preprocessing__Uni_sel__FWE__alpha": dist_alpha,
              "preprocessing__PCA__n_components": dist_exp_var,
              "modeling__criterion": dist_criterion,
              "modeling__splitter": dist_splitter,
              "modeling__max_depth": dist_max_depth,
              "modeling__min_samples_split": dist_max_depth,
              "modeling__min_samples_leaf": dist_max_depth,
              "modeling__min_impurity_decrease": min_impurity_decrease}

rand_clf_DT = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_try, scoring=scoring_params, n_jobs=-1, cv=cv_type, refit=True,
                                         error_score=0, verbose=1, return_train_score=True, random_state=rd_state).fit(X_train, y_train)

print("The AUC during CV using Decision trees is: ", rand_clf_DT.best_score_)
```

## Deep neural networks 


```python
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(learning_rate  = "adaptive")
model = Pipeline(steps=[('preprocessing', preprocc_pipeline), ('modeling', clf)])


from scipy.stats import uniform
dist_hidden_layer_sizes = [x for x in itertools.product(np.arange(20, 60, 1),repeat=5)]
dist_alpha_m = uniform(loc=0, scale=1)
dist_learning_rate_init = uniform(loc=0, scale=1)

param_dist = {"preprocessing__Uni_sel__FPR__alpha": dist_alpha,
              "preprocessing__Uni_sel__FWE__alpha": dist_alpha,
              "preprocessing__PCA__n_components": dist_exp_var,
              "modeling__hidden_layer_sizes": dist_hidden_layer_sizes,
              "modeling__alpha": dist_alpha_m,
              "modeling__learning_rate_init": dist_learning_rate_init}

rand_clf_MLP = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_try, scoring=scoring_params, n_jobs=-1, cv=cv_type, refit=True,
                                         error_score=0, verbose=1, return_train_score=True, random_state=rd_state).fit(X_train, y_train)

print("The AUC during CV using MLP is: ", rand_clf_MLP.best_score_)
```

# Application of only the best ML model (LARS) during training to validation set


```python
#SUMMARY OF RESULTS
print("The AUC during CV using Logistic regression is: ", rand_clf_logreg.best_score_)
print("The AUC during CV using KNN is: ", rand_clf_knn.best_score_)
print("The AUC during CV using RF is: ", rand_clf_RF.best_score_)
print("The AUC during CV using Lars is: ", rand_clf_Lars.best_score_)
print("The AUC during CV using Naive Bayes is: ", rand_clf_GNB.best_score_)
print("The AUC during CV using Gaussian Process is: ", rand_clf_GP.best_score_)
print("The AUC during CV using Decision trees is: ", rand_clf_DT.best_score_)
print("The AUC during CV using MLP is: ", rand_clf_MLP.best_score_)

predicted_Lars = rand_clf_Lars.predict_proba(X_test)
print("APPLICATION OF BEST ML MODEL (LARS) ON TRAINING TO THE VALIDATION DATASET, AUC = ", roc_auc_score(y_test, predicted_Lars))
```
