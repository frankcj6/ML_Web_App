import matplotlib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn import metrics
from sklearn import naive_bayes
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from time import time
from xgboost import plot_importance
import os
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
import base64


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file, sidebar_file):
    bin_str = get_base64_of_bin_file(png_file)
    bin_str_2 = get_base64_of_bin_file(sidebar_file)
    page_bg_img = '''
    <style>
    body{
    background: url('data:image/jpg; base64, %s');
    }
    .sidebar .sidebar-content('data:image/jgp, %s');
    </style>
    ''' % (bin_str, bin_str_2)

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg('C:/ML Web App/static/background_4.jpg', 'C:/ML Web App/static/sidebar_3.jpg')

st.title('Machine Learning Web App Beta Version')

my_expander = st.beta_expander('View Dataset')
with my_expander:
    st.subheader('Dataset')
    data_file = st.sidebar.file_uploader('Upload your datafile', type=['csv'])
    if data_file is not None:
        st.write(type(data_file))
        file_details = {'FileName:': data_file.name,
                        'Filetype:': data_file.type,
                        'Filesize:': data_file.size}
        st.write(file_details)
        df = pd.read_csv(data_file)
        col = df.columns
        st.dataframe(df)
    else:
        st.write('Upload your datafile from here. ')
target_var = st.sidebar.text_input('Clarify your target variable')

my_expander = st.beta_expander('Target Variable Information')
with my_expander:
    target = str(target_var)
    try:
        x = df.drop(target, axis=1)
        y = df[target]
        st.write('Your target variable is: ', target)
        st.write('Shape of dataset:', x.shape)
        st.write('Number of Classes:', len(np.unique(y)))
    except NameError:
        st.write('Upload your datafile from here. ')
    except KeyError:
        st.write('Please Enter a valid variable name. ')

classifier_name = st.sidebar.selectbox('Select Your Machine Learning Algorithm',
                                       ('KNN', 'SVM', 'Random Forest', 'LightGBM', 'XGBoost', 'Naive Bayes'))


def add_parameter(alg_name):
    params = dict()
    if alg_name == 'SVM':
        C = st.sidebar.slider('C', 0.1, 1.0)
        kernel = st.sidebar.selectbox('kernel type', ['linear', 'rbf', 'poly'])
        if kernel == 'rbf':
            Gamma = st.sidebar.slider('Gamma', 0.01, 1.0)
            params['gammas'] = Gamma
        if kernel == 'poly':
            degree = st.sidebar.slider('degree', 2, 10)
            params['degree'] = degree
        feature = x.columns.values
        selected_feature = st.multiselect('Select two features you are interested in', feature)
        params['selected_feature'] = selected_feature
        params['C'] = C
        params['kernel'] = kernel
    elif alg_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        weights = st.sidebar.selectbox('weights', ('uniform', 'distance'))
        leaf_size = st.sidebar.slider('leaf_size', 30, 100)
        try:
            feature = x.columns.values
            selected_feature = st.multiselect('Select two features you are interested in', feature)
            params['selected_feature'] = selected_feature
        except:
            pass
        params['K'] = K
        params['weights'] = weights
        params['leaf_size'] = leaf_size
    elif alg_name == 'Random Forest':
        n_estimators = st.sidebar.slider('n_estimators', 1, 200)
        criterion = st.sidebar.selectbox('criterion', ('gini', 'entropy'))
        max_features = st.sidebar.selectbox('max_features', ('auto', 'sqrt', 'log2'))
        params['n_estimators'] = n_estimators
        params['criterion'] = criterion
        params['max_features'] = max_features
    elif alg_name == 'LightGBM':
        num_leaves = st.sidebar.slider('num_leaves', 30, 50)
        learning_rate = st.sidebar.slider('learning_rate', 0.01, 0.10)
        n_estimators = st.sidebar.slider('n_estimators', 1, 200)
        objective = st.sidebar.selectbox('objective', ('regression', 'binary', 'multiclass', 'lambdarank'))
        params['num_leaves'] = num_leaves
        params['learning_rate'] = learning_rate
        params['n_estimators'] = n_estimators
        params['objective'] = objective
    elif alg_name == 'XGBoost':
        learning_rate = st.sidebar.slider('learning_rate', 0.01, 0.1)
        max_depth = st.sidebar.slider('max_depth', 3, 10)
        objective = st.sidebar.selectbox('objective', ('binary:logistic', 'multi:softmax', 'multi:softprob'))
        eval_metrics = st.sidebar.selectbox('eval_metrics',
                                            ('rmse', 'mae', 'logloss', 'error', 'merror', 'mlogloss', 'auc'))
        params['learning_rate'] = learning_rate
        params['max_depth'] = max_depth
        params['objective'] = objective
        params['eval_metrics'] = eval_metrics
    elif alg_name == 'Naive Bayes':
        distribution = st.sidebar.selectbox('Please Select Your Algorithm',
                                            ('Gaussian Naive Bayes', 'Multinomial Naive Bayes', 'Complement Naive Bayes'
                                             , 'Bernoulli Naive Bayes', 'Categorical Naive Bayes'))
        try:
            feature_2 = x.columns.values
            selected_feature_2 = st.multiselect('Select two features you are interested in', feature_2)
            params['selected_feature_2'] = selected_feature_2
        except:
            pass

        if distribution == 'Multinomial Naive Bayes':
            alpha = st.sidebar.slider('alpha', 0.00, 1.00)
            prior_bool = st.sidebar.selectbox('prior_bool', (True, False))
            params['alpha'] = alpha
            params['fit_prior'] = prior_bool
        elif distribution == 'Complement Naive Bayes':
            alpha = st.sidebar.slider('alpha', 0.00, 1.00)
            prior_bool = st.sidebar.selectbox('prior_bool', (True, False))
            norm_bool = st.sidebar.selectbox('norm_bool', (False, True))
            params['alpha'] = alpha
            params['fit_prior'] = prior_bool
            params['norm'] = norm_bool
        elif distribution == 'Bernoulli Naive Bayes':
            alpha = st.sidebar.slider('alpha', 0.00, 1.00)
            prior_bool = st.sidebar.selectbox('prior_bool', (True, False))
            binarize_threshold = st.sidebar.selectbox('binary_threshold', (False, True))
            params['binarize'] = binarize_threshold
            if binarize_threshold == True:
                threshold = st.sidebar.text_input('Identify your threshold')
                params['binarize'] = threshold
            params['alpha'] = alpha
            params['fit_prior'] = prior_bool
        elif distribution == 'Categorical Naive Bayes':
            alpha = st.sidebar.slider('alpha', 0.00, 1.00)
            prior_bool = st.sidebar.selectbox('prior_bool', (True, False))
            params['alpha'] = alpha
            params['fit_prior'] = prior_bool
        params['distribution'] = distribution

    return params


params = add_parameter(classifier_name)


def model_build(alg_name, params):
    alg = None
    if alg_name == 'SVM':
        if params['kernel'] == 'linear':
            alg = SVC(C=params['C'], probability=True, kernel='linear')
        elif params['kernel'] == 'rbf':
            alg = SVC(C=params['C'], gamma=params['gammas'], probability=True, kernel=params['kernel'])
        elif params['kernel'] == 'poly':
            alg = SVC(C=params['C'], degree=params['degree'], probability=True, kernel=params['kernel'])
    elif alg_name == 'KNN':
        alg = KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'], leaf_size=params['leaf_size'])
    elif alg_name == 'Random Forest':
        alg = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'],
                                     max_features=params['max_features'], random_state=1234)
    elif alg_name == 'LightGBM':
        alg = lgb.LGBMClassifier(learning_rate=params['learning_rate'], num_leaves=params['num_leaves'],
                                 n_estimators=params['n_estimators'], objective=params['objective'])
    elif alg_name == 'XGBoost':
        alg = XGBClassifier(objective=params['objective'], eval_metrics=params['eval_metrics'],
                            learning_rate=params['learning_rate'], max_depth=params['max_depth'])
    elif alg_name == 'Naive Bayes':
        if params['distribution'] == 'Multinomial Naive Bayes':
            alg = naive_bayes.MultinomialNB(alpha=params['alpha'], fit_prior=params['fit_prior'])
        elif params['distribution'] == 'Gaussian Naive Bayes':
            alg = naive_bayes.GaussianNB()
        elif params['distribution'] == 'Complement Naive Bayes':
            alg = naive_bayes.ComplementNB(alpha=params['alpha'], fit_prior=params['fit_prior'], norm=params['norm'])
        elif params['distribution'] == 'Bernoulli Naive Bayes':
            alg = naive_bayes.BernoulliNB(alpha=params['alpha'], fit_prior=params['fit_prior'],
                                          binarize=params['binarize'])
        elif params['distribution'] == 'Categorical Naive Bayes':
            alg = naive_bayes.CategoricalNB(alpha=params['alpha'], fit_prior=params['fit_prior'])
    return alg


alg = model_build(classifier_name, params)

try:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
    alg.fit(x_train, y_train)
    pred = alg.predict(x_test)
    prob = alg.predict_proba(x_test)[:, 1]

    ### metrics
    accuracy = metrics.accuracy_score(y_test, pred)
    auc = metrics.roc_auc_score(y_test, prob)
    f1 = metrics.f1_score(y_test, pred)
    precision_score = metrics.precision_score(y_test, pred)
    recall_score = metrics.recall_score(y_test, pred)

    st.write(f'Current Machine Learning Algorithm using:  {classifier_name}')
    st.write(f'Accuracy: {accuracy} ')
    st.write(f'AUC: {auc}')
    st.write(f'Precision: {precision_score}')
    st.write(f'Recall: {recall_score}')
    st.write(f'F1 score: {f1}')

except NameError:
    pass


def plot_cm(labels, predictions, p=0.5):
    cm = metrics.confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix @{:.2f}'.format(p))
    plt.ylabel('Actual Results')
    plt.xlabel('Predicted Results')


def plot_roc_curve(prob, y_test, title):
    fp, tp, threshold = metrics.roc_curve(y_test, prob)
    roc_auc = metrics.auc(fp, tp)
    plt.figure(figsize=(5, 5))
    plt.plot(fp, tp, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
    plt.xlabel('False Positive Rate (1- Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.1])
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()


def plot_prec_recall_curve(prob, y_test, title):
    prec, recall, threshold = metrics.precision_recall_curve(y_test, prob)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, prec, label='Precision Recall Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.1])
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()


def plot_feature_importance(alg_name):
    if alg_name == 'Random Forest':
        features = x.columns
        importance = alg.feature_importances_
        indices = np.argsort(importance)
        plt.figure(figsize=(5, 5))
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    elif alg_name == 'LightGBM':
        features = x.columns
        importance = alg.feature_importances_
        indices = np.argsort(importance)
        plt.figure(figsize=(5, 5))
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    elif alg_name == 'XGBoost':
        features = x.columns
        importance = alg.feature_importances_
        indices = np.argsort(importance)
        plt.figure(figsize=(5, 5))
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')

    elif alg_name == 'KNN' or 'SVM':
        try:
            plt.figure(figsize=(5, 5))
            feature_num = len(x.columns)
            feature_1_idx = x.columns.get_loc(str(params['selected_feature'][0]))
            feature_2_idx = x.columns.get_loc(str(params['selected_feature'][1]))
            feature_values = {i: 1 for i in range(0, feature_num) if i != feature_1_idx and i != feature_2_idx}
            feature_width = {i: 1 for i in range(0, feature_num) if i != feature_1_idx and i != feature_2_idx}
            plot_decision_regions(x.values,
                                  y.values,
                                  clf=alg,
                                  feature_index=[feature_1_idx, feature_2_idx],
                                  filler_feature_values=feature_values,
                                  filler_feature_ranges=feature_width,
                                  legend=2)
            plt.xlabel(str(params['selected_feature'][0]))
            plt.ylabel(str(params['selected_feature'][1]))
            if alg_name == 'KNN':
                plt.title('KNN Classifier with K=' + str(params['K']))
            else:
                plt.title('SVM Classifier')
        except:
            pass


st.set_option('deprecation.showPyplotGlobalUse', False)
try:

    cols = st.beta_columns(2)
    prec_recal_plot = plot_prec_recall_curve(y_test, pred, 'Precision Recall Curve (w threshold)')
    cols[0].pyplot(prec_recal_plot)
    roc_graph = plot_roc_curve(prob, y_test, 'ROC Curve (w threshold)')
    cols[1].pyplot(roc_graph)
    fig = plot_cm(y_test, pred)
    cols[0].pyplot(fig)
    if classifier_name == 'Naive Bayes':
        x_ = x[[str(params['selected_feature_2'][0]), str(params['selected_feature_2'][1])]]
        x_ = x_.to_numpy()
        y_ = y.to_numpy()
        plt.scatter(x_[:, 0], x_[:, 1], c=y_, s=50, cmap='RdBu')
        plt.title('Naive Bayes Model', size=14)
        plt.xlabel(str(params['selected_feature_2'][0]))
        plt.ylabel(str(params['selected_feature_2'][1]))
        cols[1].pyplot()
    else:
        importance_plot = plot_feature_importance(classifier_name)
        cols[1].pyplot(importance_plot)
except:
    pass
