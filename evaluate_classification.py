
# coding: utf-8

# In[5]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def score_df(y_train, y_test, y_pred_train, y_pred_test, average='binary'):
    if len(y_train) != len(y_pred_train): raise Exception('Lengths of true and predicted for train do not match.')
    if len(y_pred_test) != len(y_pred_test): raise Exception('Lengths of true and predicted for test do not match.')
    score_2darray = [                      [                       len(y_),
                      accuracy_score(y_, y_pred_), 
                      precision_score(y_, y_pred_, average=average), 
                      recall_score(y_, y_pred_, average=average), 
                      f1_score(y_, y_pred_, average=average) \
                     ] \
                     for (y_, y_pred_) in [(y_train, y_pred_train), (y_test, y_pred_test)] \
                    ]
    score_df = pd.DataFrame(score_2darray,
                            index = ['train', 'test'], 
                            columns = ['support', 'accuracy', 'precision', 'recall', 'f1'] )
    return score_df


# In[6]:


from sklearn.metrics import confusion_matrix

def conf_mat_df(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    num_class = len(conf_mat)
    true_labels = [f'True_{i}' for i in range(num_class)]
    pred_labels = [f'Pred_{i}' for i in range(num_class)]
    conf_mat_df = pd.DataFrame(conf_mat, index = true_labels, columns = pred_labels )
    return conf_mat_df


# In[8]:


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets, preprocessing
    from sklearn.model_selection import train_test_split, GridSearchCV
    import pandas as pd
    from IPython.display import display
    
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=33)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    params = {'C': [0.0001, 0.001, 0.01]}
    model = GridSearchCV(LogisticRegression(), params, cv=2, return_train_score=False, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f'Best parameter for logistic regression:\n{model.best_params_}')
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print('\nScore Table:')
    display(score_df(y_train, y_test, y_pred_train, y_pred_test, average='macro'))
    print('\nConfusion Matrix for Train:')
    display(conf_mat_df(y_train, y_pred_train))
    print('\nConfusion Matrix for Test:')
    display(conf_mat_df(y_test, y_pred_test))
    

