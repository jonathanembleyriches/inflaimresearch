import random

import pandas as pd
import numpy as np
import os

import sklearn.metrics
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold, train_test_split, cross_val_score, LeaveOneOut
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Masking
from sklearn.metrics import accuracy_score,hamming_loss
import matplotlib.pyplot as plt
from numpy import NAN
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
'''
format-two-noar: re-ordered agefu
format-three-noar: scaled time added for each instance
format-four-noar: increment time added for each instance

'''





def reorder_agefu(df):
    formatted_df = df.iloc[0:0,:].copy()
    start_i = 0
    temp = df.iloc[0:0, :].copy()
    print(temp)
    for i in range(1, df.shape[0]+1):



        temp = temp.append(df.iloc[i-1])

        if i >= df.shape[0]:
            temp.sort_values(by=['agefu'], inplace=True)
            formatted_df = formatted_df.append(temp)

        elif df.loc[i, 'regno'] != df.loc[i - 1, 'regno']:
            #sort and add to main df
            temp.sort_values(by=['agefu'], inplace=True)
            if df.loc[i-1, 'regno'] ==900171:
                print(temp)
            formatted_df = formatted_df.append(temp, sort=False)
            #formatted_df = pd.concat([formatted_df, temp], sort=False)
            temp = df.iloc[0:0, :].copy()
            print(f' {i}/{df.shape[0]}  :  {formatted_df.shape[0]}')


    #formatted_df.to_csv('format-two-noar.csv', index=False)
    return formatted_df


def create_fill_time_column_scale(df):
    df = df.assign(time=pd.Series(np.random.randn(len(df.index))).values)
    df['time'] = df.apply(lambda x: x['agefu']-x['ageons'] , axis=1)
    #df.to_csv('format-three-noar.csv', index=False)
    return df

def create_fill_time_column_order(df):
    df = df.assign(time=pd.Series(np.random.randn(len(df.index))).values)
    prev=0
    for i in range(1, df.shape[0] + 1):
        df.loc[i-1, 'time'] = prev + 1
        prev+=1

        if i<df.shape[0] and df.loc[i, 'regno'] != df.loc[i - 1, 'regno']:
            prev=0
    #df.to_csv('format-four-noar.csv', index=False)
    return df

def modify_fill_time_column_order(df):

    #df = df.assign(time=pd.Series(np.random.randn(len(df.index))).values)
    prev=0
    for i in range(1, df.shape[0] + 1):
        df.loc[i-1, 'time'] = prev + 1
        prev+=1

        if i<df.shape[0] and df.loc[i, 'regno'] != df.loc[i - 1, 'regno']:
            prev=0
    #df.to_csv('format-four-noar.csv', index=False)
    return df
def add_pad(count,df,reg_no):
    arr = [[0,reg_no,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'caucasian','male','no',0,0,0,0,0,0,0,0,0,'remission',0,0,0,0,0,0,0]]
    arr = np.asarray(arr)
    for i in range(count):
        df = df.append(pd.DataFrame(arr, columns=df.columns), ignore_index=True)
    return df

def set_seq_length(df):
    seq_length = 15
    count = 0
    pos = 1
    x = True
    length = df.shape[0]
    while(x):
        pos += 1
        count+=1
        reg_no = df.loc[pos - 1, "regno"]
        print(f'{pos} / {length} : {reg_no}')
        if pos >= length:
            if count < seq_length:
                # add rows here
                rows_to_add = seq_length - count
                df = add_pad(rows_to_add, df,reg_no)
                x=False
                break
        if df.loc[pos, 'regno'] != reg_no:
            if count < seq_length:
                #add rows here
                rows_to_add = seq_length-count

                df = add_pad(rows_to_add,df,reg_no)
                #pos+=rows_to_add
            #set pos to the correct place
            count = 0
    return df

def filter_do_data(df):
    df['Days_on_Steroids'] = df['Days_on_Steroids'].apply(lambda x: np.nan if x< 0 else x)
    df['Days_on_DMARDs'] = df['Days_on_DMARDs'].apply(lambda x: np.nan if x< 0 else x)
    df['Days_on_Biologics'] = df['Days_on_Biologics'].apply(lambda x: np.nan if x< 0 else x)

    return df


def full_process_scale(df):
    df = reorder_agefu(df)
    df = create_fill_time_column_scale(df)
    df = filter_do_data(df)
    df.to_csv('NOAR-proc-scale.csv', index=False)

def full_process_increment(df):
    print('Re-ordering ')
    df = reorder_agefu(df)
    df = create_fill_time_column_order(df)
    #df = filter_do_data(df)
    df.to_csv('NOAR-proc-increment.csv', index=False)
    return df


def load_proc_inc():
    return pd.read_csv("NOAR-proc-increment.csv")


def drop_columns(df):
    columns = ['Unnamed: 0','regno','dobyear','ageons','agefu','censored_status','dead_status','died','X1','X2.1','Z1','clus','id','Imputation']
    df = df.drop(columns=columns)
    return df



def seperate_input_labels(df):
    return df.iloc[:,13:],df.iloc[:,0:13]


def get_model_rnn(n_inputs, n_outputs,x,y):
    # create model
    model = Sequential()
    #model.add(Dense((n_inputs*2/3)+n_outputs, input_dim=n_inputs, kernel_initializer='he_uniform',activation='relu'))
    #model.add(Dense((n_inputs*2/3)+n_outputs,activation='relu'))

    #model.add(LSTM((13),batch_input_shape=(None,15,13),return_sequences=True))
    model.add(LSTM(13, dropout=0.2, recurrent_dropout=0.2,input_dim=13))

    #model.add(LSTM((13),return_sequences=False))

    model.add(Dense(n_outputs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
def get_model(n_inputs, n_outputs,x,y):
    # create model
    model = Sequential()
    model.add(Dense((n_inputs*2/3)+n_outputs, input_dim=n_inputs, kernel_initializer='he_uniform',activation='relu'))
    model.add(Dense((n_inputs*2/3)+n_outputs,activation='relu'))

    #model.add(LSTM((13),batch_input_shape=(None,15,13),return_sequences=True))
    #model.add(LSTM(13, dropout=0.2, recurrent_dropout=0.2,input_dim=13))

    #model.add(LSTM((13),return_sequences=False))

    model.add(Dense(n_outputs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
def get_model2():
    # create model
    model = Sequential()
    #model.add(Dense((n_inputs*2/3)+n_outputs, input_dim=n_inputs, kernel_initializer='he_uniform',activation='relu'))
    #model.add(Dense((n_inputs*2/3)+n_outputs,activation='relu'))

    model.add(LSTM((13),batch_input_shape=(None,15,13),return_sequences=True))
    model.add(LSTM((13),return_sequences=False))
    model.add(Dense(13))
    model.add(Dense(13, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
def get_model3():
    # create model
    model = Sequential()
    #model.add(Dense((n_inputs*2/3)+n_outputs, input_dim=n_inputs, kernel_initializer='he_uniform',activation='relu'))
    #model.add(Dense((n_inputs*2/3)+n_outputs,activation='relu'))

    model.add(LSTM((13),batch_input_shape=(None,None,1),return_sequences=True))
    model.add(LSTM((13),return_sequences=False))
    model.add(Dense(13))
    model.add(Dense(13, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
def get_model_nn():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=13, kernel_initializer='he_uniform',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(13, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    return model
def get_model_nn_param():
    # create model
    model = Sequential()
    model.add(Dense((28*2/3)+13, input_dim=28, kernel_initializer='he_uniform',activation='relu'))
    #model.add(Dense(10, input_dim=21, kernel_initializer='he_uniform',activation='relu'))

    model.add(Dropout(0.2))
    #model.add(Dense((2*(28*2/3)+13), activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense((2 * (28 * 2 / 3) + 13), activation='relu'))#rem these to get 81
    #model.add(Dropout(0.2))
    model.add(Dense(13, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    print('')
    return model

def build_model(inputs,outputs,hidden_layer_count,node_count):
    # create model
    model = Sequential()
    model.add(Dense(node_count, input_dim=inputs, kernel_initializer='he_uniform',activation='relu'))
    model.add(Dropout(0.2))
    if(hidden_layer_count>0):
        for i in range(hidden_layer_count):
            model.add(Dense(node_count, activation='relu'))
            model.add(Dropout(0.2))


    model.add(Dense(outputs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    #print(model.summary())
    #print('')
    return model

def create_groupings(df):
    print(df)
    data = []
    target = []
    data_columns = ['ethnicity','gender','weight','bmi','crpscore','Days_on_Steroids',
                    'Days_on_DMARDs','Days_on_Biologics','esr_score','das28','smokestatus',
                    'disease_activity','time']
    target_columns = ['ang_status','bp_status','htattk_status','htfail_status','stroke_status','diab_status',
                      'liver_status','kidney_status','cancer_status','disc_status','depress_status',
                      'lung_status','glauc_status']

    temp_target = []
    temp_data = []

    for i in range(1, df.shape[0] + 1):



        #add data to temps
        temp_data.append(df.loc[i-1, data_columns].values.flatten().tolist())
        temp_target.append(df.loc[i-1, target_columns].values.flatten().tolist())

        print(i)
        if i >= df.shape[0]:
            data.append(temp_data)
            #target.append(temp_target)
            target.append(df.loc[i-1, target_columns].values.flatten().tolist())
        elif df.loc[i, 'regno'] != df.loc[i - 1, 'regno']:

            data.append(temp_data)
            #target.append(temp_target)
            target.append(df.loc[i-1, target_columns].values.flatten().tolist())

            #reset temps
            temp_data = []
            temp_target = []
            print(f' {i}/{df.shape[0]}  :  {df.shape[0]}')

    return data, target

def create_groupings_pad(df):
    data = []
    target = []

    temp_data = []
    temp_target = []

    empty_data =[]
    empty_target = []
    count = 0
    seq_length = 15
    for i in range(1, df.shape[0] + 1):


        count +=1
        #add data to temps
        temp_data.append(df.loc[i-1, df.columns].values.flatten().tolist())



        if i >= df.shape[0]:
            data.append(temp_data)
            target.append(temp_target)


        elif df.loc[i, 'regno'] != df.loc[i - 1, 'regno']:
            if count < seq_length:
                rows_to_add = seq_length - count
                for kk in range(rows_to_add):

                    temp_data.insert(0,[0, df.loc[i - 1, 'regno'], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'caucasian', 'male', 'no', 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 'remission', 0, 0, 0, 0, 0, 0, 0])
            count = 0

            data.append(temp_data)

            temp_data = []
            print(f' {i}/{df.shape[0]}  :  {df.shape[0]}')
    newD = np.asarray(data)
    return pd.DataFrame(newD,
             columns=['Unnamed: 0', 'regno', 'dobyear', 'ageons', 'agefu', 'ang_status', 'bp_status', 'htattk_status', 'htfail_status', 'stroke_status', 'diab_status', 'liver_status', 'kidney_status', 'cancer_status', 'disc_status', 'depress_status', 'lung_status', 'glauc_status', 'censored_status', 'dead_status', 'ethnicity', 'gender', 'died', 'weight', 'bmi', 'crpscore', 'Days_on_Steroids', 'Days_on_DMARDs', 'Days_on_Biologics', 'esr_score', 'das28', 'smokestatus', 'disease_activity', 'X1', 'X2.1', 'Z1', 'clus', 'id', 'Imputation', 'time'])
    #return data, target

def remove_zero(df):
    target_columns = ['ang_status', 'bp_status', 'htattk_status', 'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status', 'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']
    count = 0
    to_remove_count = 0
    seq_length = 15
    to_remove = []
    for i in range(1, df.shape[0] + 1):



        #add data to temps
        if 1 in df.loc[i-1, target_columns].values.flatten().tolist():
            count+=1



        if i >= df.shape[0]:
            if count == 0:
                to_remove.append(df.loc[i-1, 'regno'])
                to_remove_count+=1


        elif df.loc[i, 'regno'] != df.loc[i - 1, 'regno']:
            if count == 0:
                to_remove.append(df.loc[i-1, 'regno'])
                to_remove_count+=1
            count = 0

    print(f'Removing {to_remove_count} rows out of {df.shape[0]}')
    return to_remove

def calc_longest_seq(df):
    current_count = 0
    highest_count = 0
    for i in range(1, df.shape[0] + 1):

        current_count += 1

        if i >= df.shape[0]:
            if current_count > highest_count:
                highest_count = current_count

        elif df.loc[i, 'regno'] != df.loc[i - 1, 'regno']:

            if current_count > highest_count:
                highest_count = current_count
            current_count = 0

    print(f'Highest seq length {highest_count}')




def rnn_load():
    df = pd.read_csv("padded-front-2.csv")  # padtesttest
    todel = remove_zero(df)
    for n in todel:
        df.drop(df.loc[df['regno'] == n], inplace=True)
    #tr, te = split_leave_one_out(df, 105261)
    #print(te)
    to = ['ethnicity', 'gender', 'died', 'disease_activity']

    # convert into categorical variables
    for n in to:
        df[n] = pd.Categorical(df[n])
        df[n] = df[n].cat.codes

    # convert to nump array (with groupings for recurrent)
    data, targ = create_groupings(df)
    data = np.asarray(data)
    targ = np.asarray(targ)
    targ = targ.astype(int)
    print(np.shape(targ))
    xtrain, xtest, ytrain, ytest = train_test_split(data, targ, test_size=0.2, random_state=4)

    model = get_model(13, 13, ytrain, ytrain)
    # model = KerasClassifier(build_fn=get_model2, epochs=150,verbose=0)

    # cv = LeaveOneOut()
    # cs=cross_val_score(model, data, targ,scoring='neg_mean_squared_error',

    #                         cv=cv, n_jobs=-1)
    # print(f'cross val score {cs}')

    hist = model.fit(xtrain, ytrain, epochs=75, validation_data=(xtest, ytest))

    yhat = model.predict(xtest)
    print(np.abs(yhat.round()))
    acc = accuracy_score(ytest, np.abs(yhat.round()))
    print(f'accuracu {acc}')
    pd.DataFrame(np.abs(yhat.round())).to_csv("results_pred2.csv")
    pd.DataFrame(ytest).to_csv("results_actual2.csv")
    plt.plot(hist.history['loss'])
    plt.show()


def split_leave_one_out(df,reg_no):
    test = df.loc[df['regno'] == reg_no]
    train = df.loc[df['regno'] != reg_no]
    return train,test

def cross_validate(df):
    reg_nos = df['regno'].unique()
    model = get_model_nn()
    accuracy = 0
    total_length = len(reg_nos)
    for i,reg in  enumerate(reg_nos):
        print(f'fold {i}/{total_length}')
        train,test = split_leave_one_out(df,reg)
        train = drop_columns(train)
        test = drop_columns(test)
        xtrain,ytrain = seperate_input_labels(train)
        xtest,ytest = seperate_input_labels(test)
        xtrain = np.asarray(xtrain).astype(np.float32)
        ytrain = np.asarray(ytrain).astype(np.int8)
        xtest = np.asarray(xtest).astype(np.float32)
        ytest = np.asarray(ytest).astype(np.int8)

        model.fit(xtrain, ytrain,batch_size=32, epochs=50, workers=24,use_multiprocessing=True)

        yhat = model.predict(xtest)
        local_acc = accuracy_score(ytest, np.abs(yhat.round()))
        print(f'local acc {local_acc}')
        accuracy += local_acc
    accuracy = accuracy/len(reg_nos)
    print(f'accuracy: {accuracy}')

def train_test_split(df):
    reg_nos = df['regno'].unique()
    random.shuffle(reg_nos)
    model = get_model_nn()
    percentage = 0.8
    index = round(percentage*len(reg_nos))
    print(index)
    train_ids = reg_nos[:index]
    test_ids = reg_nos[index:]
    train = df.loc[df['regno'].isin(train_ids)]
    test = df.loc[df['regno'].isin(test_ids)]
    train = drop_columns(train)
    test = drop_columns(test)
    xtrain, ytrain = seperate_input_labels(train)
    xtest, ytest = seperate_input_labels(test)
    xtrain = np.asarray(xtrain).astype(np.float32)
    ytrain = np.asarray(ytrain).astype(np.int8)
    xtest = np.asarray(xtest).astype(np.float32)
    ytest = np.asarray(ytest).astype(np.int8)

    hist = model.fit(xtrain, ytrain, batch_size=32, epochs=50, workers=24, use_multiprocessing=True)

    yhat = model.predict(xtest)
    local_acc = accuracy_score(ytest, np.abs(yhat.round()))
    print(f'local acc {local_acc}')
    pd.DataFrame(np.abs(yhat.round())).to_csv("pred-new.csv")
    pd.DataFrame(ytest).to_csv("actual-new.csv")
    plt.plot(hist.history['loss'])
    plt.show()


def format_and_split_data(df):
    # format data inside
    to = ['ethnicity', 'gender', 'died', 'disease_activity']
    for n in to:
        df[n] = pd.Categorical(df[n])
        df[n] = df[n].cat.codes

    # seperate dataframe into data and targ
    data_columns = ['ageons','agefu','ethnicity','gender','weight','bmi','crpscore','Days_on_Steroids','Days_on_DMARDs',
                    'Days_on_Biologics','esr_score','das28','smokestatus','disease_activity',
                    'time','prev_ang_status','prev_bp_status','prev_cancer_status','prev_depress_status','prev_diab_status',
                    'prev_disc_status','prev_glauc_status','prev_htattk_status','prev_htfail_status','prev_kidney_status',
                    'prev_liver_status','prev_lung_status','prev_stroke_status']
    '''data_columns = ['agefu', 'ethnicity', 'gender', 'weight', 'bmi', 'crpscore', 'Days_on_Steroids',
                    'Days_on_DMARDs',
                    'Days_on_Biologics', 'esr_score', 'das28', 'smokestatus', 'disease_activity',
                    'prev_ang_status', 'prev_bp_status', 'prev_cancer_status', 'prev_depress_status',
                    'prev_diab_status',
                    'prev_disc_status', 'prev_glauc_status', 'prev_htattk_status', 'prev_htfail_status',
                    'prev_kidney_status',
                    'prev_liver_status', 'prev_lung_status', 'prev_stroke_status']'''
    target_columns = ['ang_status', 'bp_status', 'htattk_status', 'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status', 'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']
    '''data_columns = ['agefu', 'ethnicity', 'gender', 'weight', 'bmi', 'crpscore', 'Days_on_Steroids',
                    'Days_on_DMARDs',
                    'Days_on_Biologics', 'esr_score', 'das28', 'smokestatus', 'disease_activity',
                    'prev_ang_status', 'prev_bp_status', 'prev_cancer_status', 'prev_depress_status',
                    'prev_diab_status',
                    'prev_disc_status', 'prev_htattk_status',
                    'prev_lung_status']
    target_columns = ['ang_status', 'bp_status', 'htattk_status', 'diab_status',
                      'cancer_status', 'disc_status', 'depress_status',
                      'lung_status', ]'''
    data = df[data_columns]
    target = df[target_columns]

    # convert to nump array (with groupings for recurrent)
    data = np.asarray(data)
    target = np.asarray(target)

    return data,target

def k_fold_person_safe(df, k):
    #get amount of unique patients
    patient_list = df['regno'].unique()
    total = len(patient_list)
    patients_per_fold = round(total/k)
    start = 0
    acc = 0
    for n in range(k):
        test_ids = patient_list[start:start+patients_per_fold+1]
        start = start+patients_per_fold
        train_ids = [x for x in patient_list if x not in test_ids]
        test = df.loc[df['regno'].isin(test_ids)]
        train = df.loc[df['regno'].isin(train_ids)]

        train_x,train_y = format_and_split_data(train)
        test_x,test_y = format_and_split_data(test)
        model = get_model_nn_param()
        hist = model.fit(train_x, train_y, batch_size=32, epochs=20)
        #plt.plot(hist.history['loss'])
        #plt.show()
        yhat = model.predict(test_x)
        local_acc = accuracy_score(test_y, np.abs(yhat.round()))
        acc += local_acc
        print(f'local acc {local_acc}')
    print(f'Total accuracy {local_acc / k}')
    return acc

def k_fold_person_safe_param_finder(df, k,layer_count,node_count):
    #get amount of unique patients
    patient_list = df['regno'].unique()
    random.Random(4).shuffle(patient_list)
    total = len(patient_list)
    patients_per_fold = round(total/k)
    start = 0
    acc = 0
    hamming_loss_tot = 0
    recall_tot = 0
    precision_tot =0
    f1_tot = 0
    for n in range(k):
        test_ids = patient_list[start:start+patients_per_fold+1]
        start = start+patients_per_fold
        train_ids = [x for x in patient_list if x not in test_ids]
        test = df.loc[df['regno'].isin(test_ids)]
        train = df.loc[df['regno'].isin(train_ids)]

        train_x,train_y = format_and_split_data(train)
        test_x,test_y = format_and_split_data(test)
        model = build_model(28,13,layer_count,node_count)
        hist = model.fit(train_x, train_y, batch_size=32, epochs=15)
        #plt.plot(hist.history['loss'])
        #plt.show()
        yhat = model.predict(test_x)
        local_acc = accuracy_score(test_y, np.abs(yhat.round()))
        hamming_loss_local = hamming_loss(test_y, np.abs(yhat.round()))
        recall_tot_local = sklearn.metrics.recall_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        precision_tot_local = sklearn.metrics.precision_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        f1_tot_local = sklearn.metrics.f1_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')

        acc += local_acc
        hamming_loss_tot+=hamming_loss_local
        recall_tot +=recall_tot_local
        precision_tot +=precision_tot_local
        f1_tot +=f1_tot_local
        print(f'local acc {local_acc}')
        print(f'local hamming {hamming_loss_local}')
        print(f'local recall {recall_tot_local}')
        print(f'local prec {precision_tot_local}')
        print(f'local f1 {f1_tot_local}')

    print(f'Total accuracy {acc / k}')
    print(f'Total hamming {hamming_loss_tot / k}')
    print(f'Total recall {recall_tot / k}')
    print(f'Total prec {precision_tot / k}')
    print(f'Total f1 {f1_tot / k}')

    return acc/k,hamming_loss_tot/k,recall_tot/k,precision_tot/k,f1_tot

def k_fold_person_safe_param_finder2(df, k,layer_count,node_count):
    #get amount of unique patients
    patient_list = df['regno'].unique()
    random.Random(4).shuffle(patient_list)
    total = len(patient_list)
    patients_per_fold = round(total/k)
    start = 0
    acc = 0
    hamming_loss_tot = 0
    recall_tot = 0
    precision_tot =0
    f1_tot = 0
    for n in range(k):
        test_ids = patient_list[start:start+patients_per_fold+1]
        start = start+patients_per_fold
        train_ids = [x for x in patient_list if x not in test_ids]
        test = df.loc[df['regno'].isin(test_ids)]
        train = df.loc[df['regno'].isin(train_ids)]

        train_x,train_y = format_and_split_data(train)
        test_x,test_y = format_and_split_data(test)
        model = build_model(28,13,layer_count,node_count)
        hist = model.fit(train_x, train_y, batch_size=32, epochs=15)
        #plt.plot(hist.history['loss'])
        #plt.show()
        yhat = model.predict(test_x)
        local_acc = accuracy_score(test_y, np.abs(yhat.round()))
        hamming_loss_local = hamming_loss(test_y, np.abs(yhat.round()))
        recall_tot_local = sklearn.metrics.recall_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        precision_tot_local = sklearn.metrics.precision_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        f1_tot_local = sklearn.metrics.f1_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')

        acc += local_acc
        hamming_loss_tot+=hamming_loss_local
        recall_tot +=recall_tot_local
        precision_tot +=precision_tot_local
        f1_tot +=f1_tot_local
        print(f'local acc {local_acc}')
        print(f'local hamming {hamming_loss_local}')
        print(f'local recall {recall_tot_local}')
        print(f'local prec {precision_tot_local}')
        print(f'local f1 {f1_tot_local}')

    print(f'Total accuracy {acc / k}')
    print(f'Total hamming {hamming_loss_tot / k}')
    print(f'Total recall {recall_tot / k}')
    print(f'Total prec {precision_tot / k}')
    print(f'Total f1 {f1_tot / k}')

    return acc/k,hamming_loss_tot/k,recall_tot/k,precision_tot/k,f1_tot

#add n-1 status to input
#repeat so data has even spread
def format_reg_rep(df):

    #Create new dataframe
    new_df = df.iloc[0:0, :].copy()
    columns_to_add = ['prev_ang_status', 'prev_bp_status', 'prev_htattk_status', 'prev_htfail_status', 'prev_stroke_status', 'prev_diab_status',
                      'prev_liver_status', 'prev_kidney_status', 'prev_cancer_status', 'prev_disc_status', 'prev_depress_status',
                      'prev_lung_status', 'prev_glauc_status']
    target_columns = ['ang_status', 'bp_status', 'htattk_status', 'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status', 'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']
    for col in columns_to_add:
        df[col] = pd.Series(np.zeros(df.shape[0]), index=df.index)

    for i in range(1, df.shape[0] + 1):



        if i >= df.shape[0]:
            pass
            #for col,prev in zip(columns_to_add,target_columns):

                #df.loc[i, col] = df.loc[i - 1, prev]

        elif df.loc[i, 'regno'] == df.loc[i - 1, 'regno']:

            for col,prev in zip(columns_to_add,target_columns):
                df.loc[i, col] = df.loc[i - 1, prev]
    df.to_csv('NOAR-newinp-1.csv', index=False)

    for i in range(1, df.shape[0] + 1):

        # append the current row
        #print(new_df)
        new_df = new_df.append(df.loc[i-1, df.columns])

        if i >= df.shape[0]:
            pass

        # if next is equal
        elif df.loc[i, 'regno'] == df.loc[i - 1, 'regno']:
            num_to_copy = df.loc[i, 'agefu'] - df.loc[i - 1, 'agefu']-1
            for x in range(round(num_to_copy)):

                age_fu = df.loc[i-1, 'agefu'] + x+1
                new_row = df.loc[i - 1, df.columns]
                new_row['agefu'] = age_fu
                new_df = new_df.append(new_row)
    new_df.to_csv('NOAR-newinp-2.csv', index=False)






#rnn_load()
#load sorted data
#df = pd.read_csv("NOAR-proc-increment.csv")
#df = format_reg_rep(df)

#prev 81.69

df = pd.read_csv("NOAR-newinp-3.csv")
todel = remove_zero(df)
for n in todel:
    df.drop(df.loc[df['regno'] ==n].index, inplace=True)


acc,ham_loss,rec,prec,f1 = k_fold_person_safe_param_finder(df,5,0,200)


layer_count = [n for n in range(0,3)]
node_count = [10,50,100,200]
param_acc = {}
param_ham_loss = {}
param_rec={}
param_prec={}
param_f1={}
for l_count in layer_count:
    for n_count in node_count:
        print(f'{l_count}   :   {n_count}')
        acc,ham_loss,rec,prec,f1 = k_fold_person_safe_param_finder(df,5,l_count,n_count)
        key = str(l_count)+':'+str(n_count)
        param_acc[key] = acc
        param_ham_loss[key] = ham_loss
        param_rec[key] = rec
        param_prec[key] = prec
        param_f1[key] = f1


#
print('acc')
print(param_acc)
print('hamm')
print(param_ham_loss)
print('rec')
print(param_rec)
print('prec')
print(param_prec)
print('f1')
print(param_f1)

print('acc')

print({k: v for k, v in sorted(param_acc.items(), key=lambda item: item[1])})
print('hamm')

print({k: v for k, v in sorted(param_ham_loss.items(), key=lambda item: item[1])})
print('rec')

print({k: v for k, v in sorted(param_rec.items(), key=lambda item: item[1])})
print('prec')

print({k: v for k, v in sorted(param_prec.items(), key=lambda item: item[1])})
print('f1')

print({k: v for k, v in sorted(param_f1.items(), key=lambda item: item[1])})

#{'4:10': 0.33564133501924787, '3:10': 0.3397131645281756, '4:20': 0.353706964825765, '2:10': 0.35579706473222383, '3:20': 0.3958957044767296, '1:10': 0.3961972473312052, '2:20': 0.4560402323295346, '0:10': 0.4666944578700406, '4:50': 0.49493758928392484, '3:50': 0.5153813008244306, '1:20': 0.520644908926523, '0:20': 0.5570521707256979, '4:100': 0.5868607432764981, '2:50': 0.5911222146640998, '3:100': 0.6198608709562917, '1:50': 0.6251123932839404, '4:200': 0.6298208221152917, '2:100': 0.6463585608212578, '3:200': 0.6548362394823889, '1:200': 0.6663393467278361, '2:200': 0.6695747877854833, '1:100': 0.6770299674231047, '0:50': 0.7325436137312148, '0:100': 0.777750027251638, '0:200': 0.7943335143176872}

'''
{'2:10': 0.32007230358937655, '1:10': 0.3448196539281553, '0:10': 0.4178145348312885, '2:50': 0.4697659616349081, '1:50': 0.5173033648063228, '2:100': 0.5386799478365192, '1:200': 0.5492091939677538, '1:100': 0.5790121533882621, '2:200': 0.5979258066368783, '0:50': 0.6452624145347399, '0:100': 0.7425851791325863, '0:200': 0.771813974883479}
hamm
{'0:200': 0.025523894625675374, '0:100': 0.029132975095871743, '0:50': 0.04213012906514155, '2:200': 0.046566712972244476, '1:100': 0.04936689432020234, '2:100': 0.05517148066360855, '1:200': 0.057057115890572684, '1:50': 0.058899591768771666, '2:50': 0.06701920661011378, '0:10': 0.07643384521714533, '1:10': 0.08881051426775571, '2:10': 0.09217561209187475}
rec
{'2:10': 5.428636055191135e-05, '1:10': 0.04433189162817094, '0:10': 0.16555546308275776, '2:50': 0.23868839145308543, '1:50': 0.28499395961588403, '2:100': 0.30491327704290205, '1:200': 0.3241424344763768, '1:100': 0.34164738677306294, '2:200': 0.3591717345831141, '0:50': 0.4246508982362022, '0:100': 0.4932070141380664, '0:200': 0.5085951953460003}
prec
{'2:10': 0.0002714318027595567, '1:10': 0.06930797657455584, '0:10': 0.25376858871005, '2:50': 0.3312780434373028, '1:50': 0.37076126084166156, '2:100': 0.38332323671513285, '1:200': 0.3987167802975057, '1:100': 0.40742449964857874, '2:200': 0.4215188350011264, '0:50': 0.49442165300051216, '0:100': 0.534312298198973, '0:200': 0.5423777884051975}
f1
{'2:10': 0.00045238633793259444, '1:10': 0.2554125734329121, '0:10': 0.9504031949463729, '2:50': 1.3319945326303357, '1:50': 1.5571130626464929, '2:100': 1.6467864825383638, '1:200': 1.7353797717083594, '1:100': 1.809704672948074, '2:200': 1.8959803593819102, '0:50': 2.222811904450337, '0:100': 2.528196577415094, '0:200': 2.5981527686980637}

Process finished with exit code 0

'''




#df = modify_fill_time_column_order(df)
#df.to_csv('NOAR-newinp-3.csv', index=False)
"""print('============================')
df = load_proc_inc()
format_reg_rep(df)
#df = drop_columns(df)
to_format = ['weight','bmi','crpscore','Days_on_Steroids','Days_on_DMARDs',
            'Days_on_Biologics','esr_score','das28','time']

print(df.dtypes)

df[to_format] =  np.asarray(df[to_format]).astype(np.float32)

to = ['ethnicity','gender','disease_activity']

#convert into categorical variables
for n in to:
    df[n] = pd.Categorical(df[n])
    df[n] = df[n].cat.codes

print('cross validate')
todel = remove_zero(df)

for n in todel:
    df.drop(df.loc[df['regno'] ==n].index, inplace=True)
print('cross validate')
print(df.shape[0])
train_test_split(df)
cross_validate(df)
print('cross validate ended')
data,targ = seperate_input_labels(df)

print(data.dtypes)
print(targ.dtypes)


#data = np.asarray(data)
#targ = np.asarray(targ)
data = np.asarray(data).astype(np.float32)
targ = np.asarray(targ).astype(np.int8)
pd.DataFrame(targ).to_csv("targg.csv")
#targ = targ.astype(int)
print(data)
print(targ)

#data=tf.convert_to_tensor(data)
#targ=tf.convert_to_tensor(targ)

#print(np.shape(targ))
xtrain,xtest,ytrain,ytest = train_test_split(data,targ,test_size=0.2,random_state=4)
pd.DataFrame(ytrain).to_csv("ytrain.csv")


#model = get_model(13,13,ytrain,ytrain)
model =get_model_nn()#KerasClassifier(build_fn=get_model_nn, epochs=150,verbose=0)

hist = model.fit(xtrain, ytrain, epochs=75,validation_data=(xtest,ytest))

yhat = model.predict(xtest)
print( np.abs(yhat.round()))
pd.DataFrame(np.abs(yhat.round())).to_csv("pred-new.csv")
pd.DataFrame(ytest).to_csv("actual-new.csv")

acc = accuracy_score(ytest, np.abs(yhat.round()))
print(f'accuracu {acc}')"""
'''cv = LeaveOneOut()
cs=cross_val_score(model, data, targ,scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=None)
print(f'cross val score {cs}')'''

"""hist = model.fit(xtrain, ytrain,  epochs=75,validation_data=(xtest,ytest))"""

