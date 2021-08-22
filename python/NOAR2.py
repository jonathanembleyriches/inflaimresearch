import io
import os
import random
from time import sleep
from tkinter import filedialog
import EvalMetrics
import pandas as pd
import numpy as np
import sklearn.metrics
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold, train_test_split, cross_val_score, LeaveOneOut
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Masking
from sklearn.metrics import accuracy_score,hamming_loss
#import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error
from pywaffle import Waffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter import *
import tkinter as tk
import seaborn as sns
from PIL import ImageTk, Image, ImageGrab, EpsImagePlugin
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
def drop_all(df):
    target_columns = ['lbl_ang_status', 'lbl_bp_status', 'lbl_htattk_status', 'lbl_htfail_status',
                        'lbl_stroke_status', 'lbl_diab_status',
                        'lbl_liver_status', 'lbl_kidney_status', 'lbl_cancer_status', 'lbl_disc_status',
                        'lbl_depress_status',
                        'lbl_lung_status', 'lbl_glauc_status']
    new_df = pd.DataFrame(data=None, columns=df.columns, index=df.index)
    new_df.drop(df.index, inplace=True)
    for i in range(df.shape[0]):
        this = df[target_columns].iloc[i].values.flatten().tolist()
        if 1 in this:
            new_df = new_df.append(df.iloc[i])
    return new_df
def build_model(inputs,outputs,hidden_layer_count,node_count,decay):
    # create model
    model = Sequential()
    #
    model.add(Dense(node_count, input_dim=inputs, kernel_initializer='he_uniform',kernel_regularizer=l2(decay),activation='relu'))#



    model.add(Dense(outputs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    #print(model.summary())
    #print('')
    return model


def build_model_decay(inputs,outputs,node_count,decay,act):
    # create model
    model = Sequential()
    model.add(Dense(node_count, input_dim=inputs, kernel_initializer='he_uniform',kernel_regularizer=l2(decay),activation=act))#
    model.add(Dense(outputs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

def build_model_drop(inputs,outputs,node_count,drop,act):
    # create model
    model = Sequential()

    model.add(Dense(node_count, input_dim=inputs, kernel_initializer='he_uniform',activation=act))#
    model.add(Dropout(drop))
    model.add(Dense(outputs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model


def format_and_split_data(df):
    # format data inside
    to = ['ethnicity', 'gender', 'died', 'disease_activity']
    for n in to:
        df[n] = pd.Categorical(df[n])
        df[n] = df[n].cat.codes

    # seperate dataframe into data and targ


    data_columns = ['agefu', 'ethnicity', 'gender', 'weight',
                    'bmi', 'crpscore', 'Days_on_Steroids',
                    'Days_on_DMARDs','Days_on_Biologics', 'esr_score',
                    'das28', 'smokestatus', 'disease_activity',
                    'time', 'prev_ang_status', 'prev_bp_status',
                    'prev_cancer_status', 'prev_depress_status','prev_diab_status',
                    'prev_disc_status', 'prev_glauc_status', 'prev_htattk_status',
                    'prev_htfail_status','prev_kidney_status',
                    'prev_liver_status', 'prev_lung_status', 'prev_stroke_status']
    '''target_columns= ['ang_status', 'bp_status', 'htattk_status',
                      'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status',
                      'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']'''
    target_columns = ['lbl_ang_status', 'lbl_bp_status', 'lbl_htattk_status', 'lbl_htfail_status',
                        'lbl_stroke_status', 'lbl_diab_status',
                        'lbl_liver_status', 'lbl_kidney_status', 'lbl_cancer_status', 'lbl_disc_status',
                        'lbl_depress_status',
                        'lbl_lung_status', 'lbl_glauc_status']


    df[data_columns] = df[data_columns].astype(int)
    df[target_columns]=df[target_columns].astype(int)

    dataout = df[data_columns]
    target = df[target_columns]
    print(f'len of split data {dataout.shape[0]}')
    #print(dataout.dtypes)
    #print(target.dtypes)
    # convert to nump array (with groupings for recurrent)
    dataout = np.asarray(dataout)
    target = np.asarray(target)#.astype('int32')
    #print(target)
    #print(np.asarray(df[target_columns2]))

    return dataout,target


def format_all(df):
    to = ['ethnicity', 'gender', 'died', 'disease_activity']
    for n in to:
        df[n] = pd.Categorical(df[n])
        df[n] = df[n].cat.codes

    # seperate dataframe into data and targ

    data_columns = ['agefu', 'ethnicity', 'gender', 'weight',
                    'bmi', 'crpscore', 'Days_on_Steroids',
                    'Days_on_DMARDs', 'Days_on_Biologics', 'esr_score',
                    'das28', 'smokestatus', 'disease_activity',
                    'time', 'prev_ang_status', 'prev_bp_status',
                    'prev_cancer_status', 'prev_depress_status', 'prev_diab_status',
                    'prev_disc_status', 'prev_glauc_status', 'prev_htattk_status',
                    'prev_htfail_status', 'prev_kidney_status',
                    'prev_liver_status', 'prev_lung_status', 'prev_stroke_status']
    target_columns = ['ang_status', 'bp_status', 'htattk_status',
                      'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status',
                      'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']
    target_columns2 = ['lbl_ang_status', 'lbl_bp_status', 'lbl_htattk_status', 'lbl_htfail_status',
                       'lbl_stroke_status', 'lbl_diab_status',
                       'lbl_liver_status', 'lbl_kidney_status', 'lbl_cancer_status', 'lbl_disc_status',
                       'lbl_depress_status',
                       'lbl_lung_status', 'lbl_glauc_status']

    df[data_columns] = df[data_columns].astype(float)
    df[target_columns2] = df[target_columns2].astype(int)
    df[target_columns] = df[target_columns].astype(int)
    return df

def format_and_split_data3(df):

    data_columns = ['agefu', 'ethnicity', 'gender', 'weight',
                    'bmi', 'crpscore', 'Days_on_Steroids',
                    'Days_on_DMARDs','Days_on_Biologics', 'esr_score',
                    'das28', 'smokestatus', 'disease_activity',
                    'time', 'prev_ang_status', 'prev_bp_status',
                    'prev_cancer_status', 'prev_depress_status','prev_diab_status',
                    'prev_disc_status', 'prev_glauc_status', 'prev_htattk_status',
                    'prev_htfail_status','prev_kidney_status',
                    'prev_liver_status', 'prev_lung_status', 'prev_stroke_status']
    target_columns= ['ang_status', 'bp_status', 'htattk_status',
                      'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status',
                      'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']
    target_columns2 = ['lbl_ang_status', 'lbl_bp_status', 'lbl_htattk_status', 'lbl_htfail_status',
                        'lbl_stroke_status', 'lbl_diab_status',
                        'lbl_liver_status', 'lbl_kidney_status', 'lbl_cancer_status', 'lbl_disc_status',
                        'lbl_depress_status',
                        'lbl_lung_status', 'lbl_glauc_status']


    dataout = df[data_columns]
    target = df[target_columns]
    target2 = df[target_columns2]
    print(f'len of split data {dataout.shape[0]}')

    dataout = np.asarray(dataout)
    target = np.asarray(target)#.astype('int32')
    target2 = np.asarray(target2)#.astype('int32')


    return dataout,target,target2

def format_and_split_data2(df):
    # format data inside
    to = ['ethnicity', 'gender', 'died', 'disease_activity']
    for n in to:
        df[n] = pd.Categorical(df[n])
        df[n] = df[n].cat.codes

    # seperate dataframe into data and targ


    data_columns = ['agefu', 'ethnicity', 'gender', 'weight',
                    'bmi', 'crpscore', 'Days_on_Steroids',
                    'Days_on_DMARDs','Days_on_Biologics', 'esr_score',
                    'das28', 'smokestatus', 'disease_activity',
                    'time', 'prev_ang_status', 'prev_bp_status',
                    'prev_cancer_status', 'prev_depress_status','prev_diab_status',
                    'prev_disc_status', 'prev_glauc_status', 'prev_htattk_status',
                    'prev_htfail_status','prev_kidney_status',
                    'prev_liver_status', 'prev_lung_status', 'prev_stroke_status']
    target_columns= ['ang_status', 'bp_status', 'htattk_status',
                      'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status',
                      'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']
    target_columns2 = ['lbl_ang_status', 'lbl_bp_status', 'lbl_htattk_status', 'lbl_htfail_status',
                        'lbl_stroke_status', 'lbl_diab_status',
                        'lbl_liver_status', 'lbl_kidney_status', 'lbl_cancer_status', 'lbl_disc_status',
                        'lbl_depress_status',
                        'lbl_lung_status', 'lbl_glauc_status']


    df[data_columns] = df[data_columns].astype(int)
    df[target_columns2]=df[target_columns2].astype(int)
    df[target_columns] = df[target_columns].astype(int)
    dataout = df[data_columns]
    target = df[target_columns]
    target2 = df[target_columns2]
    print(f'len of split data {dataout.shape[0]}')
    #print(dataout.dtypes)
    #print(target.dtypes)
    # convert to nump array (with groupings for recurrent)
    dataout = np.asarray(dataout)
    target = np.asarray(target)#.astype('int32')
    target2 = np.asarray(target2)#.astype('int32')
    #print(target)
    #print(np.asarray(df[target_columns2]))

    return dataout,target,target2


def k_fold_person_safe_param_finder(df, k,layer_count,node_count,weight):
    #df = drop_all(df)
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
    coverage_error_tot = 0
    label_ranking_average_precision_score_tot = 0
    label_ranking_loss_tot = 0
    viss = visuliser(['Angina', 'High\nBlood\nPressure', 'Heart\nAttack',
                     'Heart \nFailure', 'Stroke', 'Diabetes',
                     'Liver\nDisease', 'Kidney\nDisease', 'Cancer',
                     'Disc', 'Depres-\nsion',
                     'Lung\nDisease', 'Glaucoma'])
    for n in range(k):
        test_ids = patient_list[start:start+patients_per_fold+1]
        start = start+patients_per_fold
        train_ids = [x for x in patient_list if x not in test_ids]
        test = df.loc[df['regno'].isin(test_ids)]
        train = df.loc[df['regno'].isin(train_ids)]


        #train_x,train_y = format_and_split_data(train)
        train_x,train_y = format_and_split_data(drop_all(train))
        test_x,real,test_y = format_and_split_data2(test)
        model = build_model(27,13,layer_count,node_count,weight)
        hist = model.fit(train_x, train_y, batch_size=32, epochs=30)
        #plt.plot(hist.history['loss'])
        #plt.show()
        yhat = model.predict(test_x)

        #Generate
        '''if n >= 0:
            temp_act = []
            temp_pred = []
            for i in range(1,len(order)):

                temp_pred.append(np.abs(yhat)[i-1].tolist())
                temp_act.append(real[i-1].tolist())

                if order[i] != order[i-1]:
                    #if order[i-1] in reg_no_gen or order[i-1] in reg_no_gen2:
                    viss.gen(temp_act, temp_pred,str(order[i-1]))
                    temp_act = []
                    temp_pred = []'''
                    #hal = input('Next: ')



        #vis.gen(test_y[:15].tolist(), np.abs(yhat)[:15].tolist())
        local_acc = accuracy_score(test_y, np.abs(yhat.round()))

        recall_tot_local = sklearn.metrics.recall_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        precision_tot_local = sklearn.metrics.precision_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        f1_tot_local = sklearn.metrics.f1_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        hamming_loss_local = hamming_loss(test_y, np.abs(yhat.round()))
        coverage_error = sklearn.metrics.coverage_error(test_y, np.abs(yhat.round()))
        label_ranking_average_precision_score = sklearn.metrics.label_ranking_average_precision_score(test_y, np.abs(yhat.round()))
        label_ranking_loss = sklearn.metrics.label_ranking_loss(test_y, np.abs(yhat.round()))
        acc += local_acc
        hamming_loss_tot+=hamming_loss_local
        recall_tot +=recall_tot_local
        precision_tot +=precision_tot_local
        f1_tot +=f1_tot_local
        coverage_error_tot +=coverage_error
        label_ranking_average_precision_score_tot +=label_ranking_average_precision_score
        label_ranking_loss_tot +=label_ranking_loss


    return acc/k,hamming_loss_tot/k,recall_tot/k,precision_tot/k,f1_tot/k,coverage_error_tot/k,\
           label_ranking_average_precision_score_tot/k,label_ranking_loss_tot/k

def generate_fast_vis(actual,pred,id):

    dtypes = np.dtype(
        [
            ("Angina", float),
            ('High\nBlood\nPressure', float),
            ('Heart\nAttack', float),
            ('Heart \nFailure', float),
            ('Stroke', float),
            ('Diabetes', float),
            ('Liver\nDisease', float),
            ('Kidney\nDisease', float),
            ('Cancer', float),
            ('Disc', float),
            ('Depres-\nsion', float),
            ('Lung\nDisease', float),
            ('Glaucoma', float),
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

    predicted = pd.DataFrame(np.empty(0, dtype=dtypes))
    for row in pred:
        data_to_append = {}
        for i in range(len(predicted.columns)):
            data_to_append[predicted.columns[i]] = row[i]
        predicted = predicted.append(data_to_append, ignore_index=True)
    print(predicted)
    labels = pd.DataFrame(actual)
    labels = labels.mask(labels == 1, "X")
    labels = labels.mask(labels == 0, "")
    annotations = labels.astype(str)

    k = sns.heatmap(data=predicted, cmap='Reds', linewidths=1, linecolor="black", annot=annotations, fmt="s", ax=ax,vmin=0, vmax=1
                    ,xticklabels=['Angina', 'High\nBlood\nPressure', 'Heart\nAttack',
                     'Heart\nFailure', 'Stroke', 'Diabetes',
                     'Liver\nDisease', 'Kidney\nDisease', 'Cancer',
                     'Disc', 'Depression',
                     'Lung\nDisease', 'Glaucoma'], annot_kws={"size": 18, "va": "center_baseline", "color": "green"})

    k.set_xlabel("Labels", fontsize=20)
    k.set_ylabel("Years", fontsize=20)

    plt.title(f'Predicted vs True labels for {id}')
    #plt.show()
    fig.savefig(f'fast_output_1/{id}.png')
def k_fold_person_safe_param_finder_decay(df, k,node_count,weight,act,mode,patient_list):
    #df = drop_all(df)
    #get amount of unique patients
    #patient_list = df['regno'].unique()
    #random.Random(4).shuffle(patient_list)
    total = len(patient_list)
    patients_per_fold = round(total/k)


    start = 0
    acc = 0
    hamming_loss_tot = 0
    recall_tot = 0
    precision_tot = 0
    f1_tot = 0
    coverage_error_tot = 0
    label_ranking_average_precision_score_tot = 0
    label_ranking_loss_tot = 0
    roc_auc_tot = 0
    emr = 0
    hamming_loss = 0
    prec_micro= 0
    prec_macro= 0
    prec_weighted= 0
    prec_samples= 0
    prec_avg_micro= 0
    prec_avg_macro= 0
    prec_avg_weighted= 0
    prec_avg_samples= 0
    recall_avg_micro= 0
    recall_avg_macro= 0
    recall_avg_weighted= 0
    recall_avg_samples= 0
    f1_avg_micro = 0
    f1_avg_macro = 0
    f1_avg_weighted = 0
    f1_avg_samples = 0
    roc_auc_micro = 0
    roc_auc_macro = 0
    roc_auc_weighted = 0
    roc_auc_samples = 0
    coverage_error=0
    label_ranking_average_precision_score = 0
    label_ranking_loss = 0

    '''viss = visuliser(['Angina', 'High\nBlood\nPressure', 'Heart\nAttack',
                     'Heart \nFailure', 'Stroke', 'Diabetes',
                     'Liver\nDisease', 'Kidney\nDisease', 'Cancer',
                     'Disc', 'Depres-\nsion',
                     'Lung\nDisease', 'Glaucoma'])'''
    for n in range(k):
        test_ids = patient_list[start:start+patients_per_fold+1]
        start = start+patients_per_fold
        train_ids = [x for x in patient_list if x not in test_ids]
        test = df.loc[df['regno'].isin(test_ids)]
        train = df.loc[df['regno'].isin(train_ids)]


        #train_x,train_y = format_and_split_data(train)
        train_x,train_real_y,train_y = format_and_split_data3(train)
        #train_x,train_y = format_and_split_data(drop_all(train))
        test_x,real,test_y = format_and_split_data3(test)
        if mode == 1:
            test_y = real
            train_y = train_real_y
        #model = build_model(27,13,layer_count,node_count,weight)
        model = build_model_decay(27,13,node_count,weight,act)
        hist = model.fit(train_x, train_y, batch_size=32, epochs=5)
        #plt.plot(hist.history['loss'])
        #plt.show()
        yhat = model.predict(test_x)
        yhat_int = np.abs(yhat.round()).astype(int)
        yhat = np.abs(yhat)
        #print(yhat)
        #print(test_y)
        #yhat = np.array([[1,0.3],[1,1]])
        #test_y = np.array([[1,0],[1,1]])
        '''t_emr += EvalMetrics.emr(test_y,yhat)
        t_hamming_loss += EvalMetrics.hamming_loss(test_y,yhat)
        t_example_based_accuracy += EvalMetrics.example_based_accuracy(test_y,yhat)
        t_example_based_precision += EvalMetrics.example_based_precision(test_y,yhat)
        t_label_based_macro_accuracy += EvalMetrics.label_based_macro_accuracy(test_y,yhat)
        t_label_based_macro_precision += EvalMetrics.label_based_macro_precision(test_y,yhat)
        t_label_based_macro_recall += EvalMetrics.label_based_macro_recall(test_y,yhat)

        t_label_based_micro_accuracy += EvalMetrics.label_based_micro_accuracy(test_y,yhat)
        t_label_based_micro_precision += EvalMetrics.label_based_micro_precision(test_y,yhat)
        t_label_based_micro_recall += EvalMetrics.label_based_micro_recall(test_y,yhat)
        t_alpha_evaluation_score += EvalMetrics.alpha_evaluation_score(test_y,yhat)'''
        '''emr += sklearn.metrics.accuracy_score(test_y, yhat_int)
        prec_micro += sklearn.metrics.precision_score(test_y, yhat_int, average='micro')
        prec_macro += sklearn.metrics.precision_score(test_y, yhat_int, average='macro')
        prec_weighted += sklearn.metrics.precision_score(test_y, yhat_int, average='weighted')
        prec_samples += sklearn.metrics.precision_score(test_y, yhat_int, average='samples')

        prec_avg_micro += sklearn.metrics.average_precision_score(test_y, yhat, average='micro')
        prec_avg_macro += sklearn.metrics.average_precision_score(test_y, yhat, average='macro')
        prec_avg_weighted += sklearn.metrics.average_precision_score(test_y, yhat, average='weighted')
        prec_avg_samples += sklearn.metrics.average_precision_score(test_y, yhat, average='samples')

        recall_avg_micro += sklearn.metrics.recall_score(test_y, yhat_int, average='micro')
        recall_avg_macro += sklearn.metrics.recall_score(test_y, yhat_int, average='macro')
        recall_avg_weighted += sklearn.metrics.recall_score(test_y, yhat_int, average='weighted')
        recall_avg_samples += sklearn.metrics.recall_score(test_y, yhat_int, average='samples')

        f1_avg_micro += sklearn.metrics.f1_score(test_y, yhat_int, average='micro')
        f1_avg_macro += sklearn.metrics.f1_score(test_y, yhat_int, average='macro')
        f1_avg_weighted += sklearn.metrics.f1_score(test_y, yhat_int, average='weighted')
        f1_avg_samples += sklearn.metrics.f1_score(test_y, yhat_int, average='samples')'''

        '''roc_auc_micro += sklearn.metrics.roc_auc_score(test_y, yhat, average='micro')
        roc_auc_macro += sklearn.metrics.roc_auc_score(test_y, yhat, average='macro')
        roc_auc_weighted += sklearn.metrics.roc_auc_score(test_y, yhat, average='weighted')
        roc_auc_samples += sklearn.metrics.roc_auc_score(test_y, yhat, average='samples')'''

        '''hamming_loss += sklearn.metrics.hamming_loss(test_y, yhat_int)
        coverage_error += sklearn.metrics.coverage_error(test_y, yhat_int)
        label_ranking_average_precision_score += sklearn.metrics.label_ranking_average_precision_score(test_y, yhat_int)
        label_ranking_loss += sklearn.metrics.label_ranking_loss(test_y, yhat_int)'''
        local_acc = accuracy_score(test_y, np.abs(yhat.round()))

        recall_tot_local = sklearn.metrics.recall_score(y_true=test_y, y_pred=yhat_int, average='samples')
        precision_tot_local = sklearn.metrics.precision_score(y_true=test_y, y_pred=yhat_int,
                                                              average='samples')
        f1_tot_local = sklearn.metrics.f1_score(y_true=test_y, y_pred=yhat_int, average='samples')
        hamming_loss_local = sklearn.metrics.hamming_loss(test_y, yhat_int)
        coverage_error = sklearn.metrics.coverage_error(test_y, yhat)
        label_ranking_average_precision_score = sklearn.metrics.label_ranking_average_precision_score(test_y, yhat)
        label_ranking_loss = sklearn.metrics.label_ranking_loss(test_y, yhat)
        acc += local_acc
        hamming_loss_tot += hamming_loss_local
        recall_tot += recall_tot_local
        precision_tot += precision_tot_local
        f1_tot += f1_tot_local
        coverage_error_tot += coverage_error
        label_ranking_average_precision_score_tot += label_ranking_average_precision_score
        label_ranking_loss_tot += label_ranking_loss
        roc_auc_tot +=sklearn.metrics.roc_auc_score(test_y, yhat, average=None)

    return acc / k, hamming_loss_tot / k, recall_tot / k, precision_tot / k, f1_tot / k, coverage_error_tot / k, \
           label_ranking_average_precision_score_tot / k, label_ranking_loss_tot / k,roc_auc_tot/k
    '''return emr/k, hamming_loss/k,prec_micro/k,\
    prec_macro/k,\
    prec_weighted/k,\
    prec_samples/k,\
    prec_avg_micro/k,\
    prec_avg_macro/k,\
    prec_avg_weighted/k,\
    prec_avg_samples/k,\
    recall_avg_micro/k,\
    recall_avg_macro/k,\
    recall_avg_weighted/k,\
    recall_avg_samples/k,\
    f1_avg_micro/k,\
    f1_avg_macro/k,\
    f1_avg_weighted/k,\
    f1_avg_samples/k,\
    roc_auc_micro/k,\
    roc_auc_macro/k,\
    roc_auc_weighted/k,\
    roc_auc_samples/k,\
    coverage_error/k,\
    label_ranking_average_precision_score/k,\
    label_ranking_loss/k,'''
    '''return t_emr/k,t_hamming_loss/k,t_example_based_accuracy/k,t_example_based_precision/k,t_label_based_macro_accuracy/k\
        ,t_label_based_macro_precision/k,t_label_based_macro_recall/k, t_label_based_micro_accuracy/k, t_label_based_micro_precision/k\
        ,t_label_based_micro_recall/k,t_alpha_evaluation_score/k'''



def k_fold_person_safe_param_finder_decay2(df, k,mode,name):



    weights_decay = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    node_counts = [10, 25, 50, 100]
    activation_func = ['relu', 'sigmoid']

    column_names = [
        'id',
        'acc', \
        'hamming_loss_tot', \
        'recall_tot', \
        'precision_tot', \
        'f1_tot', \
        'coverage_error_tot', \
        'label_ranking_average_precision_score_tot', \
        'label_ranking_loss_tot', 'roc_auc_tot'

    ]
    patient_list = df['regno'].unique()

    total = len(patient_list)
    patients_per_fold = round(total / k)
    start = 0
    random.Random(4).shuffle(patient_list)

    results = pd.DataFrame(columns=column_names)

    count = 0
    total = k*len(weights_decay)*len(node_counts)*len(activation_func)

    for n in range(k):
        test_ids = patient_list[start:start+patients_per_fold+1]
        start = start+patients_per_fold
        train_ids = [x for x in patient_list if x not in test_ids]
        test = df.loc[df['regno'].isin(test_ids)]
        train = df.loc[df['regno'].isin(train_ids)]

        train_x,train_real_y,train_y = format_and_split_data3(train)
        #train_x,train_y = format_and_split_data(drop_all(train))
        test_x,real,test_y = format_and_split_data3(test)
        if mode == 1:
            test_y = real
            train_y = train_real_y
        for weight in weights_decay:
            for node_count in node_counts:
                for act in activation_func:
                    print(f'====== {count}/{total}')
                    count+=1
                    key = str(weight) + ':' + str(node_count) + ':' + act
                    if key not in results['id'].values:
                        print('creating new')
                        results = results.append({'id':key,
                        'acc':0,
                        'hamming_loss_tot':0,
                        'recall_tot':0,
                        'precision_tot':0,
                        'f1_tot':0,
                        'coverage_error_tot':0,
                        'label_ranking_average_precision_score_tot':0,
                        'label_ranking_loss_tot':0,
                        'roc_auc_tot':0}, ignore_index=True)
                    else:
                        print('append mode')




                    model = build_model_decay(27,13,node_count,weight,act)
                    hist = model.fit(train_x, train_y, batch_size=32, epochs=7)
                    yhat = model.predict(test_x)


                    temp_act = []
                    temp_pred = []
                    order =  test['regno'].tolist()
                    for i in range(1,len(order)):

                        temp_pred.append(np.abs(yhat)[i-1].tolist())
                        temp_act.append(test_y[i-1].tolist())

                        if order[i] != order[i-1]:
                            #if order[i-1] in reg_no_gen or order[i-1] in reg_no_gen2:
                            #viss.gen(temp_act, temp_pred,str(order[i-1]))
                            generate_fast_vis(temp_act, temp_pred,str(order[i-1]))
                            temp_act = []
                            temp_pred = []


                    yhat_int = np.abs(yhat.round()).astype(int)
                    yhat = np.abs(yhat)
                    #results.loc[df['id'] == key, ['acc']]
                    results.loc[results['id'] == key, ['acc']] =  results.loc[results['id'] == key, ['acc']] + accuracy_score(test_y, np.abs(yhat.round()))
                    results.loc[results['id'] == key, ['recall_tot']] = results.loc[results['id'] == key, ['recall_tot']] +sklearn.metrics.recall_score(y_true=test_y, y_pred=yhat_int, average='samples')
                    results.loc[results['id'] == key, ['precision_tot']] = results.loc[results['id'] == key, ['precision_tot']]+sklearn.metrics.precision_score(y_true=test_y, y_pred=yhat_int,                                                 average='samples')
                    results.loc[results['id'] == key, ['f1_tot']] = results.loc[results['id'] == key, ['f1_tot']]+sklearn.metrics.f1_score(y_true=test_y, y_pred=yhat_int, average='samples')
                    results.loc[results['id'] == key, ['hamming_loss_tot']] = results.loc[results['id'] == key, ['hamming_loss_tot']]+ sklearn.metrics.hamming_loss(test_y, yhat_int)
                    results.loc[results['id'] == key, ['coverage_error_tot']]= results.loc[results['id'] == key, ['coverage_error_tot']]+sklearn.metrics.coverage_error(test_y, yhat)
                    results.loc[results['id'] == key, ['label_ranking_average_precision_score_tot']]=results.loc[results['id'] == key, ['label_ranking_average_precision_score_tot']]+ sklearn.metrics.label_ranking_average_precision_score(test_y, yhat)
                    results.loc[results['id'] == key, ['label_ranking_loss_tot']]= results.loc[results['id'] == key, ['label_ranking_loss_tot']]+sklearn.metrics.label_ranking_loss(test_y, yhat)
                    average_auc_roc = sklearn.metrics.roc_auc_score(test_y, yhat, average=None)

                    results.loc[results['id'] == key, ['roc_auc_tot']]= results.loc[results['id'] == key, ['roc_auc_tot']]+np.mean(average_auc_roc)
                    '''
                    results['acc'] += accuracy_score(test_y, np.abs(yhat.round()))
                    results['recall_tot'] += sklearn.metrics.recall_score(y_true=test_y, y_pred=yhat_int, average='samples')
                    results['precision_tot'] += sklearn.metrics.precision_score(y_true=test_y, y_pred=yhat_int,                                                 average='samples')
                    results['f1_tot'] += sklearn.metrics.f1_score(y_true=test_y, y_pred=yhat_int, average='samples')
                    results['hamming_loss_tot'] += sklearn.metrics.hamming_loss(test_y, yhat_int)
                    results['coverage_error_tot'] += sklearn.metrics.coverage_error(test_y, yhat)
                    results['label_ranking_average_precision_score_tot'] += sklearn.metrics.label_ranking_average_precision_score(test_y, yhat)
                    results['label_ranking_loss_tot'] += sklearn.metrics.label_ranking_loss(test_y, yhat)
                    average_auc_roc = sklearn.metrics.roc_auc_score(test_y, yhat, average=None)

                    results['roc_auc_tot'] +=np.mean(average_auc_roc)'''

    results.to_csv(f"eval_results/{name}_temp.csv")
    lbls = ['acc', \
        'hamming_loss_tot', \
        'recall_tot', \
        'precision_tot', \
        'f1_tot', \
        'coverage_error_tot', \
        'label_ranking_average_precision_score_tot', \
        'label_ranking_loss_tot', 'roc_auc_tot']
    #results[lbls] = results[lbls]/k
    results.to_csv(f"eval_results/{name}.csv")

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


'''def weight_decay_search(df):
    all_acc = []
    all_ham = []
    all_rec = []
    all_prec = []
    all_f1 = []
    weights_decay = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for weight in weights_decay:
        acc, ham_loss, rec, prec, f1 = k_fold_person_safe_param_finder(df, 5, 0, 200, weight)
        all_acc.append(acc)
        all_ham.append(ham_loss)
        all_rec.append(rec)
        all_prec.append(prec)
        all_f1.append(f1)

    for i in range(len(weights_decay)):
        print(weights_decay[i])
        print(all_acc[i])
        print(all_ham[i])
        print(all_rec[i])
        print(all_prec[i])
        print(all_f1[i])'''

def grid_search(df):
    all_acc = {}
    all_ham = {}
    all_rec = {}
    all_prec = {}
    all_f1 = {}
    all_coverage ={}
    all_ranking_average_precision_score={}
    all_ranking_loss = {}
    weights_decay = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    node_counts = [10,25,50,100]

    for weight in weights_decay:
        for node_count in node_counts:
            key = str(weight) + ':' + str(node_count)

            acc, ham_loss, rec, prec, f1,cov,rank_avg,rank_loss = k_fold_person_safe_param_finder(df, 5, 0, node_count, weight)
            all_acc[key] = acc
            all_ham[key] = ham_loss
            all_rec[key] = rec
            all_prec[key] = prec
            all_f1[key] = f1
            all_coverage[key] = cov
            all_ranking_average_precision_score[key] = rank_avg
            all_ranking_loss[key] = rank_loss

    print('acc')

    print({k: v for k, v in sorted(all_acc.items(), key=lambda item: item[1])})
    print('hamm')

    print({k: v for k, v in sorted(all_ham.items(), key=lambda item: item[1])})
    print('rec')

    print({k: v for k, v in sorted(all_rec.items(), key=lambda item: item[1])})
    print('prec')

    print({k: v for k, v in sorted(all_prec.items(), key=lambda item: item[1])})
    print('f1')

    print({k: v for k, v in sorted(all_f1.items(), key=lambda item: item[1])})
    print('Coverage error')

    print({k: v for k, v in sorted(all_coverage.items(), key=lambda item: item[1])})
    print('Label ranking average precision')

    print({k: v for k, v in sorted(all_ranking_average_precision_score.items(), key=lambda item: item[1])})
    print('Ranking loss')

    print({k: v for k, v in sorted(all_ranking_loss.items(), key=lambda item: item[1])})

def grid_search_no_decay(df):
    all_acc = {}
    all_ham = {}
    all_rec = {}
    all_prec = {}
    all_f1 = {}
    all_coverage ={}
    all_ranking_average_precision_score={}
    all_ranking_loss = {}

    node_counts = [10,25,50,100]


    for node_count in node_counts:
        key = str(node_count)

        acc, ham_loss, rec, prec, f1,cov,rank_avg,rank_loss = k_fold_person_safe_param_finder(df, 5, 0, node_count, 1)
        all_acc[key] = acc
        all_ham[key] = ham_loss
        all_rec[key] = rec
        all_prec[key] = prec
        all_f1[key] = f1
        all_coverage[key] = cov
        all_ranking_average_precision_score[key] = rank_avg
        all_ranking_loss[key] = rank_loss

    print('acc')

    print({k: v for k, v in sorted(all_acc.items(), key=lambda item: item[1])})
    print('hamm')

    print({k: v for k, v in sorted(all_ham.items(), key=lambda item: item[1])})
    print('rec')

    print({k: v for k, v in sorted(all_rec.items(), key=lambda item: item[1])})
    print('prec')

    print({k: v for k, v in sorted(all_prec.items(), key=lambda item: item[1])})
    print('f1')

    print({k: v for k, v in sorted(all_f1.items(), key=lambda item: item[1])})
    print('Coverage error')

    print({k: v for k, v in sorted(all_coverage.items(), key=lambda item: item[1])})
    print('Label ranking average precision')

    print({k: v for k, v in sorted(all_ranking_average_precision_score.items(), key=lambda item: item[1])})
    print('Ranking loss')

    print({k: v for k, v in sorted(all_ranking_loss.items(), key=lambda item: item[1])})

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect
img = mpimg.imread('img.png')
data = [
    [0,0,0,0,0,0.5,1,1,1,0],
    [0,0,0,0,0,1,0,0,1,0],
    [0,0,1,0,0,0,1,1,0,0],
    [0,0,1,0,0,1,1,0,1,0],
    [0,0,1,0,1,0,0,1,1,0],
    [1,0,0,1,0,1,0,0,1,0],
    [0,1,0,0,0,1,1,1,1,1],
    [0,1,0,0,0,0,1,1,1,1],
    [1,0,0,0,1,1,1,0,1,0],
    [1,1,1,1,0,0,0,1,1,0]
]


data2 = [[random.random() for i in range(len(data[0]))] for j in range(len(data))]
import matplotlib.pyplot as plt

#plt.imshow(img)

#plt.imshow(data,alpha=.1)
plt.imshow(data2, interpolation='nearest',vmin=0, vmax=1)
plt.imshow(data, interpolation='nearest',vmin=0, vmax=1,alpha=0.5)
highlight_cell(2,1, color="limegreen", linewidth=3)
#plt.show()




class visuliser():
    def __init__(self,column_names):

        self.column_names = column_names
        #self.root = tk.Tk()

        #self.c = tk.Canvas(self.root, height=1080, width=1080, bg='white')
        #self.root.configure(background='white')
        self.scale = 1
        self.root = tk.Tk()

        self.c = tk.Canvas(self.root, height=1080, width=1080, bg='white')
        self.root.configure(background='white')
        self.c.pack(fill=tk.BOTH, expand=True)

        self.c.bind('<Configure>', self.gen_grid)

    def gen(self,actual,pred,name):

        self.name = name
        self.actual = actual
        self.pred = pred
        print(type(self.actual))
        print(self.actual)
        print(type(self.pred))
        print(self.pred)
        self.gen_grid()
        self.root.update()
        #self.root.mainloop()


    def rgbtohex(self,r, g, b):
        return f'#{r:02x}{g:02x}{b:02x}'

    def create_cell(self,x,y,width,height,val):
        #print(x,y)
        cmap = cm.Reds
        norm = Normalize(vmin=0, vmax=1)
        r,g,b,a = cmap(norm(val))
        co = self.rgbtohex(int(r*255),int(g*255),int(255*b))
        self.c.create_rectangle(x, y, x+width, y+height,
                           outline="black", fill=co)


    def save_as_png(self):
        EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.54.0\bin\gswin64c'
        #save_name = filedialog.asksaveasfilename()
        self.c.postscript(file="output2/"+self.name + ".ps", colormode='color')  # save canvas as encapsulated postscript
        #img = Image.open(save_name + ".ps")

        cmd = f"magick -density 500 -depth 16 output/{self.name}.ps output2/{self.name}.jpg"
        os.system(cmd)
        #img.save(save_name + ".png", format='png', subsampling=0, quality=100)
        #sleep(2)
        '''x0 = self.c.winfo_rootx()
        y0 = self.c.winfo_rooty()
        x1 = x0 + self.c.winfo_width()
        y1 = y0 + self.c.winfo_height()
        im = ImageGrab.grab((x0, y0, x1, y1))
        im.save('mypic2.png') # Can also say im.show() to display it
        self.c.postscript(file="file_name.ps", colormode='color')
        img = Image.open("file_name.ps")
        img.save("file_name.png")'''
        '''EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.54.0\bin\gswin64c'
        ps = self.c.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save('test.jpg')'''
        #self.root.quit()

    def gen_grid(self,event=None):
        w = (self.c.winfo_reqwidth()*self.scale/10)*9#(self.c.winfo_width()*self.scale/10)*9  # Get current width of canvas
        h = (self.c.winfo_reqheight()*self.scale/10)*9  # Get current height of canvas
        self.c.delete('all')  # Will only remove the grid_line
        col_count = len(self.actual[0])+1
        row_count = len(self.actual)+1

        w_spacing = int(w / col_count)
        h_spacing = int(h / row_count)



        smallest = min(w_spacing, h_spacing)

        img = Image.open("pill3.png")
        img2=Image.open("data_vis/scalenew2.PNG")
        img2 = img2.resize((int(smallest*((col_count-2)/2)), 25), Image.ANTIALIAS)

        img = img.resize((int(smallest*.7), int(smallest*.7)*self.scale), Image.ANTIALIAS)

        self.c.image = ImageTk.PhotoImage(img, master=self.root)  # Use self.image
        #xpos = 0
        #ypos = 0
        for y in range(0, row_count):

            for x in range(0, col_count):
                if y == 0:
                    # place names
                   if x!=0:
                        self.c.create_text((x * smallest) + (smallest / 2), (y * smallest) + (smallest / 2),
                                           fill="black",
                                           font="Times 12 bold",
                                           text=self.column_names[x-1])
                   else:
                       self.c.create_text((x * smallest) + (smallest / 2), (y * smallest) + (smallest / 2),
                                          fill="black",
                                          font="Times 12 bold",
                                          text=self.name)

                elif x > 0:

                    pred_val = self.pred[y - 1][x - 1]
                    actual_val = self.actual[y - 1][x - 1]
                    self.create_cell(x*smallest, y*smallest, smallest, smallest, pred_val)
                    if actual_val == 1:
                        self.c.create_image((x) * smallest + (int(smallest*.3/2)), (y) * smallest+ (int(smallest*.3/2)), image=self.c.image,
                                            anchor=tk.NW)
                else:
                    self.c.create_text((x*smallest) + (smallest / 2), (y*smallest) + (smallest / 2), fill="black",
                                       font="Times 12 bold",
                                       text=f"Year {y}")

        self.c.create_text(100 ,(row_count+2)*smallest, fill="black",
                           font="Times 12 bold",
                           text=f"Prediction Confidence")
        self.c.image2 = ImageTk.PhotoImage(img2, master=self.root)  # Use self.image
        self.c.create_image(10 ,(row_count+3)*smallest, image=self.c.image2, anchor=tk.NW)

        self.c.create_text(700, (row_count+2)*smallest, fill="black",
                           font="Times 12 bold",
                           text=f"Actual")
        self.c.create_image(700 ,(row_count+3)*smallest, image=self.c.image, anchor=tk.NW)
        self.save_as_png()


def new_occur_label(df):
    target_columns = ['ang_status', 'bp_status', 'htattk_status', 'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status', 'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']
    target_columns_2 = ['lbl_ang_status', 'lbl_bp_status', 'lbl_htattk_status', 'lbl_htfail_status',
                        'lbl_stroke_status', 'lbl_diab_status',
                        'lbl_liver_status', 'lbl_kidney_status', 'lbl_cancer_status', 'lbl_disc_status',
                        'lbl_depress_status',
                        'lbl_lung_status', 'lbl_glauc_status']
    # Get all unique patient ids
    patient_list = df['regno'].unique()
    print(f'len of patient list {len(patient_list)}')
    for col in target_columns_2:
        df[col] = np.zeros(df.shape[0])
    new_df = pd.DataFrame(data=None, columns=df.columns, index=df.index)
    new_df.drop(df.index, inplace=True)
    start = 0
    new = [0]*13
    # iter over all patients
    for j, patient_id in enumerate(patient_list):
        occured = [0]*13
        # get all data regarding 1 user
        subframe = df[df.regno.isin([patient_id])]
        for i in range(0, subframe.shape[0]):
            for col, tcol in zip(target_columns, target_columns_2):
                subframe[tcol].iloc[i]  =  subframe[col].iloc[i]

            vals = subframe[target_columns_2].iloc[i].values.flatten().tolist()
            for jk in range(len(vals)):
                if occured[jk] > 0 and vals[jk]==1:
                    vals[jk] = 0
            for index, col in enumerate(target_columns_2):
                subframe[col].iloc[i] = vals[index]
            for ii, jj in enumerate(vals):
                occured[ii] += jj
            #print(vals)
            #occured = map(lambda x,y:x+y, occured, vals)

                #occured = subframe[target_columns].iloc[0].values.flatten().tolist()
        #print(subframe)
        print(f'{j}/{len(patient_list)}')

        ''' # create consistent gap
                  for i in range(1, subframe.shape[0]):
                      prev = subframe[target_columns].iloc[i - 1].values.flatten().tolist()
                      this = subframe[target_columns].iloc[i].values.flatten().tolist()
                      new = [0] * 13
                      for k, (p, t) in enumerate(zip(occured, this)):
                          #new.append(int(t - p))
                          if t == 1 and p == 0:
                              new[k] = 1
                              occured[k] = 1
                  '''
            #subframe[target_columns_2].iloc[i] = new

        new_df = new_df.append(subframe)

    return new_df


def hyperparam_search_decay(df,name,name2):

    weights_decay = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    node_counts = [10,25,50,100]
    activation_func = ['relu','sigmoid']

    column_names = [
        'id',
        'acc', \
        'hamming_loss_tot', \
        'recall_tot', \
        'precision_tot', \
        'f1_tot', \
        'coverage_error_tot', \
        'label_ranking_average_precision_score_tot', \
        'label_ranking_loss_tot', 'roc_auc'


    ]
    ''''emr' ,\
                    'hamming_loss' ,\
                    'prec_micro' ,\
                    'prec_macro' ,\
                    'prec_weighted' ,\
                    'prec_samples' ,\
                    'prec_avg_micro' ,\
                    'prec_avg_macro' ,\
                    'prec_avg_weighted' ,\
                    'prec_avg_samples' ,\
                    'recall_avg_micro' ,\
                    'recall_avg_macro' ,\
                    'recall_avg_weighted' ,\
                    'recall_avg_samples' ,\
                    'f1_avg_micro' ,\
                    'f1_avg_macro' ,\
                    'f1_avg_weighted' ,\
                    'f1_avg_samples' ,\
                    'roc_auc_micro' ,\
                    'roc_auc_macro' ,\
                    'roc_auc_weighted' ,\
                    'roc_auc_samples' ,\
                    'coverage_error',\
                    'label_ranking_average_precision_score' ,\
                    'label_ranking_loss'''

    results = pd.DataFrame(columns=column_names)
    results2 = pd.DataFrame(columns=column_names)

    patient_list = df['regno'].unique()
    random.Random(4).shuffle(patient_list)
    for weight in weights_decay:
        for node_count in node_counts:
            for act in activation_func:

                key = str(weight) + ':' + str(node_count) + ':' + act

                '''emr ,\
                hamming_loss ,\
                prec_micro ,\
                prec_macro ,\
                prec_weighted ,\
                prec_samples ,\
                prec_avg_micro ,\
                prec_avg_macro ,\
                prec_avg_weighted ,\
                prec_avg_samples ,\
                recall_avg_micro ,\
                recall_avg_macro ,\
                recall_avg_weighted ,\
                recall_avg_samples ,\
                f1_avg_micro ,\
                f1_avg_macro ,\
                f1_avg_weighted ,\
                f1_avg_samples ,\
                roc_auc_micro ,\
                roc_auc_macro ,\
                roc_auc_weighted ,\
                roc_auc_samples ,\
                coverage_error,\
                label_ranking_average_precision_score ,\
                label_ranking_loss'''

                ''' acc ,\
                hamming_loss_tot,\
                recall_tot ,\
                precision_tot ,\
                f1_tot ,\
                coverage_error_tot ,\
                label_ranking_average_precision_score_tot ,\
                label_ranking_loss_tot,roc_auc =k_fold_person_safe_param_finder_decay(df,5,node_count,weight,act,0)
                results = results.append(pd.Series([key,acc,\
                hamming_loss_tot,\
                recall_tot ,\
                precision_tot ,\
                f1_tot ,\
                coverage_error_tot ,\
                label_ranking_average_precision_score_tot ,\
                label_ranking_loss_tot,roc_auc], index=column_names), ignore_index=True)'''

                acc, \
                hamming_loss_tot, \
                recall_tot, \
                precision_tot, \
                f1_tot, \
                coverage_error_tot, \
                label_ranking_average_precision_score_tot, \
                label_ranking_loss_tot,roc_auc = k_fold_person_safe_param_finder_decay(df, 5, node_count, weight, act, 1, patient_list)
                results2 = results2.append(pd.Series([key,acc ,\
                hamming_loss_tot,\
                recall_tot ,\
                precision_tot ,\
                f1_tot ,\
                coverage_error_tot ,\
                label_ranking_average_precision_score_tot ,\
                label_ranking_loss_tot,roc_auc ], index=column_names), ignore_index=True)
    #results.to_csv(f"eval_results/{name}.csv")
    results2.to_csv(f"eval_results/{name2}.csv")


def format_reg_rep(df):

    #Create new dataframe
    new_df = df.iloc[0:0, :].copy()

    for i in range(1, df.shape[0] + 1):
        print(f'{i}/{df.shape[0]}')
        # append the current row
        #print(new_df)
        new_df = new_df.append(df.loc[i-1, df.columns])

        if i >= df.shape[0]:
            pass

        # if next is equal
        elif df.loc[i, 'regno'] == df.loc[i - 1, 'regno']:
            num_to_copy = df.loc[i, 'agefu'] - df.loc[i - 1, 'agefu']
            if num_to_copy != 0:
                num_to_copy = (num_to_copy/0.5)-1
            for x in range(round(num_to_copy)):

                age_fu = df.loc[i-1, 'agefu'] + x+0.5
                new_row = df.loc[i - 1, df.columns]
                new_row['agefu'] = age_fu
                new_df = new_df.append(new_row)
    new_df.to_csv('NOAR-6month.csv', index=False)



'''actual = [[0,1,0],[0,1,1],[0,1,1]]
pred = [[0.2,0.3,0.5],[0.7,0.1,0.91],[0.4,0.8,1.0]]
vis = visuliser(["c1","c2","c3"])
plt.ion()
plt.matshow(np.array(pred))
vis.gen(actual,pred,'sdf')
vis.gen(actual,pred,'sdfsd')'''
target_columns_2 = ['lbl_ang_status', 'lbl_bp_status', 'lbl_htattk_status', 'lbl_htfail_status',
                        'lbl_stroke_status', 'lbl_diab_status',
                        'lbl_liver_status', 'lbl_kidney_status', 'lbl_cancer_status', 'lbl_disc_status',
                        'lbl_depress_status',
                        'lbl_lung_status', 'lbl_glauc_status']
'''df = pd.read_csv("NOAR-newinp-3.csv")#noar_newcols.csv" NOAR-newinp-3.csv
#df = pd.read_csv("NOAR-6month.csv")#noar_newcols.csv" NOAR-newinp-3.csv
#df = pd.read_csv("NOAR-6month-fullform.csv")#noar_newcols.csv" NOAR-newinp-3.csv

todel = remove_zero(df)
for n in todel:
    df.drop(df.loc[df['regno'] ==n].index, inplace=True)
df = new_occur_label(df)


df = format_all(df)'''
df = pd.read_csv("NOAR-6month-fullform.csv")#noar_newcols.csv" NOAR-newinp-3.csv

#df.to_csv('NOAR-1year-fullform.csv', index=False)
k_fold_person_safe_param_finder_decay2(df,5,1,'naive 1 year final final')
#hyperparam_search_decay(df,'naive 1 year','naive 6mon')

'''t_emr,\
t_hamming_loss,\
t_example_based_accuracy,\
t_example_based_precision,\
t_label_based_macro_accuracy,\
t_label_based_macro_precision,\
t_label_based_macro_recall,\
t_label_based_micro_accuracy,\
t_label_based_micro_precision,\
t_label_based_micro_recall,t_alpha_evaluation_score = k_fold_person_safe_param_finder2(df,5,0,100,1e-05)
print(t_emr,\
t_hamming_loss,\
t_example_based_accuracy,\
t_example_based_precision,\
t_label_based_macro_accuracy,\
t_label_based_macro_precision,\
t_label_based_macro_recall,\
t_label_based_micro_accuracy,\
t_label_based_micro_precision,\
t_label_based_micro_recall,t_alpha_evaluation_score)'''

#grid_search_no_decay(df)
#print(df)
#1e-05:100  -new columns
#50,1e-06 - old columns
#0.1:10'
#acc, ham_loss, rec, prec, f1, cov, rank_avg, rank_loss = k_fold_person_safe_param_finder(df,5,0,100,1e-05)
#print(acc, ham_loss, rec, prec, f1, cov, rank_avg, rank_loss)







'''
=====
NOAR dataset
=====
df = pd.read_csv("NOAR-newinp-3.csv")
todel = remove_zero(df)
for n in todel:
    df.drop(df.loc[df['regno'] ==n].index, inplace=True)

grid_search_no_decay(df)
#grid_search(df)'''

'''#load dataset
df = pd.read_csv("data/biobank.csv")
patient_list = df['id'].unique()
print(f' {len(patient_list)} unique patients')
print(f'{df.shape[0]} rows')
#drop obsolete columns
columns_to_drop=[]

#print(df.head)'''