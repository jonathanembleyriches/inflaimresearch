import random
import pandas as pd
import numpy as np
import sklearn.metrics
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold, train_test_split, cross_val_score, LeaveOneOut
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Masking
from sklearn.metrics import accuracy_score,hamming_loss
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error
def build_model(inputs,outputs,hidden_layer_count,node_count,decay):
    # create model
    model = Sequential()
    #
    model.add(Dense(node_count, input_dim=inputs, kernel_initializer='he_uniform',activation='sigmoid'))
    model.add(Dropout(0.2))
    if(hidden_layer_count>0):
        for i in range(hidden_layer_count):
            model.add(Dense(node_count, activation='sigmoid'))
            model.add(Dropout(0.2))


    model.add(Dense(outputs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    #print(model.summary())
    #print('')
    return model


def format_and_split_data(df):
    # format data inside
    to = ['ethnicity', 'gender', 'died', 'disease_activity']
    for n in to:
        df[n] = pd.Categorical(df[n])
        df[n] = df[n].cat.codes

    # seperate dataframe into data and targ
    '''data_columns = ['agefu','ethnicity','gender','weight','bmi','crpscore','Days_on_Steroids','Days_on_DMARDs',
                    'Days_on_Biologics','esr_score','das28','smokestatus','disease_activity',
                    'prev_ang_status','prev_bp_status','prev_cancer_status','prev_depress_status','prev_diab_status',
                    'prev_disc_status','prev_htattk_status',
                    'prev_lung_status']

    target_columns = ['ang_status', 'bp_status', 'htattk_status',  'diab_status',
                    'cancer_status', 'disc_status', 'depress_status',
                      'lung_status']'''

    data_columns = ['agefu', 'ethnicity', 'gender', 'weight',
                    'bmi', 'crpscore', 'Days_on_Steroids',
                    'Days_on_DMARDs','Days_on_Biologics', 'esr_score',
                    'das28', 'smokestatus', 'disease_activity',
                    'time', 'prev_ang_status', 'prev_bp_status',
                    'prev_cancer_status', 'prev_depress_status','prev_diab_status',
                    'prev_disc_status', 'prev_glauc_status', 'prev_htattk_status',
                    'prev_htfail_status','prev_kidney_status',
                    'prev_liver_status', 'prev_lung_status', 'prev_stroke_status']
    target_columns = ['ang_status', 'bp_status', 'htattk_status',
                      'htfail_status', 'stroke_status', 'diab_status',
                      'liver_status', 'kidney_status', 'cancer_status',
                      'disc_status', 'depress_status',
                      'lung_status', 'glauc_status']

    data = df[data_columns]
    target = df[target_columns]

    # convert to nump array (with groupings for recurrent)
    data = np.asarray(data)
    target = np.asarray(target)

    return data,target


def k_fold_person_safe_param_finder(df, k,layer_count,node_count,weight):
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

    for n in range(k):
        test_ids = patient_list[start:start+patients_per_fold+1]
        start = start+patients_per_fold
        train_ids = [x for x in patient_list if x not in test_ids]
        test = df.loc[df['regno'].isin(test_ids)]
        train = df.loc[df['regno'].isin(train_ids)]

        train_x,train_y = format_and_split_data(train)
        test_x,test_y = format_and_split_data(test)
        model = build_model(27,13,layer_count,node_count,weight)
        hist = model.fit(train_x, train_y, batch_size=32, epochs=20)
        #plt.plot(hist.history['loss'])
        #plt.show()
        yhat = model.predict(test_x)
        local_acc = accuracy_score(test_y, np.abs(yhat.round()))
        hamming_loss_local = hamming_loss(test_y, np.abs(yhat.round()))
        recall_tot_local = sklearn.metrics.recall_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        precision_tot_local = sklearn.metrics.precision_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
        f1_tot_local = sklearn.metrics.f1_score(y_true=test_y, y_pred=np.abs(yhat.round()), average='samples')
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


def weight_decay_search(df):
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
        print(all_f1[i])

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



df = pd.read_csv("NOAR-newinp-3.csv")
todel = remove_zero(df)
for n in todel:
    df.drop(df.loc[df['regno'] ==n].index, inplace=True)

grid_search_no_decay(df)
#grid_search(df)
'''acc, ham_loss, rec, prec, f1 = k_fold_person_safe_param_finder(df, 5, 0, 200, 1e-06)
print(acc)
print(ham_loss)
print(rec)
print(prec)
print(f1)'''


'''
1e-06 : weight decay
0.7246887339558434
0.034594144611388954
0.5222807149555143
0.5266917138026259
0.5154590497239875

'''

'''
1e-06 : weight decay 20 epoch
0.7271823808919538
0.031022969861004595
0.5297559314060099
0.5360463680791147
0.5247455543551973
'''


'''
1e-06 : weight decay 35 epoch
0.7625259960041684
0.028479452093892115
0.5412127704625231
0.5486128453876
0.5380804223630491

1e-06 : weight decay 35 epoch, sigmoid
0.7924147106727102
0.023103818131342657
0.5337449830278539
0.5593635727004074
0.5411860647259278
'''

'''
200 hidden 0.2 drop
0.7874599199743659
0.023766365715945956
0.5177667286955543
0.5458034397884601
0.5260740858489918

200 hidden 0.2 drop 35 epoch
0.795953013178058
0.022582089476269183
0.5274981386570583
0.5497865213169331
0.5336580742723436
'''