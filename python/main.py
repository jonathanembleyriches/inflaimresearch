import pandas as pd
import numpy as np
global status_counts,ethnicity_counts,gender_counts,smoke_counts,disease_counts, bmi, crpscore,esr_score,das28,do_steroids,do_bmards,do_biologics
status_counts = { 'ang_status' : 0,
'bp_status': 0,
'htattk_status': 0,
'htfail_status': 0,
'stroke_status': 0,
'diab_status': 0,
'liver_status': 0,
'kidney_status': 0,
'cancer_status': 0,
'disc_status': 0,
'depress_status': 0,
'lung_status': 0,
'glauc_status': 0}

ethnicity_counts = {}
gender_counts = {}
smoke_counts = {}
disease_counts = {}
bmi = []
crpscore=[]
esr_score = []
das28 = []
do_steroids = []
do_bmards = []
do_biologics=[]


def load_data(filename):
    return pd.read_csv(filename)

def increase_counts(row_info):
    for key in status_counts:
        status_counts[key] += row_info[key]

    #add ethnicity
    if row_info['ethnicity'] not in ethnicity_counts:
        ethnicity_counts[row_info['ethnicity']] = 1
    else:
        ethnicity_counts[row_info['ethnicity']] =  ethnicity_counts[row_info['ethnicity']] +1

    # add gender
    if row_info['gender'] not in gender_counts:
        gender_counts[row_info['gender']] = 1
    else:
        gender_counts[row_info['gender']] = gender_counts[row_info['gender']] + 1

    # add smoke counts
    if row_info['smokestatus'] not in smoke_counts:
        smoke_counts[row_info['smokestatus']] = 1
    else:
        smoke_counts[row_info['smokestatus']] = smoke_counts[row_info['smokestatus']] + 1

    # add disease counts
    if row_info['disease_activity'] not in disease_counts:
        disease_counts[row_info['disease_activity']] = 1
    else:
        disease_counts[row_info['disease_activity']] = disease_counts[row_info['disease_activity']] + 1

def data_saver(data,file_name):
    df = pd.DataFrame(data=data, index=[0])

    df = (df.T)

    print(df)

    df.to_excel(f'{file_name}.xlsx')


if __name__ == '__main__':

    total = 0
    data = load_data("imputed_dataset1_NOAR.csv")
    print(f"lrnghto of df {data.shape[0]}")
    male_c =0
    prev_reg_no = 0
    #for i in range(0, data.shape[0]):
    #    if prev_reg_no != data.loc[i,'regno']:
    #        increase_counts(data.loc[i])
    #        prev_reg_no = data.loc[i,'regno']

    #data_saver(status_counts, 'disease status first encounter')

    for i in range(1, data.shape[0]):
        #increase_counts(data.loc[i])
        #print(f"checking {data.loc[i,'regno']} against {data.loc[i-1,'regno']}")
        if data.loc[i,'regno'] != data.loc[i-1,'regno'] and data.loc[i-1,'crpscore'] > 14:
            print(f"last one ^^ disc status {data.loc[i-1,'disc_status']}")
            bmi.append(data.loc[i-1,'bmi'])
            crpscore.append(data.loc[i-1,'crpscore'])
            esr_score.append(data.loc[i-1,'esr_score'])
            das28.append(data.loc[i-1,'das28'])
            do_steroids.append(data.loc[i-1,'Days_on_Steroids'])
            do_bmards.append(data.loc[i-1,'Days_on_DMARDs'])
            do_biologics.append(data.loc[i-1,'Days_on_Biologics'])
            total+=1
            increase_counts(data.loc[i-1])


    print(f"Total unique people {total}")
    print(f'total male {male_c}')
    print("Final disease status frequencies")
    print(status_counts)
    data_saver(status_counts,'disease status crpscore gt14')

    print("Final ethnicity frequencies")
    print(ethnicity_counts)
    #data_saver(ethnicity_counts,'ethnicity')

    print("Final gender frequencies")
    print(gender_counts)
    #data_saver(gender_counts,'gender')


    print("Final smoke frequencies")
    print(smoke_counts)
    #data_saver(smoke_counts,'smoke')


    print("Final disease frequencies")
    print(disease_counts)
    #data_saver(disease_counts,'disease severity')

    #data_saver(bmi, 'bmi')
    """pd.DataFrame(bmi).to_excel('bmi2.xlsx', header=False, index=False)
    pd.DataFrame(crpscore).to_excel('crpscore2.xlsx', header=False, index=False)
    pd.DataFrame(esr_score).to_excel('esr_score2.xlsx', header=False, index=False)
    pd.DataFrame(das28).to_excel('das282.xlsx', header=False, index=False)
    pd.DataFrame(do_steroids).to_excel('do_steroids2.xlsx', header=False, index=False)
    pd.DataFrame(do_bmards).to_excel('do_bmards2.xlsx', header=False, index=False)
    pd.DataFrame(do_biologics).to_excel('do_biologics2.xlsx', header=False, index=False)"""

