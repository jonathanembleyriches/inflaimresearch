import pandas as pd
import pandas_profiling


def repreat_data(df):

    # Get all unique patient ids
    patient_list = df['id'].unique()

    new_df = df.head()
    # iter over all patients
    for patient_id in patient_list:
        # get all data regarding 1 user
        subframe = df[df.id.isin([patient_id])].head()

        # sort by age_i
        subframe = subframe.sort_values(by=['age_i'])

        # create consistent gap
        for i in range(1,subframe.shape[0]):
            prev_age = df.iat[i-1,'age_i']
            this_age = df.iat[i,'age_i']
            gap = this_age-prev_age
            if gap > 1:
                # repeat i-1 row
        print(subframe)
        pass


df = pd.read_csv("data/biobank.csv")
repreat_data(df)
