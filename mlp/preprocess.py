import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

# Standardscaler
def ss(train):
    """
    Function to create standardscaler based on trainset to be used on train/test set (prevent information leak from test set)
    :param train: train dataset
    :return: the fitted standardscaler on numeric datapoints
    """
    num_col = ['temperature', 'feels-like-temperature', 'relative-humidity', 'windspeed', 'psi', 'guest-users', 'registered-users']
    ssfit = StandardScaler().fit(train[num_col].drop_duplicates())
    return ssfit

def user_attributes(dataframe):
    """
    Function to get user attributes from trainset to be added to train/test set (prevent information leak from test set)
    :param user_df: dataframe with guest & registered user demand
    :return: train_user_attr - guest & registered user demand attributes (mean, stddev, quartiles) from dataframe
    """
    # Negative users (target) values to be replaced with 0.
    user_df = dataframe.copy()
    user_df['guest_users'] = user_df['guest-users'].apply(lambda x: 0 if x < 0 else x)
    user_df['registered_users'] = user_df['registered-users'].apply(lambda x: 0 if x < 0 else x)
    # Create day time features
    user_df['date'] = pd.to_datetime(user_df['date'])
    user_df['date_time'] = pd.to_datetime(user_df.date) + user_df.hr.astype('timedelta64[h]')
    user_df['day'] = user_df['date_time'].apply(lambda x: x.dayofweek)
    user_df['hr'] = user_df['date_time'].apply(lambda x: x.hour)
    # Duplicated datapoints to be removed.
    df.drop_duplicates(inplace=True)
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    # attributes of guest users
    train_guest_attr = user_df.groupby(['day', 'hr'])['guest_users'].agg(
        ['mean', 'median', 'std', 'min', 'max', percentile(25), percentile(75)]).reset_index()
    train_guest_attr.columns = ['day', 'hr'] + ['guest_' + col for col in ['mean', 'median', 'std', 'min', 'max', '25th', '75th']]
    # attributes of registered users
    train_register_attr = user_df.groupby(['day', 'hr'])['registered_users'].agg(
        ['mean', 'median', 'std', 'min', 'max', percentile(25), percentile(75)]).reset_index()
    train_register_attr.columns = ['day', 'hr'] + ['register_' + col for col in ['mean', 'median', 'std', 'min', 'max', '25th', '75th']]
    train_user_attr = train_guest_attr.merge(train_register_attr, on=['day', 'hr'])

    # Standardscaler numeric
    num_cols = train_user_attr.drop(['day', 'hr'], axis=1).columns.tolist()
    Xs = StandardScaler().fit_transform(train_user_attr[num_cols])
    Xs = pd.DataFrame(Xs, columns=num_cols)
    Xs.set_index(train_user_attr.index, inplace=True)
    train_user_attr_ss = pd.concat([train_user_attr.drop(num_cols, axis=1), Xs], axis=1)

    return train_user_attr_ss


def preprocess_df(ss_fit, train_user_attr_ss, dataframe):
    """
    Function to preprocess train/test dataset
    :param ss_fit: standardized scaler fitted on training set to be used for preprocessing of dataframe
    :param train_user_attr_ss: standardized attributes of guest & registered user demand to be added to dataframe
    :param dataframe: dataset to be preprocessed
    :return: df_clean - a cleaned dataset after feature creation, removal of duplicates, standardizing of numerical values and dummying of categorical values
    """

    df = dataframe.copy()
    df.rename(columns={'guest-users': 'guest_users',
                       'registered-users': 'registered_users',
                       'feels-like-temperature': 'feels_like_temperature',
                       'relative-humidity': 'relative_humidity'},inplace=True)

    # Negative users (target) values to be replaced with 0.
    df['guest_users'] = df['guest_users'].apply(lambda x: 0 if x < 0 else x)
    df['registered_users'] = df['registered_users'].apply(lambda x: 0 if x < 0 else x)

    # Duplicated datapoints to be removed.
    df.drop_duplicates(inplace=True)

    # Missing datapoints to be removed (days with < 22 datapoints) and forward filled (days with >= 22 datapoints)
    # create dummy range of datetime
    df['date'] = pd.to_datetime(df['date'])
    df['date_time'] = pd.to_datetime(df.date) + df.hr.astype('timedelta64[h]')
    date_rng = pd.date_range(start=df['date_time'].min(), end=df['date_time'].max(), freq='H')
    df_full = pd.DataFrame(date_rng, columns=['date_time']).merge(df, how='left')
    # convert to date and hr
    df_full['date'] = df_full['date_time'].apply(lambda x: x.date())
    df_full['hr'] = df_full['date_time'].apply(lambda x: x.hour)
    # drop dates with less than 22 datapoints
    date_count = df['date'].value_counts().reset_index()
    dates_to_remove = date_count[date_count['date'] < 22]['index'].apply(lambda x: x.date()).tolist()
    df_full = df_full[~df_full['date'].isin(dates_to_remove)]
    # forward fill the remaining nulls with previous hour values
    df_full.ffill(axis=0, inplace=True)
    df_full.dropna(inplace=True)

    # Rectify weather category mispellings
    df_full['weather'] = df_full.weather.apply(lambda x: x.upper())
    df_full['weather'] = df_full.weather.apply(lambda x: 'CLOUDY' if x in 'CLOUDY' else
    'CLEAR' if x in 'CLEAR' else
    'RAIN' if x in 'LIGHT SNOW/RAIN' else
    'RAIN' if x in 'HEAVY SNOW/RAIN' else x)

    # Create time features
    df_full['day'] = df_full['date_time'].apply(lambda x: x.dayofweek)
    df_full['month'] = df_full['date_time'].apply(lambda x: x.month)
    # Break day into different parts
    df_full['daypart'] = df_full.hr.apply(lambda x: 'morn' if (x >= 6 and x <= 10) else
    'aftern' if (x >= 11 and x <= 15) else
    'eve' if (x >= 16 and x <= 20) else
    'night' if (x >= 21 or x == 0) else
    'postmid' if (x >= 1 and x <= 5) else 0)

    # Add user attributes
    df_merge = df_full.merge(train_user_attr_ss, on=['day', 'hr'], how='left')

    # Create 24lags of user demand as features
    df_merge['gs_user_lag'] = df_merge['guest_users'].shift(24)
    df_merge['reg_user_lag'] = df_merge['registered_users'].shift(24)

    # Standardscaler numeric
    num_cols = ['temperature', 'feels_like_temperature', 'relative_humidity', 'windspeed', 'psi','gs_user_lag','reg_user_lag']
    Xs = ssfit.transform(df_merge[num_cols])
    Xs = pd.DataFrame(Xs, columns=num_cols)
    Xs.set_index(df_merge.index, inplace=True)
    df_ss = pd.concat([df_merge.drop(num_cols, axis=1), Xs], axis=1)

    # Create dummy variables
    df_ss['wind'] = df_ss['windspeed'].apply(lambda x: 1 if x > 0 else 0)
    df_ss['hr'] = df_ss['hr'].astype('category')
    df_ss['day'] = df_ss['day'].astype('category')
    df_ss['month'] = df_ss['month'].astype('category')
    df_ss.set_index('date_time', inplace=True)
    df_ss_dum = pd.get_dummies(df_ss.drop('date', axis=1), drop_first=True)

    # Get interaction terms
    daypart_dum = df_ss_dum.filter(like='daypart').columns.to_list()
    # time of day and humidity
    daypart_hum = df_ss_dum[daypart_dum].multiply(df_ss_dum["relative_humidity"], axis="index")
    daypart_hum.columns = [str(col) + '_humd' for col in daypart_hum.columns]
    # time of day and temperature
    daypart_temp = df_ss_dum[daypart_dum].multiply(df_ss_dum["temperature"], axis="index")
    daypart_temp.columns = [str(col) + '_temp' for col in daypart_temp.columns]
    # time of day and weekend
    df_ss_dum['weekend'] = df_ss_dum['day_5'] + df_ss_dum['day_6']
    weekend_daypart = df_ss_dum[daypart_dum].multiply(df_ss_dum['weekend'], axis="index")
    weekend_daypart.columns = [str(col) + '_weekend' for col in weekend_daypart.columns]
    df_ss_dum_inter = pd.concat([df_ss_dum.drop('weekend', axis=1), daypart_hum, daypart_temp, weekend_daypart], axis=1)

    # Transform target variables
    df_ss_dum_inter['target'] = df_ss_dum_inter['registered_users'] + df_ss_dum_inter['guest_users']
    # log and daily-lag difference of target variables for stationarity
    df_ss_dum_inter['lg_reg_user'] = df_ss_dum_inter['registered_users'].apply(lambda x: np.log(x + 0.1))
    df_ss_dum_inter['lg_gs_user'] = df_ss_dum_inter['guest_users'].apply(lambda x: np.log(x + 0.1))
    df_ss_dum_inter['diff_lg_reg_user'] = df_ss_dum_inter['lg_reg_user'] - df_ss_dum_inter['lg_reg_user'].shift(24)
    df_ss_dum_inter['diff_lg_gs_user'] = df_ss_dum_inter['lg_gs_user'] - df_ss_dum_inter['lg_gs_user'].shift(24)
    df_clean = df_ss_dum_inter.copy()

    return df_clean

def test_column_check(train_clean, test_clean):
    """
    Create dummy columns if training features are not found in test set
    and to drop columns if test features are not found in train set
    :param train_clean: the processed training set
    :param test_clean: the processed test set
    :return: test dataframe with the same columns as train set
    """
    train_cols = train_clean.columns
    # Get test set missing columns from the training set
    missing_cols = [col for col in train_cols if col not in test_clean.columns]
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test_clean[c] = 0
    # Remove additional columns in test set but not in train set
    # Ensure the order of column in the test set is in the same order than in train set
    test_clean_check = test_clean[train_cols]

    return test_clean_check

def output_clean_x_y(target, train_clean, test_clean_check):
    """
    Function to split train/test data into X,y numpy files as model inputs
    :param target: target variable for the dataframe
    :param train_clean: cleaned train set to split into X,y
    :param test_clean_check: cleaned test set to split into X,y
    :return: numpy data files in the data folder
    """
    X_train = train_clean.drop(target,axis=1).values
    X_test = test_clean_check.drop(target, axis=1).values
    y_train = train_clean[target].values
    y_test = test_clean_check[target].values
    X_col_names = train_clean.drop(target,axis=1).columns
    y_col_names = target

    to_save_dct = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
                   'X_col_names': X_col_names, 'y_col_names': y_col_names}
    for key, file in to_save_dct.items():
        np.save('./mlp/process_data/' + key, file)

#read dataset
df = pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap6-assessment-data/scooter_rental_data.csv')
df['date_time'] = pd.to_datetime(df.date) + df.hr.astype('timedelta64[h]')
df.sort_values('date_time',inplace=True)

#train test split
train = df[:int(np.round(len(df)*0.8))]
test =df[int(np.round(len(df)*0.8))-24:] #due to 24hr differencing, require a day of data from training

#run functions
ssfit = ss(train)
train_user_attr_ss = user_attributes(train)
clean_train = preprocess_df(ssfit, train_user_attr_ss, train)
clean_test = preprocess_df(ssfit, train_user_attr_ss, test)
test_clean_check = test_column_check(clean_train, clean_test)
target_cols = ['guest_users','registered_users','target','lg_reg_user', 'lg_gs_user', 'diff_lg_reg_user', 'diff_lg_gs_user']
output_clean_x_y(target_cols, clean_train, test_clean_check)