import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# script parameters
SEED = 1239
output_file = "model.bin"
generate_kaggle_submission = True

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

def transform_date(df, col):
    df = df.copy()
    df[f'{col}_year'] = df[col].dt.year.astype('float32')
    df[f'{col}_month'] = df[col].dt.month.astype('float32')
    df[f'{col}_day'] = df[col].dt.day.astype('float32')
    df[f'{col}_day_of_week'] = df[col].dt.dayofweek.astype('float32')
    
    df[f'{col}_year_sin'] = np.sin(2 * np.pi * df[f'{col}_year'])
    df[f'{col}_year_cos'] = np.cos(2 * np.pi * df[f'{col}_year'])
    df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12) 
    df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12)
    return df

# remove null rows
df_train_clean = df_train.dropna()
# df_test_clean = df_test.dropna()

df_train_clean = transform_date(df_train_clean, 'date')
df_test_clean = transform_date(df_test, 'date')

# Label Encode
for col in ['country', 'store', 'product']:
    combined_data = pd.concat([df_train_clean[col], df_test_clean[col]])
    le = LabelEncoder()
    le.fit(combined_data)
    df_train_clean[col] = le.transform(df_train_clean[col])
    df_test_clean[col] = le.transform(df_test_clean[col])

y_train = df_train_clean['num_sold']

X_train = df_train_clean.drop(columns=['id', 'num_sold', 'date'])
X_test = df_test_clean.drop(columns=['id', 'date'])

def fit_model_with_tss(X, y, model, n_splits=5):
    tss = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    fold = 1

    for train_idx, val_idx in tss.split(X):
        print(f'    Training fold {fold}...', end='\r')
        fold += 1
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        mape = mean_absolute_percentage_error(y_val, y_pred)
        scores.append(mape)
    print(' '*30, end='\r')
    
    return [np.mean(scores), np.std(scores)]

rf_final = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=SEED,
                                n_jobs=-1, min_samples_leaf=1)
rf_final.fit(X_train, y_train)
final_pred = rf_final.predict(X_test)

# create submission file for kaggle
if generate_kaggle_submission:
    submission = pd.read_csv("data/sample_submission.csv")
    submission['num_sold'] = final_pred
    submission.to_csv('submission.csv', index=False)

# save model
with open(output_file, 'wb') as f_out:
    pickle.dump((rf_final), f_out)

print(f'The model is saved to {output_file}')
