from sklearn.model_selection import StratifiedKFold,train_test_split
import os
import pandas as pd
import numpy as np

NUM_FOLD=5
CSV_FOLDER="/media/tungthanhlee/SSD/alaska/csv"


BASE_PATH = "/media/tungthanhlee/SSD/alaska/alaska2-image-steganalysis"
PATH = "alaska2-image-steganalysis"


def append_path(pre):
    return np.vectorize(lambda file: os.path.join(PATH, pre, file))

def make_df():
    """
    Make training dataframe
    """
    #Get all file name of cover
    train_filenames = np.array(os.listdir(BASE_PATH+"/Cover/"))

    #https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
    np.random.seed(42)
    #clone negative and positives
    positives = train_filenames.copy()
    negatives = train_filenames.copy()
    np.random.shuffle(positives)
    np.random.shuffle(negatives)

    #create positive samples
    jmipod = append_path('JMiPOD')(positives)
    juniward = append_path('JUNIWARD')(positives)
    uerd = append_path('UERD')(positives)
    pos_paths = np.concatenate([jmipod, juniward, uerd])

    #create negative samples
    neg_paths = append_path('Cover')(negatives)

    #create train path and train label
    train_paths = np.concatenate([pos_paths, neg_paths])
    train_labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))
    
    #training dataframe
    df = pd.DataFrame({ 'images': list(train_paths), 'label': train_labels},columns=['images','label'])
    
    return df

def make_df_4class():
    """
    Make training dataframe with 4 classes
    {
        1: 'JMiPOD',
        2: 'JUNIWARD',
        3: 'UERD'
    }
    """
    train_filenames = np.array(os.listdir(BASE_PATH+"/Cover/"))

    #https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
    np.random.seed(42)
    #clone negative and positives
    positives = train_filenames.copy()
    negatives = train_filenames.copy()
    np.random.shuffle(positives)
    np.random.shuffle(negatives)

    #create positive samples
    jmipod = append_path('JMiPOD')(positives)
    juniward = append_path('JUNIWARD')(positives)
    uerd = append_path('UERD')(positives)

    #create negative samples
    neg_paths = append_path('Cover')(negatives)

    #create train path and train label
    train_paths = np.concatenate([jmipod, juniward, uerd, neg_paths])
    train_labels = np.array([1] * len(jmipod) +
                            [2] * len(juniward) +
                            [3] * len(uerd) +
                            [0] * len(neg_paths))
    
    #training dataframe
    df = pd.DataFrame({ 'images': list(train_paths), 'label': train_labels},columns=['images','label'])

    return df


def split(csv_folder, df):
    """
    Kfold split
    """
    skf = StratifiedKFold(n_splits=NUM_FOLD, random_state=42, shuffle=True)
    X, y = df['images'], df['label']
    # folds_path = os.path.join(csv_folder, 'Folds')
    if not os.path.isdir(csv_folder):
        os.mkdir(csv_folder)
        
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        df_train = pd.DataFrame(zip(X_train, y_train), columns=('images', 'label'))
        df_val = pd.DataFrame(zip(X_val, y_val), columns=('images', 'label'))
        # df_train.to_csv(os.path.join(csv_folder, f'train_fold{fold_idx}.csv'), index=False)
        # df_val.to_csv(os.path.join(csv_folder, f'valid_fold{fold_idx}.csv'), index=False)
        df_train.to_csv(os.path.join(csv_folder, f'train4classes_fold{fold_idx}.csv'), index=False)
        df_val.to_csv(os.path.join(csv_folder, f'valid4classes_fold{fold_idx}.csv'), index=False)

def train_valid_split(csv_folder, df):
    """
    Split train/valid
    """
    X, y = df['images'], df['label']
    if not os.path.isdir(csv_folder):
        os.mkdir(csv_folder)

    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
                                                X, y, test_size=0.03, random_state=2020, stratify=y)
    df_train = pd.DataFrame(zip(train_paths, train_labels), columns=('images', 'label'))
    df_valid = pd.DataFrame(zip(valid_paths, valid_labels), columns=('images', 'label'))

    df_train.to_csv(os.path.join(csv_folder, f'train.csv'), index=False)
    df_valid.to_csv(os.path.join(csv_folder, f'valid.csv'), index=False)


if __name__ == "__main__":
    df = make_df_4class()
    # print(df)
    # split(CSV_FOLDER, df)
    train_valid_split(CSV_FOLDER, df)
