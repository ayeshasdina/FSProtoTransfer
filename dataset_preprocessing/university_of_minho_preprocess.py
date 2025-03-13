import pandas as pd
import os

dataset_folder = "/home/colbyjacobe/Documents/Research/Datasets/University of Minho/vanet_datasets"

final_train_file = "../datasets/train_university_of_minho.csv"
final_val_file = "../datasets/val_university_of_minho.csv"
final_test_file = "../datasets/test_university_of_minho.csv"

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

dirs = [
    os.path.join(dataset_folder, "dos_datasets"),
    os.path.join(dataset_folder, "random_info_datasets/random_info_datasets_speed"),
    os.path.join(dataset_folder, "random_info_datasets/random_info_datasets_acc"),
    os.path.join(dataset_folder, "random_info_datasets/random_info_datasets_heading"),
]

final_dirs = []
for dir in dirs:
    final_dirs.extend(sorted(list_files(dir)))

columns = ['receiverTime', 'diffTime', 'heading', 'speed', 'longAcceleration', 'latitude', 'longitude', 'bitLen', 'diffPos', 'diffSpeed', 'diffHeading', 'diffAcc', 'isAttack']

train_df = pd.DataFrame(columns=[2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17])
val_df = pd.DataFrame(columns=[2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17])
test_df = pd.DataFrame(columns=[2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17])
for file in final_dirs:
    df = pd.read_csv(file, header=None, usecols=[2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17])

    if "out_7" in file:  # Append data from final scenario to testing set.
        print("Appending to test set: ", file)
        test_df = pd.concat([test_df, df])
    elif "out_6" in file:
        print("Appending to val set: ", file)
        val_df = pd.concat([val_df, df])
    else:
        print("Appending to train set: ", file)
        train_df = pd.concat([train_df, df])

    del df

print("Setting column names...")
train_df.columns = columns
val_df.columns = columns
test_df.columns = columns

print("Sorting values...")
train_df = train_df.sort_values(by=['receiverTime'])
val_df = val_df.sort_values(by=['receiverTime'])
test_df = test_df.sort_values(by=['receiverTime'])

print("Dropping Timestamp column...")
train_df.drop(columns=['receiverTime'], inplace=True)
val_df.drop(columns=['receiverTime'], inplace=True)
test_df.drop(columns=['receiverTime'], inplace=True)

print("Resetting indices...")
train_df.reset_index(inplace=True, drop=True)
val_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True)

print("Train: ", train_df.head())
print("Val: ", val_df.head())
print("Test: ", test_df.head())

print("Train Value Counts: ", train_df['isAttack'].value_counts())
print("Val Value Counts: ", val_df['isAttack'].value_counts())
print("Test Value Counts: ", test_df['isAttack'].value_counts())

train_df.to_csv(final_train_file)
val_df.to_csv(final_val_file)
test_df.to_csv(final_test_file)
