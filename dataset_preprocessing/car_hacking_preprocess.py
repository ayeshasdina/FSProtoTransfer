import pandas as pd
import csv

fieldnames = ['Timestamp', 'ID', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Label']

null_token = 100  # Whenever there is an empty value in the data, fill it with 0x100, the next value after the highest possible value in the dataset.

normal_file = "/home/colbyjacobe/Documents/Research/Datasets/Car-Hacking/normal_run_data.txt"
preprocessed_normal_file = "car_hacking_normal_full.csv"
dos_file = "/home/colbyjacobe/Documents/Research/Datasets/Car-Hacking/DoS_dataset.csv"
preprocessed_dos_file = "car_hacking_dos_full.csv"
fuzzy_file = "/home/colbyjacobe/Documents/Research/Datasets/Car-Hacking/Fuzzy_dataset.csv"
preprocessed_fuzzy_file = "car_hacking_fuzzy_full.csv"
gear_file = "/home/colbyjacobe/Documents/Research/Datasets/Car-Hacking/gear_dataset.csv"
preprocessed_gear_file = "car_hacking_gear_full.csv"
rpm_file = "/home/colbyjacobe/Documents/Research/Datasets/Car-Hacking/RPM_dataset.csv"
preprocessed_rpm_file = "car_hacking_rpm_full.csv"

final_train_file = "../datasets/train_car_hacking.csv"
final_val_file = "../datasets/val_car_hacking.csv"
final_test_file = "../datasets/test_car_hacking.csv"


def split_normal(input_file, preprocessed_file):
    print("Splitting Normal data.")
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(preprocessed_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for line in lines:
            fields = line.strip().split()
            Timestamp = fields[1]
            if Timestamp == '1479121941.287013':
                break  # This timestamp contains no further data.
            ID = fields[3]
            DLC = fields[6]
            Data0 = fields[7] if (int(DLC) > 0) else null_token
            Data1 = fields[8] if (int(DLC) > 1) else null_token
            Data2 = fields[9] if (int(DLC) > 2) else null_token
            Data3 = fields[10] if (int(DLC) > 3) else null_token
            Data4 = fields[11] if (int(DLC) > 4) else null_token
            Data5 = fields[12] if (int(DLC) > 5) else null_token
            Data6 = fields[13] if (int(DLC) > 6) else null_token
            Data7 = fields[14] if (int(DLC) > 7) else null_token
            writer.writerow({
                "Timestamp": Timestamp,
                "ID": ID,
                "DLC": DLC,
                "Data0": Data0,
                "Data1": Data1,
                "Data2": Data2,
                "Data3": Data3,
                "Data4": Data4,
                "Data5": Data5,
                "Data6": Data6,
                "Data7": Data7,
                "Label": 'Normal'
            })

    df = pd.read_csv(preprocessed_file)
    train = df.head(int(len(df.index) * .6))  # Allocate 60% of data to training.
    temp = df.loc[~df.index.isin(train.index)]  # Subset (20% of total data) that contains test and val data.
    val = temp.head(int(len(temp.index) * .25))  # Assign 50% of test/val subset (20% of total data) to val subset.
    test = temp.loc[~temp.index.isin(val.index)]  # Assign remaining 20% of total data to test subset.

    return train, val, test


def split_attack(input_file, preprocessed_file, attack_type):
    print(f"Splitting {attack_type} data.")
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(preprocessed_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for line in lines:
            fields = line.strip().split(',')
            Timestamp = fields[0]
            ID = fields[1]
            DLC = fields[2]
            Data0 = fields[3] if (int(DLC) > 0) else null_token
            Data1 = fields[4] if (int(DLC) > 1) else null_token
            Data2 = fields[5] if (int(DLC) > 2) else null_token
            Data3 = fields[6] if (int(DLC) > 3) else null_token
            Data4 = fields[7] if (int(DLC) > 4) else null_token
            Data5 = fields[8] if (int(DLC) > 5) else null_token
            Data6 = fields[9] if (int(DLC) > 6) else null_token
            Data7 = fields[10] if (int(DLC) > 7) else null_token
            Label = attack_type if fields[-1] == 'T' else 'Normal'
            writer.writerow({
                "Timestamp": Timestamp,
                "ID": ID,
                "DLC": DLC,
                "Data0": Data0,
                "Data1": Data1,
                "Data2": Data2,
                "Data3": Data3,
                "Data4": Data4,
                "Data5": Data5,
                "Data6": Data6,
                "Data7": Data7,
                "Label": Label
            })

    df = pd.read_csv(preprocessed_file)
    train = df.head(int(len(df.index) * .6))  # Allocate 60% of data to training.
    temp = df.loc[~df.index.isin(train.index)]  # Subset (40% of total data) that contains test and val data.
    val = temp.head(int(len(temp.index) * .25))  # Assign 50% of test/val subset (20% of total data) to val subset.
    test = temp.loc[~temp.index.isin(val.index)]  # Assign remaining 20% of total data to test subset.

    return train, val, test

def hex_to_int(value):
    if value != null_token:
        return int(value, 16)
    else:
        return value
def finalize(dataset):
    dataset.replace(['Normal', 'DoS', 'Fuzzy', 'Gear', 'RPM'], [0, 1, 2, 3, 4], inplace=True)  # Convert label to int.
    dataset['ID'] = dataset['ID'].apply(int, base=16)  # Convert hex ID to int.
    for col in dataset.columns:
        if col.startswith('Data'):
            dataset[col] = dataset[col].apply(hex_to_int)
    dataset.sort_index(inplace=True)  # Sort by index number.
    return dataset


# Get training, validation, and testing subsets for each file.
normal_train, normal_val, normal_test = split_normal(normal_file, preprocessed_normal_file)
dos_train, dos_val, dos_test = split_attack(dos_file, preprocessed_dos_file, 'DoS')
fuzzy_train, fuzzy_val, fuzzy_test = split_attack(fuzzy_file, preprocessed_fuzzy_file, 'Fuzzy')
gear_train, gear_val, gear_test = split_attack(gear_file, preprocessed_gear_file, 'Gear')
rpm_train, rpm_val, rpm_test = split_attack(rpm_file, preprocessed_rpm_file, 'RPM')

print("Writing training data.")
train = pd.concat([normal_train, dos_train, fuzzy_train, gear_train, rpm_train], ignore_index=True)
train = finalize(train)
train = train.reindex(columns=['Timestamp', 'ID', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Label'])
train.to_csv(final_train_file)

print("Writing validation data.")
val = pd.concat([normal_val, dos_val, fuzzy_val, gear_val, rpm_val], ignore_index=True)
val = finalize(val)
val = val.reindex(columns=['Timestamp', 'ID', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Label'])
val.to_csv(final_val_file)

print("Writing testing data.")
test = pd.concat([normal_test, dos_test, fuzzy_test, gear_test, rpm_test], ignore_index=True)
test = finalize(test)
test = test.reindex(columns=['Timestamp', 'ID', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Label'])
test.to_csv(final_test_file)

full = pd.concat([train, val, test], ignore_index=True)
