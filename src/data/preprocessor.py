import pandas as pd


def merge_uah_dataframes(df_acc, df_gps):
    """
    Merges accelerometer and GPS dataframes into a single dataframe.
    """
    df_acc = df_acc.reset_index(drop=True)
    df_gps = df_gps.reset_index(drop=True)

    pd.concat([df_acc, df_gps], axis=1, join='inner', ignore_index=False)


def correct_timestamps(df):

    last_timestamp = 0
    anchor_timestamp = 0
    for i, row in df.iterrows():
        if row['timestamp'] < last_timestamp:
            anchor_timestamp = last_timestamp

        if anchor_timestamp != 0:
            df.at[i, 'timestamp'] = anchor_timestamp + row['timestamp']

        last_timestamp = row['timestamp']
