import pandas as pd
if __name__ == '__main__':

    train_df = pd.read_csv('train.csv')
    print(len(train_df))
    train_df.drop_duplicates(subset=['text'],inplace=True)
    print(len(train_df))

    dev_df = pd.read_csv('dev.csv')
    print(len(dev_df))
    dev_df.drop_duplicates(subset=['text'], inplace=True)
    print(len(dev_df))