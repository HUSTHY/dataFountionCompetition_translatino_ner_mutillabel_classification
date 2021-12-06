import pandas as pd
if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    dev_df = pd.read_csv('dev.csv')
    new_df = pd.concat([train_df,dev_df],axis=0)
    new_df.to_csv('train_dev.csv',index=False)