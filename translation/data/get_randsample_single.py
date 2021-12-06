import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('single_en.csv')
    print(len(df))
    df = df.sample(n=50000)
    print(len(df))

    df.to_csv('part_single_en_3W.csv',index=False,header=False)