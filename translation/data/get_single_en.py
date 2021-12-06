import pandas as pd
from glob import glob
if __name__ == '__main__':
    paths = glob('./*/*/*.en')
    texts = []
    for  path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [ line.strip('\n') for line in lines]
        texts.extend(lines)

    df = pd.DataFrame()
    df['text'] = texts
    print('len(df)',len(df))
    df.to_csv('single_en.csv',index=False,header=False)

