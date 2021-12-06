from glob import glob
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import  re


if __name__ == '__main__':
    # s = '<talkid>639</talkid>'
    # pattern = '<[/A-Za-z]*>'
    # b = re.findall(pattern,s)
    # print(b)
    # print(len(b))
    # exit()

    pattern = '<[A-Za-z]*>'
    paths = glob('*/*/*.zh2en')
    print(paths)
    src_texts = []
    tar_texts = []
    labels = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip('\n').split('\t')
            if line[0] != line[1] and '<url>' not in line[0] and '</url>' not in line[0]:
                if len(re.findall(pattern,line[0])) > 0 :
                    if re.findall(pattern,line[0]) == re.findall(pattern,line[1]):
                        src_texts.append(line[0].replace('【', '[').replace('】', ']'))
                        tar_texts.append(line[1])
                        if 'medical' in path:
                            labels.append(0)
                        elif 'oral' in path:
                            labels.append(1)
                        else:
                            labels.append(2)
                else:
                    src_texts.append(line[0].replace('【', '[').replace('】', ']'))
                    tar_texts.append(line[1])
                    if 'medical' in path:
                        labels.append(0)
                    elif 'oral' in path:
                        labels.append(1)
                    else:
                        labels.append(2)



    print(len(src_texts))

    df = pd.DataFrame()
    df['src'] = src_texts
    df['tar'] = tar_texts
    df['id'] = list(range(len(tar_texts)))


    sk = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

    for (train_index,dev_index) in sk.split(tar_texts,labels):
        print(train_index,'----',dev_index)
        print('*'*100)
        train_df = df[df['id'].isin(train_index)][['src','tar']]
        dev_df = df[df['id'].isin(dev_index)][['src','tar']]
        print(len(train_df))
        print(len(dev_df))

        train_df.to_csv('train_dataset.csv',index=False,sep='\t',header=False)

        dev_df.to_csv('dev_dataset.csv', index=False, sep='\t', header=False)
        exit()
        