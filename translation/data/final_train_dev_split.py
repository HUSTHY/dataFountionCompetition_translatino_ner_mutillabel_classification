from sklearn.model_selection import KFold
import pandas as pd

if __name__ == '__main__':

    with open('seed_0_en2zh_3W_2021-12-02.txt','r',encoding='utf-8') as f:
        a_lines = f.readlines()

    a_lines = [ line.strip('\n') for line in a_lines]


    with open('../submit/v3_submit_example_zh2en_2021-12-03_post_process.txt','r',encoding='utf-8') as f:
        tag_lines = f.readlines()

    tag_lines = [ line.strip('\n') for line in tag_lines]


    with open('test_dataset.csv','r',encoding='utf-8') as f:
        src_lines = f.readlines()

    src_lines = [ line.strip('\n') for line in src_lines]

    b_lines = [  src+'\t'+tag  for src,tag in zip(src_lines,tag_lines)]



    a_lines.extend(b_lines)


    df = pd.DataFrame()

    df['text'] = a_lines

    df['id'] = list(range(len(df)))

    sk = KFold(n_splits=10, random_state=1, shuffle=True)

    for (train_index, dev_index) in sk.split(a_lines):
        print(train_index, '----', dev_index)
        print('*' * 100)
        train_df = df[df['id'].isin(train_index)][['text']]
        dev_df = df[df['id'].isin(dev_index)][['text']]
        print(len(train_df))
        print(len(dev_df))

        train_df.to_csv('train_dataset_final.csv', index=False, sep='\t', header=False)

        dev_df.to_csv('dev_dataset_final.csv', index=False, sep='\t', header=False)
        exit()