import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    all_labels = []

    df = pd.read_excel('train.xlsx')[['text','BIO_anno']]
    texts = df['text'].values.tolist()
    labels = df['BIO_anno'].values.tolist()
    with open('comment_train.txt','w',encoding='utf-8') as f:
        for text, label in tqdm(zip(texts,labels),desc='train'):
            label = label.split(' ')
            for word, symbol in zip(text,label):
                s = word+'\t'+symbol
                f.write(s+'\n')
                all_labels.extend(label)
            f.write('\n')

    df = pd.read_excel('dev.xlsx')[['text', 'BIO_anno']]
    texts = df['text'].values.tolist()
    labels = df['BIO_anno'].values.tolist()
    with open('comment_dev.txt', 'w', encoding='utf-8') as f:
        for text, label in tqdm(zip(texts, labels),desc='dev'):
            label = label.split(' ')
            for word, symbol in zip(text, label):
                s = word + '\t' + symbol
                f.write(s + '\n')
                all_labels.extend(label)
            f.write('\n')

    # df = pd.read_excel('test.xlsx')[['text']]
    # texts = df['text'].values.tolist()
    # labels = df['text'].values.tolist()
    # i = 0
    #
    # with open('comment_test.txt', 'w', encoding='utf-8') as f:
    #     for text in tqdm(texts,desc='test'):
    #         text = text.replace(' ','，').replace('\t','')
    #         for word in text:
    #             s = word
    #             f.write(s + '\n')
    #         f.write('\n')
    # all_labels = list(set(all_labels))
    # print(all_labels)