import json
import pandas as pd

if __name__ == '__main__':
    'id,text,label,event_type,text_length'
    train_df = pd.read_csv('train.csv')
    print(len(train_df))

    with open('event_types.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    event_types = []
    for line in lines:
        event_types.append(line.strip('\n'))

    predict_df = pd.DataFrame()
    predict_texts = []
    predict_ids = []
    predict_labels = []
    predict_events = []
    with open('../submit/k_fold/kfold_MLCE_L512_bert_vote_submition_1_3_4_best.json') as f:
        lines = f.readlines()

    for line in lines:
        dict = json.loads(line.strip('\n'))
        predict_ids.append(dict['id'])
        predict_texts.append(dict['text'])
        predict_events.append(dict['event_chain'])
        label = []
        for ele in event_types:
            if ele in dict['event_chain']:
                label.append(str(1))
            else:
                label.append(str(0))
        label = '\t'.join(label)
        predict_labels.append(label)
    predict_textlength = [ len(ele) for ele in predict_texts]

    predict_df['id'] = predict_ids
    predict_df['text'] = predict_texts
    predict_df['label'] = predict_labels
    predict_df['event_type'] = predict_events
    predict_df['text_length'] = predict_textlength
    print(len(predict_df))

    df = pd.concat([train_df,predict_df])
    print(len(df))
    df.drop_duplicates(subset=['text'], inplace=True)
    print(len(df))
    df.to_csv('Labeled_and_unlabeled_train.csv',index=False)
