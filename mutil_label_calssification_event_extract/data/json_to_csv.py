import json
import pandas as pd

if __name__ == '__main__':
    event_types_set = []
    with open('train_dataset.json','r',encoding='utf-8') as f:
        lines = f.readlines()

    train_texts = []
    train_event_types =[]
    train_ids = []

    for line in lines:
        line = json.loads(line)
        id = line['id']
        text = line['text']
        train_texts.append(text)
        train_ids.append(id)
        temp_events = []
        for ele in line['event_chain']:
            event_type = ele['event_type']
            temp_events.append(event_type)
            event_types_set.append(event_type)
        temp_events = '\t'.join(temp_events)
        train_event_types.append(temp_events)



    with open('dev_dataset.json','r',encoding='utf-8') as f:
        lines = f.readlines()

    dev_texts = []
    dev_event_types =[]
    dev_ids = []

    for line in lines:
        line = json.loads(line)
        id = line['id']
        text = line['text']
        dev_texts.append(text)
        dev_ids.append(id)
        temp_events = []
        for ele in line['event_chain']:
            event_type = ele['event_type']
            temp_events.append(event_type)
            event_types_set.append(event_type)
        temp_events = '\t'.join(temp_events)
        dev_event_types.append(temp_events)

    event_types_set = list(set(event_types_set))
    event_types_set.sort()
    print('len(event_types_set)',len(event_types_set))
    with open('event_types.txt','w',encoding='utf-8') as f:
        for ele in event_types_set:
            f.write(ele+'\n')

    train_labels = []
    for train_event_type in train_event_types:
        temp_label = []
        for ele in event_types_set:
            if ele in train_event_type:
                temp_label.append(str(1))
            else:
                temp_label.append(str(0))
        temp_label = '\t'.join(temp_label)
        train_labels.append(temp_label)


    train_df = pd.DataFrame()
    train_df['id'] = train_ids
    train_df['text'] = train_texts
    train_df['label'] = train_labels
    train_df['event_type'] = train_event_types
    lenghts = [ len(ele) for ele in train_texts]
    train_df['text_length'] = lenghts
    train_df.to_csv('train.csv',index=False)


    dev_labels = []
    for dev_event_type in dev_event_types:
        temp_label = []
        for ele in event_types_set:
            if ele in dev_event_type:
                temp_label.append(str(1))
            else:
                temp_label.append(str(0))
        temp_label = '\t'.join(temp_label)
        dev_labels.append(temp_label)

    dev_df = pd.DataFrame()
    dev_df['id'] = dev_ids
    dev_df['text'] = dev_texts
    dev_df['label'] = dev_labels
    dev_df['event_type'] = dev_event_types
    lenghts = [len(ele) for ele in dev_texts]
    dev_df['text_length'] = lenghts
    dev_df.to_csv('dev.csv', index=False)

    
    with open('test_dataset.json','r',encoding='utf-8') as f:
        lines = f.readlines()

    test_texts = []
    test_ids = []

    for line in lines:
        line = json.loads(line)
        id = line['id']
        text = line['text']
        test_texts.append(text)
        test_ids.append(id)

    test_df = pd.DataFrame()
    test_df['id'] = test_ids
    test_df['text'] = test_texts
    test_df['label'] = [str(0)]*len(test_ids)
    lenghts = [len(ele) for ele in test_texts]
    test_df['text_length'] = lenghts
    test_df.to_csv('test.csv',index=False)