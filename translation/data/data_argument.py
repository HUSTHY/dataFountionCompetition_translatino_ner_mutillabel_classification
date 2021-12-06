import pandas as pd

if __name__ == '__main__':
    with open('train_dataset.csv','r',encoding='utf-8') as f:
        trian_lines = f.readlines()



    with open('../submit/v3_seed1_en2zh_2021-12-02.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    srcs = [ line.strip('\n') for line in lines]


    with open('single_en.csv','r',encoding='utf-8') as f:
        lines = f.readlines()
    tags = [ line.strip('\n') for line in lines]

    print('len(srcs)',len(srcs))
    assert len(srcs) == len(tags)


    with open('train_dataset_argument.csv','w',encoding='utf-8') as f:
        for trian_line in trian_lines:
            f.write(trian_line.replace('\n','')+'\n')

        for src, tag in zip(srcs,tags):
            if len(src) < 512:
                f.write(src.replace('\n','') + '\t' + tag.replace('\n','') + '\n')

    with open('train_dataset_argument.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print('len(lines)',len(lines))