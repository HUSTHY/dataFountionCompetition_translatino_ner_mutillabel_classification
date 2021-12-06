import torch
def label_smooth(label, n_class=3,alpha=0.1):
    """
    标签平滑
    :param label: 真实lable
    :param n_class: 类别数目
    :param alpha: 平滑系数
    :return:
    """
    k = alpha / (n_class - 1)
    # temp [batch_size,n_class]
    temp = torch.full((label.shape[0], n_class), k).to(label.data.device)
    # scatter_.(int dim, Tensor index, Tensor src),这个函数比较难理解——用src张量根据dim和index来修改temp中的元素
    temp = temp.scatter_(1, label.unsqueeze(1), (1-alpha))
    return temp

if __name__ == '__main__':
    label = [0,1,1,2,1,0]
    label = torch.as_tensor(label,dtype=torch.long)
    print(label)
    label = label_smooth(label)
    print(label)