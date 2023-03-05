import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random

def process_data():
    rawdata = pd.read_csv('/root/zmy/data/data.csv')
    rawarray = (np.array(rawdata))[:,1:]
    set_second = set((rawarray[:,1]).flatten().tolist())
    rawarray[rawarray == 'management'] = 0
    rawarray[rawarray == 'technician'] = 1
    rawarray[rawarray == 'services'] = 2
    rawarray[rawarray == 'unemployed'] = 3
    rawarray[rawarray == 'admin.'] = 4
    rawarray[rawarray == 'housemaid'] = 5
    rawarray[rawarray == 'blue-collar'] = 6
    rawarray[rawarray == 'retired'] = 7
    rawarray[rawarray == 'student'] = 8
    rawarray[rawarray == 'entrepreneur'] = 9
    rawarray[rawarray == 'self-employed'] = 10
    set_third = set((rawarray[:, 2]).flatten().tolist())
    rawarray[rawarray == 'married'] = 0
    rawarray[rawarray == 'divorced'] = 1
    rawarray[rawarray == 'single'] = 2
    set_forth = set((rawarray[:, 3]).flatten().tolist())
    rawarray[rawarray == 'professional.course'] = 0
    rawarray[rawarray == 'basic.4y'] = 1
    rawarray[rawarray == 'high.school'] = 2
    rawarray[rawarray == 'illiterate'] = 3
    rawarray[rawarray == 'basic.6y'] = 4
    rawarray[rawarray == 'university.degree'] = 5
    rawarray[rawarray == 'basic.9y'] = 6
    set_fifth = set((rawarray[:, 4]).flatten().tolist())
    rawarray[rawarray == 'yes'] = 1
    rawarray[rawarray == 'no'] = 0
    set_seventh = set((rawarray[:, 7]).flatten().tolist())
    rawarray[rawarray == 'telephone'] = 1
    rawarray[rawarray == 'cellular'] = 0
    set_eightth = set((rawarray[:, 8]).flatten().tolist())
    rawarray[rawarray == 'aug'] = 0
    rawarray[rawarray == 'mar'] = 1
    rawarray[rawarray == 'dec'] = 2
    rawarray[rawarray == 'may'] = 3
    rawarray[rawarray == 'sep'] = 4
    rawarray[rawarray == 'jul'] = 5
    rawarray[rawarray == 'apr'] = 6
    rawarray[rawarray == 'oct'] = 7
    rawarray[rawarray == 'jun'] = 8
    rawarray[rawarray == 'nov'] = 9
    set_nineth = set((rawarray[:, 9]).flatten().tolist())
    rawarray[rawarray == 'thu'] = 0
    rawarray[rawarray == 'wed'] = 1
    rawarray[rawarray == 'mon'] = 2
    rawarray[rawarray == 'tue'] = 3
    rawarray[rawarray == 'fri'] = 4
    set_14 = set((rawarray[:, 14]).flatten().tolist())
    rawarray[rawarray == 'nonexistent'] = 0
    rawarray[rawarray == 'success'] = 1
    rawarray[rawarray == 'failure'] = 2
    return rawarray



def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_data(data_x,data_y,args):
    set_random_seeds(0)
    true_idex = (torch.where(data_y == 0))[0]
    true_idex = true_idex[torch.randperm(true_idex.size(0))]
    false_idex = (torch.where(data_y == 1))[0]
    false_idex = false_idex[torch.randperm(false_idex.size(0))]
    train_data_x = torch.cat((data_x[true_idex[:args.num]],data_x[false_idex[:args.num]]), 0)
    train_data_y = torch.cat((data_y[true_idex[:args.num]],data_y[false_idex[:args.num]]), 0)
    test_data_x = torch.cat((data_x[true_idex[args.num:]],data_x[false_idex[args.num:]]), 0)
    test_data_y = torch.cat((data_y[true_idex[args.num:]],data_y[false_idex[args.num:]]), 0)
    return train_data_x,train_data_y,test_data_x,test_data_y







