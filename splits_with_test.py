import os
from training_hyperparameters import data_directory

all_patients = os.listdir(data_directory)
all_patients = sorted(all_patients)

unit = len(all_patients)//10

s1_indexes = {
    'test' : [i for i in range(0,2*unit)],
    'validation' : [i for i in range(2*unit,3*unit)],
    'train' : [i for i in range(3*unit,len(all_patients))],
}
s2_indexes = {
    'test' : [i for i in range(2*unit, 4*unit)],
    'validation' : [i for i in range(4*unit,5*unit)],
    'train' : [*[i for i in range(0,2*unit)] ,*[i for i in range(5*unit,len(all_patients))]],
}
s3_indexes = {
    'test' : [i for i in range(4*unit, 6*unit)],
    'validation' : [i for i in range(6*unit,7*unit)],
    'train' : [*[i for i in range(0,4*unit)] ,*[i for i in range(7*unit,len(all_patients))]],
}
s4_indexes = {
    'test' : [i for i in range(6*unit, 8*unit)],
    'validation' : [i for i in range(8*unit,9*unit)],
    'train' : [*[i for i in range(0,6*unit)] ,*[i for i in range(9*unit,len(all_patients))]],
}
s5_indexes = {
    'test' : [i for i in range(8*unit,len(all_patients))],
    'validation' : [i for i in range(0*unit,1*unit)],
    'train' : [*[i for i in range(1*unit,8*unit)]],
}



SPLITS = {
    1 : {
        'train' : [os.path.join(data_directory,all_patients[i]) for i in s1_indexes['train']],
        'test' : [os.path.join(data_directory,all_patients[i]) for i in s1_indexes['test']],
        'validation' : [os.path.join(data_directory,all_patients[i]) for i in s1_indexes['validation']],
        },
    2 : {
        'train' : [os.path.join(data_directory,all_patients[i]) for i in s2_indexes['train']],
        'test' : [os.path.join(data_directory,all_patients[i]) for i in s2_indexes['test']],
        'validation' : [os.path.join(data_directory,all_patients[i]) for i in s2_indexes['validation']],
        },
    3 : {
        'train' : [os.path.join(data_directory,all_patients[i]) for i in s3_indexes['train']],
        'test' : [os.path.join(data_directory,all_patients[i]) for i in s3_indexes['test']],
        'validation' : [os.path.join(data_directory,all_patients[i]) for i in s3_indexes['validation']],
        },    
    4 : {
        'train' : [os.path.join(data_directory,all_patients[i]) for i in s4_indexes['train']],
        'test' : [os.path.join(data_directory,all_patients[i]) for i in s4_indexes['test']],
        'validation' : [os.path.join(data_directory,all_patients[i]) for i in s4_indexes['validation']],
        },    
    5 : {
        'train' : [os.path.join(data_directory,all_patients[i]) for i in s5_indexes['train']],
        'test' : [os.path.join(data_directory,all_patients[i]) for i in s5_indexes['test']],
        'validation' : [os.path.join(data_directory,all_patients[i]) for i in s5_indexes['validation']],
        },
 }



if __name__ == '__main__':
    timmy = []
    for test, values in SPLITS[2].items():
        print(test, values)
        print('\n', len(values),'\n')
        # if test == "validation":
        #     continue
        timmy = [*timmy, *values]
    timmy = sorted(timmy)
    print('\n\n\n\n', timmy, '\n', len(timmy),'\n')

