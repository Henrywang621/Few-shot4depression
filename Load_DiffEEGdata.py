import numpy as np
import mne
from mne import pick_types
from mne.io import read_raw_edf
from sklearn.preprocessing import scale
import scipy.io as sio
import numpy as np
import os 
import pandas as pd
from sklearn.preprocessing import scale, StandardScaler, OneHotEncoder
import random

random.seed(666)

def create_trial(start_index, run_length, trial_duration):
    '''
    input:
    start_index - the onset of the current run
    run_length - the number of samples in the run
    trial_duration - the constructed trial in seconds 
    
    output: a list of tuple ranges
    '''

    sampling_rate = 160
    trial_length = int(trial_duration * sampling_rate)
    pad = (trial_length - run_length) // 2
    reminder = (trial_length - run_length) % 2
    end_index = start_index + run_length
    windows = []
    windows += [round(start_index - pad - reminder), round(end_index + pad)]
    return windows

def getOneHotLabels(y):
    oh = OneHotEncoder()
    out = oh.fit_transform(y.reshape(-1, 1)).toarray()
    return out

def construct_X(event, data):
    '''Segment a run'''
    
    sampling_rate = 160
    trial_duration = 6
    start_pos = int(event[0] * sampling_rate)
    run_length = int(event[1] * sampling_rate)
    windows = create_trial(start_pos, run_length, trial_duration)
    # downsample each segement to 128Hz
    # x = data[:, windows[0]: windows[1]]
    x = mne.filter.resample(data[:, windows[0]: windows[1]], down=6.25, npad='auto')
    return x

def construct_Xr(event, data):
    # To generate the same data every time
    random_seeds = [42, 1, 1120, 150, 256, 303, 80, 5374, 646, 763, 4763, 
                    947, 1004, 7, 1157, 1234, 6402, 1314, 1337, 1448, 
                    662]
    trial_duration = 6
    sampling_rate = 160
    x = []
    y = []
    trial_length = int(trial_duration * sampling_rate)
    upper_limit = int(event[0][1] * sampling_rate) - trial_length
    for i in range(len(random_seeds)):
        offset = np.random.RandomState(seed=random_seeds[i]).randint(0, upper_limit)
#         x.append(data[:, offset:offset + trial_length])
        x.append(mne.filter.resample(data[:, offset:offset + trial_length], 
                                     down=1.25, npad='auto'))
        y += [0]
    return x, y

def read_edffile(subject_id, current_run):
    base_path = '/home/datasets/BCI2000/S{0:03}/S{1:03}R{2:02}.edf'
    path = base_path.format(subject_id, subject_id, current_run)
    raw = read_raw_edf(path, preload=True, verbose=False)
    onset = raw.annotations.onset
    duration = raw.annotations.duration
    description = raw.annotations.description
    # events[Oneset of the event in seconds, Duration of the event in seconds, Description]
    events = [[onset[i], duration[i], description[i]] for i in range(len(raw.annotations))]
    picks = pick_types(raw.info, eeg=True)
    raw.filter(l_freq=4, h_freq=None, picks=picks)
    data = raw.get_data(picks=picks)
    
    return events, data


'''
Experimental groups of Binary classification

Group1: Left hand & Right hand -> T1-run_type2 (label:0) & T2-run_type2 (label:1);
Group2: Both hands & Both Feet -> T1-run_type3 (label:0) & T2-run_type3 (label:1);
'''

def load_data(subject_ids):
    
    trials = []
    labels = []
    runs_type = [4, 8, 12]


    for subject_id in subject_ids:
        for MI_run in runs_type:
            events, data = read_edffile(subject_id, MI_run)
            for event in events[:-1]:
                if event[2] == 'T0':
                    continue
                else:
                    x = construct_X(event, data)
                    y = [0] if event[2] == 'T1' else [1]
                    trials.append(x)
                    labels += y

    return np.array(trials), np.array(labels).reshape(-1, 1)
    
def load_EEGdata4PhysioNet(subject_ids):
    X, y = load_data(subject_ids)
    X = np.hstack((X[:, 0:2, :], X))
    X = np.hstack((X, X[:, -2:, :]))

    # Z-score Normalization
    shape = X.shape
    for i in range(shape[0]):
        X[i,:, :] = scale(X[i,:, :]) 
        if (i+1)%int(shape[0]//10) == 0:
            print('{:.0%} done'.format((i+1)/shape[0]))
    
    return np.swapaxes(X, 1, 2), y

def standardizeData(x):
    # out = scale(x)
    # The difference between scale() and StandardScaler(): https://stackoverflow.com/questions/46257627/scikit-learn-preprocessing-scale-vs-preprocessing-standardscalar
    out = StandardScaler().fit_transform(x)
    return out

def read_EEG(filename):
    data = np.load(filename)
    processed_data = standardizeData(data)
    return processed_data

def loadEEGdata_OUND(subjects, control_groups = [1, 2], num_trails = 100, trial_length = 3000):
    '''
    subjects: the subjects you want to use to form the dataset. It should be a dictionary.
    control_groups: there are two separate control recordings at different dates.
    num_trails: 100 6s trails for each subject. Sampling rate is 1000Hz.
    '''
    data = []
    labels = []
    dir_path = '/home/data/ourDepression_small/'

    subjects_type = list(subjects.keys())
    # print(subjects_type)
    
    # generate data
    for sub_type in subjects_type:
        if sub_type == 'H':
            for subj in subjects['H']:
                for control_num in control_groups:
                    if control_num == 1 or 2:
                        # random.seed(666)
                        # sample_trials = random.sample(list(range(num_trails)), num_trails)
                        sample_trials = list(range(num_trails))
                        for i in sample_trials:
                            filename = '{0}{1}_{2}_{3}.npy'.format('H', subj, control_num, i)
                            file_path = os.path.join(dir_path, filename)
                            trail_data = read_EEG(file_path)
                            for i in range(0, trail_data.shape[0] - trial_length+1, trial_length):
                                data.append(trail_data[i:i+trial_length, :])
                                labels += [0]


        elif sub_type == 'F':
            for subj in subjects['F']:
                for control_num in control_groups:
                    if control_num == 1 or 2:
                        # random.seed(666)
                        # sample_trials = random.sample(list(range(num_trails)), num_trails)
                        sample_trials = list(range(num_trails))
                        for i in sample_trials:
                            filename = '{0}{1}_{2}_{3}.npy'.format('F', subj, control_num, i)
                            file_path = os.path.join(dir_path, filename)
                            trail_data = read_EEG(file_path)
                            for i in range(0, trail_data.shape[0] - trial_length+1, trial_length):
                                data.append(trail_data[i:i+trial_length, :])
                                labels += [1]
                    else:
                        print(control_num)
                        raise ValueError('control_num should be 1 or 2!!!')
                             
        else:
            raise ValueError('The subject type is either H or F!!!')

    return np.array(data), getOneHotLabels(np.array(labels)) 

def load_EEGdata4PREDICT(subjects_ids = list(range(122)), trial_length = 6000, num_trials = 100, test_mode = False):
    # load EEG data of the PREDICT, also called Depression Rest, dataset.
    EEGfile_path = '/home/data/Depression_mat/Matlab Files/'
    xlsxfile_path = '/home/data/Depression_mat/Data_4_Import_REST.xlsx'
    xlsxfile = pd.read_excel(xlsxfile_path)
    id_BDI = xlsxfile[['id', 'BDI']] 
    data = []
    labels = []
    if test_mode:
        sample_subjects = subjects_ids
    else:
        # sample_subjects = random.sample(subjects_ids, 80)
        sample_subjects = subjects_ids
    
    # subject id from 507 to 628
    for i in id_BDI['id'][sample_subjects]:
        count = 0
        file_name = '{0}_Depression_REST.mat'.format(i)
        path = os.path.join(EEGfile_path, file_name)
        if os.path.exists(path):
            file = sio.loadmat(path)

            # To make the number of channels be 68, which is same as our depression dataset
            if file['EEG']['data'][0][0].shape[0] == 66:
                file_exp = np.vstack((file['EEG']['data'][0][0][0], file['EEG']['data'][0][0]))
                file_exp = np.vstack((file_exp, file_exp[-1]))
                
            elif file['EEG']['data'][0][0].shape[0]  == 67:
                file_exp = np.vstack((file['EEG']['data'][0][0][0], file['EEG']['data'][0][0]))
            
            else:
                print(file.shape[0])
                raise ValueError("There exists other number of feature channels!!")
            
            file_exp = np.swapaxes(file_exp, 0, 1)    
            file_exp = standardizeData(file_exp)
            index = id_BDI.id[id_BDI.id == i].index.tolist()[0]

            if id_BDI.BDI[index] < 7:
                cur_label = 0
            else:
                cur_label = 1
            # print(file_exp.shape[0])
            for i in range(0, file_exp.shape[0] - trial_length, trial_length):
                if count == num_trials:
                    break
                else:
                    data.append(file_exp[i:i+trial_length, :])
                    labels.append(cur_label)
            del(file_exp)
            
        else:
            # print('The file '+'{0}_Depression_REST.mat'.format(i)+' does not exist!!!')
            continue
        
    return np.array(data), getOneHotLabels(np.array(labels))

#subjects_ids from 0 to 52
def load_EEGdata4MODMA(subjects_ids = list(range(53)), trial_length = 750, num_trials=100, test_mode = False):
    EEGfile_path = '/home/data/MODMA/'
    xlsxfile_path = '/home/data/MODMA/subjects_information_EEG.xlsx'
    xlsxfile = pd.read_excel(xlsxfile_path)
    all_ids_type = xlsxfile[['subject_id', 'type']]

    data = []
    labels = []
    if test_mode:
        samples_subjs = subjects_ids
    else:
        samples_subjs = subjects_ids
    # print(samples_subjs)
    for i in all_ids_type['subject_id'][samples_subjs]:
        count = 0
        name = '0' + str(i) + 'rest.mat'
        file_path = os.path.join(EEGfile_path, name)
        if os.path.exists(file_path):
            file = sio.loadmat(file_path)
            file = file[list(file.keys())[3]]
            file = np.swapaxes(file, 0, 1)
            file = standardizeData(file)
            index = all_ids_type.subject_id[all_ids_type.subject_id == i].index.tolist()[0]
            
            if all_ids_type.type[index] == 'HC':
                cur_label = 0
            else:
                cur_label = 1

            for i in range(0, file.shape[0] - trial_length, trial_length): 
                if count == num_trials:
                    break
                else:
                    data.append(file[i:i+trial_length, :])
                    labels.append(cur_label)
                    count = count + 1
            del(file)
        else:
            print('The file '+'0{0}rest.mat'.format(i)+' does not exist!!!')
            continue
    return np.array(data)[:,:,:68], getOneHotLabels(np.array(labels))

def minibatch_generator(num_samples_sup = 20, num_samples_que = 20, test_mode = True, subjects_OUND = None, subjects_Pred = None, subject_Physio = None,
                        subjects_Mod = None, control_group = None, num_trials =None, num_trials_MOD = None, num_trials_Pred = None, trial_length = None, datasets = None, random_seed = None):
    
    
    if test_mode:
        mini_batch_que = []
        mini_batch_sup = []
        labels_que = []
        labels_sup = [] 

        if datasets == 'OUND':
            data_X, labels_y = loadEEGdata_OUND(subjects_OUND, control_group, num_trials, trial_length)     

        elif datasets == 'PREDICT':
            data_X, labels_y = load_EEGdata4PREDICT(subjects_Pred, trial_length, num_trials = num_trials_Pred, test_mode=test_mode)

        elif datasets == 'PhysioNet':
            data_X, labels_y = load_EEGdata4PhysioNet(subject_Physio)

        elif datasets == 'MODMA':
            data_X, labels_y = load_EEGdata4MODMA(subjects_Mod, trial_length, num_trials = num_trials_MOD, test_mode=test_mode)
        
        else:
            raise ValueError("The dataset you choose is not included in our experiment!!! Please check your input!")

        random.seed(random_seed)
        sup_index = random.sample(list(range(len(data_X))), num_samples_sup)
        other_index = list(range(len(data_X)))
        for index in sorted(sup_index, reverse=True):
            del other_index[index]
            
        if num_samples_que > len(other_index) or num_samples_que > len(other_index) - num_samples_que:
            raise ValueError("Try to choose the propoer number of support samples!!!")

        for i in range(0, len(other_index)-num_samples_que, num_samples_que):
            mini_batch_sup.append([data_X[j] for j in sup_index])
            labels_sup.append([labels_y[j] for j in sup_index])
            mini_batch_que.append([data_X[j] for j in other_index[i:i+num_samples_que]])
            labels_que.append([labels_y[j] for j in other_index[i:i+num_samples_que]])

        labels_que = np.array(labels_que)
        # labels_que = np.expand_dims(labels_que, axis=-1)
        labels_sup = np.array(labels_sup)
        # labels_sup = np.expand_dims(labels_sup, axis=-1)

        yield ((np.array(mini_batch_que), np.array(mini_batch_sup), labels_sup), labels_que)
        
    else:
        while True:
            mini_batch_que = []
            mini_batch_sup = []
            labels_que = []
            labels_sup = [] 
            total_num_samples = num_samples_sup + num_samples_que
            for dataset in datasets:
                if dataset == 'OUND':
                    # control_group = random.sample(control_group, 1)
                    data_X, labels_y = datasets[dataset](subjects_OUND, control_group, num_trials, trial_length)
                    sample_ourids = random.sample(range(data_X.shape[0]), total_num_samples)

                    data_X = data_X[sample_ourids]
                    # print(labels_y[sample_ourids].shape)
                    labels_y = labels_y[sample_ourids]
                    mini_batch_que.append(data_X[:num_samples_que, :, :])
                    mini_batch_sup.append(data_X[num_samples_que:, :, :])
                    labels_sup.append(labels_y[num_samples_que:, :])
                    labels_que.append(labels_y[:num_samples_que, :])

                elif dataset == 'PREDICT':
                    data_X, labels_y = datasets[dataset](subjects_Pred, trial_length)
                    # print(labels_y.shape)
                    sample_dataIds = random.sample(range(data_X.shape[0]), total_num_samples)
                    
                    data_X = data_X[sample_dataIds]
                    labels_y = labels_y[sample_dataIds]
                    # print("Pred: ")
                    # print(labels_y.shape)
                    # print(labels_y)
                    if labels_y.shape[-1] == 1:
                        labels_y = np.expand_dims(labels_y, axis= -1)
                    
                    mini_batch_que.append(data_X[:num_samples_que, :, :])
                    mini_batch_sup.append(data_X[num_samples_que:, :, :])
                    labels_sup.append(labels_y[num_samples_que:, :])
                    labels_que.append(labels_y[:num_samples_que, :])
                
                elif dataset == 'PhysioNet':
                    subject_ids = list(range(1, 110))
                    # Damaged recordings (#88, #89, #92, #100 and #104) need to be removed.
                    remove_ids = [88, 89, 92, 100, 104]
                    for id in remove_ids:
                        subject_ids.remove(id)

                    sample_ids = random.sample(subject_Physio, 5)
                    data_X, labels_y = datasets[dataset](sample_ids)
                    sample_dataIds = random.sample(range(data_X.shape[0]), total_num_samples)
                    data_X = data_X[sample_dataIds]
                    labels_y = labels_y[sample_dataIds]
                    # labels_y = np.expand_dims(labels_y, axis= -1)
                    
                    mini_batch_que.append(data_X[:num_samples_que, :, :])
                    mini_batch_sup.append(data_X[num_samples_que:, :, :])
                    labels_sup.append(labels_y[num_samples_que:, :])
                    labels_que.append(labels_y[:num_samples_que, :])
                
                elif dataset == 'MODMA':
                    data_X, labels_y = datasets[dataset](subjects_Mod, trial_length)
                    # print("MOD: ")
                    # print(labels_y.shape)
                    # print(labels_y)
                    sample_dataIds = random.sample(range(data_X.shape[0]), total_num_samples)
                    
                    data_X = data_X[sample_dataIds]
                    labels_y = labels_y[sample_dataIds]
                    # print("MOD: ")
                    # print(labels_y.shape)
                    # labels_y = np.expand_dims(labels_y, axis= -1)
                    
                    mini_batch_que.append(data_X[:num_samples_que, :, :])
                    mini_batch_sup.append(data_X[num_samples_que:, :, :])
                    labels_sup.append(labels_y[num_samples_que:, :])
                    labels_que.append(labels_y[:num_samples_que, :])

                else:
                    raise ValueError("We do not include this dataset now or you should check whether you give the wrong dataset name!!!")
            
 
            yield ((np.array(mini_batch_que), np.array(mini_batch_sup), np.array(labels_sup)), np.array(labels_que)) 
