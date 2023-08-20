from args1 import argument_parser
import tensorflow as tf
from EEGHetNet import EEGHetNet
import ast 
# from hetnet import HetNet
from Load_DiffEEGdata import *
import gc





gc.enable()
args = argument_parser()

print("########## argument sheet ########################################")
for arg in vars(args):
    print (f"#{arg:>15}  :  {str(getattr(args, arg))} ")
print("##################################################################")

loss_object = loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer     = tf.keras.optimizers.Adam(learning_rate=args.lr,)

#--------Define The Model----------------

model_type = "gru"
print("Building Network ...")
if args.hetmodel.lower() == "time":
    EEGFS_net    = EEGHetNet(dims_inf = ast.literal_eval(args.dims),
                            dims_pred = ast.literal_eval(args.dims_pred), 
                            activation="relu", 
                            time=args.tmax_length,
                            batchnorm=args.batchnorm, 
                            block = args.block.split(","))
    
    print("Using EEGFSNet")

else:

        EncM = SEncModel(control=args.control_steps)
        EEGFS_net = HetNet(EncM, "slice",
                               dims = ast.literal_eval(args.dims),
                               acti="relu",
                               drop1=0.01,
                               drop2=0.01,
                               share_qs=False)
        print("Using Hetnet")
    

EEGFS_net.compile(loss=loss_object,optimizer=optimizer,metrics=['accuracy'])

#--------Load the data----------------

# subjects_Mod = list(range(53))
# subjects_Pred = list(range(122))
# subjects_OUND = {'F': [1, 10, 11], 'H': [11, 12, 13]}

# subjects_OUND = {'F': [1, 10], 'H': [11, 12]}

subjects_OUND = {'F': [1,10,11], 'H': [11,12,13]}
# subjects_OUND = {'F': [1], 'H': [11]}
# subjects_OUND = {'F': [1,10], 'H': [11,12]}

subjects_val = {'F': [12], 'H': [14]}

# datasets_train = {'MODMA': load_EEGdata4MODMA,
#                   'PREDICT': load_EEGdata4PREDICT,
#                   'OUND': loadEEGdata_OUND,
#                  }

# datasets_train = {'PREDICT': load_EEGdata4PREDICT,
#                   'OUND': loadEEGdata_OUND,
#                  }

datasets_train = {'OUND': loadEEGdata_OUND,
                 }
# datasets_train = {'PREDICT': load_EEGdata4PREDICT,
#             }


datasets_val = {'OUND': loadEEGdata_OUND,
               }

# control_group = [1, 2]
# num_trials = 100

test_mode = True
# train_gen =  minibatch_generator(args.s_shots,args.q_shots, False, subjects_OUND = subjects_OUND, subjects_Pred = subjects_Pred, subjects_Mod = subjects_Mod, control_group = [1, 2], num_trials= 50, trial_length = 3000, datasets = datasets_train)
# train_gen =  minibatch_generator(args.s_shots,args.q_shots, False, subjects_OUND = subjects_OUND, subjects_Pred = subjects_Pred, control_group = [1, 2], num_trials= 100, trial_length = 3000, datasets = datasets_train)
# train_gen =  minibatch_generator(args.s_shots,args.q_shots, False, subjects_OUND = subjects_OUND, control_group = [1, 2], num_trials= 5, trial_length = 3000, datasets = datasets_train)
train_gen =  minibatch_generator(args.s_shots,args.q_shots, False, subjects_OUND = subjects_OUND, control_group = [1, 2], num_trials= 100, trial_length = 3000, datasets = datasets_train)
val_gen =  minibatch_generator(args.s_shots,args.q_shots, False, subjects_OUND = subjects_val, control_group = [1, 2], num_trials= 100, trial_length = 3000, datasets = datasets_val)


value = 0
for i in val_gen:
     value = i
     break

# print(value[0][2].shape)
# print(value[0][2])

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=args.best_weights)  
callbacks = None
if args.early_stopping:
    callbacks = [earlystopping]

history = EEGFS_net.fit(              x  = train_gen,
                      validation_data  = val_gen,
                      validation_steps = 50,
                      epochs           = args.num_epochs,
                      steps_per_epoch  = 20,
                      callbacks        = callbacks)

# EEGFS_net.save('saved_EEGHetNetOUNDplustwoD2',save_format='tf')
# EEGFS_net.save('saved_EEGHetNetOUND6SplusPred',save_format='tf')
EEGFS_net.save('saved_EEGHetNetOUND6S',save_format='tf')

del(train_gen)
del(val_gen)
gc.collect()
