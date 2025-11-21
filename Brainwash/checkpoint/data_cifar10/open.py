import pickle as pkl

path = "/home/jun/work/soongsil/Brainwash/data_cifar10/ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar10_class_num_2_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_4__seed_0_emb_fact_1_im_sz_32_.pkl"

data = pkl.load(open(path, "rb"))

print(data.keys())
