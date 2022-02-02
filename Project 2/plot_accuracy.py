import matplotlib.pyplot as plt
import numpy as np

lines = np.loadtxt("logs/num_epoch_20_batch_size_2_num_hidden_200_lr_0.2_num_nodes_10test_accu.txt", comments="#", delimiter=" ", unpack=False)
lines1 = np.loadtxt("logs/num_epoch_20_batch_size_4_num_hidden_200_lr_0.2_num_nodes_5test_accu.txt", comments="#", delimiter=" ", unpack=False)
lines2 = np.loadtxt("logs/num_epoch_20_batch_size_10_num_hidden_200_lr_0.2_num_nodes_2test_accu.txt", comments="#", delimiter=" ", unpack=False)
lines3 = np.loadtxt("logs/num_epoch_20_batch_size_20_num_hidden_200_lr_0.2_num_nodes_1test_accu.txt", comments="#", delimiter=" ", unpack=False)

train = np.loadtxt("logs/num_epoch_20_batch_size_2_num_hidden_200_lr_0.2_num_nodes_10global_train_accu.txt", comments="#", delimiter=" ", unpack=False)
train1 = np.loadtxt("logs/num_epoch_20_batch_size_4_num_hidden_200_lr_0.2_num_nodes_5global_train_accu.txt", comments="#", delimiter=" ", unpack=False)
train2 = np.loadtxt("logs/num_epoch_20_batch_size_10_num_hidden_200_lr_0.2_num_nodes_2global_train_accu.txt", comments="#", delimiter=" ", unpack=False)
train3 = np.loadtxt("logs/num_epoch_20_batch_size_20_num_hidden_200_lr_0.2_num_nodes_1global_train_accu.txt", comments="#", delimiter=" ", unpack=False)

train_acc = [np.mean(train[-3]), np.mean(train1[-1]), np.mean(train2[-1]), np.mean(train3[-1])]
test_acc = [np.mean(lines[-3]), np.mean(lines1[-1]), np.mean(lines2[-1]), np.mean(lines3[-1])]
batch_sizes = [2, 4, 10, 20]
plt.plot(batch_sizes, train_acc, marker='o', markerSize=5, label='train')
plt.plot(batch_sizes, test_acc, marker='o', markerSize=5, label='test')
plt.legend()
plt.xticks(batch_sizes)
plt.xlabel('Batch size')
plt.ylabel('Acuuracy')
plt.savefig('accuracy.pdf', format='pdf', dpi=None, bbox_inches='tight')
