import sys
import pickle as pkl
import matplotlib.pyplot as plt

model = sys.argv[1]

pruning_percentage = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# accuracy vs pruning percentage
with open('model/'+model+'_conv.pkl','r') as f:
	data = pkl.load(f)
	accuracy_original = data['accuracies'][-1]
	loss_original = data['losses'][-1]

accuracies_prunning = []; losses_prunning = []
for i in pruning_percentage:
	with open('model/'+model+'_'+str(i)+'_retrained_conv.pkl','r') as f:
		data = pkl.load(f)
		accuracies_prunning.append(data['accuracies'][-1])
		losses_prunning.append(data['losses'][-1])
plt.subplot(2,1,1)
plt.plot(pruning_percentage, accuracies_prunning)
plt.plot((pruning_percentage[0], pruning_percentage[-1]), (accuracy_original, accuracy_original), 'k--')
plt.subplot(2,1,2)
plt.plot(pruning_percentage, losses_prunning)
plt.plot((pruning_percentage[0], pruning_percentage[-1]), (loss_original, loss_original), 'k--')
plt.show()





# plot weight distribution
# with open('model/'+model+'_retrained_conv.pkl','r') as f:
# 	data = pkl.load(f)

# w_original = data['w_original']
# w_pruned = data['w_pruned']
# w_retrained = data['w_retrained']
# pruned_inds_by_layer = data['pruned_inds_by_layer']
# losses = data['losses']
# accuracies = data['accuracies']


# plt.figure(0)
# plt.subplot(3,1,1)
# plt.hist(w_original[0].data.numpy().flatten(), bins=100)
# plt.subplot(3,1,2)
# plt.hist(w_original[1].data.numpy().flatten(), bins=100)
# plt.subplot(3,1,3)
# plt.hist(w_original[2].data.numpy().flatten(), bins=100)
# plt.figure(1)
# plt.subplot(3,1,1)
# plt.hist(w_pruned[0].data[1-pruned_inds_by_layer[0]].numpy(), bins=100)
# plt.subplot(3,1,2)
# plt.hist(w_pruned[1].data[1-pruned_inds_by_layer[1]].numpy(), bins=100)
# plt.subplot(3,1,3)
# plt.hist(w_pruned[2].data[1-pruned_inds_by_layer[2]].numpy(), bins=100)
# plt.figure(2)
# plt.subplot(3,1,1)
# plt.hist(w_retrained[0].data[1-pruned_inds_by_layer[0]].numpy(), bins=100)
# plt.subplot(3,1,2)
# plt.hist(w_retrained[1].data[1-pruned_inds_by_layer[1]].numpy(), bins=100)
# plt.subplot(3,1,3)
# plt.hist(w_retrained[2].data[1-pruned_inds_by_layer[2]].numpy(), bins=100)
# plt.show()