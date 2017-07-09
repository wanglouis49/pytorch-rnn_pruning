import pickle as pkl
import matplotlib.pyplot as plt

model = 'lstm'

with open('model/'+model+'_retrained_conv.pkl','r') as f:
	data = pkl.load(f)

w_original = data['w_original']
w_pruned = data['w_pruned']
w_retrained = data['w_retrained']
pruned_inds_by_layer = data['pruned_inds_by_layer']
losses = data['losses']
accuracies = data['accuracies']


plt.figure(0)
plt.subplot(3,1,1)
plt.hist(w_original[0].data.numpy().reshape(384*28,1), bins=100)
plt.subplot(3,1,2)
plt.hist(w_original[1].data.numpy().reshape(384*128,1), bins=100)
plt.subplot(3,1,3)
plt.hist(w_original[2].data.numpy().reshape(10*128,1), bins=100)
plt.figure(1)
plt.subplot(3,1,1)
plt.hist(w_pruned[0].data[1-pruned_inds_by_layer[0]].numpy(), bins=100)
plt.subplot(3,1,2)
plt.hist(w_pruned[1].data[1-pruned_inds_by_layer[1]].numpy(), bins=100)
plt.subplot(3,1,3)
plt.hist(w_pruned[2].data[1-pruned_inds_by_layer[4]].numpy(), bins=100)
plt.figure(2)
plt.subplot(3,1,1)
plt.hist(w_retrained[0].data[1-pruned_inds_by_layer[0]].numpy(), bins=100)
plt.subplot(3,1,2)
plt.hist(w_retrained[1].data[1-pruned_inds_by_layer[1]].numpy(), bins=100)
plt.subplot(3,1,3)
plt.hist(w_retrained[2].data[1-pruned_inds_by_layer[4]].numpy(), bins=100)
plt.show()