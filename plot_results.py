import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def plot1(layers, f_name, pruning_percentage):
	'''
	Accuracy vs pruning percentage
	'''
	fig, axes = plt.subplots(nrows=3, ncols=1)
	models = ['rnn', 'lstm', 'gru']
	for mid, model in enumerate(models):
		# loss & accuracy vs pruning percentage
		with open('model/'+model+f_name+'_conv.pkl','r') as f:
			data = pkl.load(f)
			accuracy_original = data['accuracies'][-1]
			loss_original = data['losses'][-1]

		accuracies_prunning = []; losses_prunning = []
		accuracies_pruned = []; losses_pruned = []
		for i in pruning_percentage:
			with open('model/'+model+'_'+str(i)+f_name+'_retrained_conv2.pkl','r') as f:
				data = pkl.load(f)
				accuracies_prunning.append(data['accuracies'][0])
				losses_prunning.append(data['losses'][0])
				accuracies_pruned.append(data['accuracies'][-1])
				losses_pruned.append(data['losses'][-1])


		plt.subplot(3,1,mid+1)
		plt.plot(pruning_percentage, accuracies_pruned, marker='o')
		plt.plot(pruning_percentage, accuracies_prunning, marker='x')
		plt.plot((pruning_percentage[0], pruning_percentage[-1]), (accuracy_original, accuracy_original), 'k--')
		plt.legend(['w/ retraining','w/o retrianing','no pruning'])
		plt.xlabel('percentage pruned (%)')
		plt.ylabel('test accuracy (%)')
		plt.title(model+' '+layers+' layers')
		plt.grid(True)
	fig.tight_layout()
	plt.show()


def plot2(layers, f_name, pruning_percentage):
	'''
	Class-blinded vs class distribution
	'''
	fig, axes = plt.subplots(nrows=3, ncols=1)
	models = ['rnn', 'lstm', 'gru']
	for mid, model in enumerate(models):
		# loss & accuracy vs pruning percentage
		with open('model/'+model+f_name+'_conv.pkl','r') as f:
			data = pkl.load(f)
			accuracy_original = data['accuracies'][-1]

		accuracies_pruned = []; accuracies_pruned2 = []
		for i in pruning_percentage:
			with open('model/'+model+'_'+str(i)+f_name+'_retrained_conv.pkl','r') as f:
				data = pkl.load(f)
				accuracies_pruned.append(data['accuracies'][-1])
			with open('model/'+model+'_'+str(i)+f_name+'_retrained_conv2.pkl','r') as f:
				data = pkl.load(f)
				accuracies_pruned2.append(data['accuracies'][-1])


		plt.subplot(3,1,mid+1)
		plt.plot(pruning_percentage, accuracies_pruned, marker='o')
		plt.plot(pruning_percentage, accuracies_pruned2, marker='x')
		plt.plot((pruning_percentage[0], pruning_percentage[-1]), (accuracy_original, accuracy_original), 'k--')
		plt.legend(['class distribution','class blinded','no pruning'])
		plt.xlabel('percentage pruned (%)')
		plt.ylabel('test accuracy (%)')
		plt.title(model+' '+layers+' layers')
		plt.grid(True)
	fig.tight_layout()
	plt.show()


def plot_conv(model, layers, f_name, pruning_percentage):
	# plot convergence
	with open('model/'+model+'_conv.pkl','r') as f:
		data = pkl.load(f)
		accuracies_original = data['accuracies']
		losses_original = data['losses']

	accuracies_prunning = []; losses_prunning = []
	with open('model/'+model+'_7_bin'+f_name+'_retrained_conv2.pkl','r') as f:
		data = pkl.load(f)
		accuracies_prunning = data['accuracies']
		losses_prunning = data['losses']

	plt.plot(np.linspace(0,51,205), accuracies_original+accuracies_prunning)
	plt.xlabel('epoch')
	plt.ylabel('test accuracy (%)')
	plt.title(model+' '+layers+' layers')
	plt.grid(True)
	plt.show()



def plot_dist(model, layers, f_name, pruning_percentage):
	# plot weight distribution
	with open('model/'+model+'_'+'90'+f_name+'_retrained_conv2.pkl','r') as f:
		data = pkl.load(f)

	w_original = data['w_original']
	w_pruned = data['w_pruned']
	w_retrained = data['w_retrained']
	pruned_inds_by_layer = data['pruned_inds_by_layer']
	losses = data['losses']
	accuracies = data['accuracies']

	plt.figure(0)
	plt.subplot(3,1,1)
	plt.hist(w_original[0].data.numpy().flatten(), bins=100)
	plt.title('input to hidden')
	# plt.axis([-1.5, 1.5, 0, 750])
	plt.subplot(3,1,2)
	plt.hist(w_pruned[0].data[1-pruned_inds_by_layer[0]].numpy(), bins=100)
	# plt.axis([-1.5, 1.5, 0, 750])
	plt.subplot(3,1,3)
	plt.hist(w_retrained[0].data[1-pruned_inds_by_layer[0]].numpy(), bins=100)
	# plt.axis([-1.5, 1.5, 0, 750])
	plt.xlabel('weight value')
	plt.figure(1)
	plt.subplot(3,1,1)
	plt.hist(w_original[1].data.numpy().flatten(), bins=100)
	plt.title('hidden to hidden')
	# plt.axis([-1., 1., 0, 3000])
	plt.subplot(3,1,2)
	plt.hist(w_pruned[1].data[1-pruned_inds_by_layer[1]].numpy(), bins=100)
	# plt.axis([-1., 1., 0, 3000])
	plt.subplot(3,1,3)
	plt.hist(w_retrained[1].data[1-pruned_inds_by_layer[1]].numpy(), bins=100)
	# plt.axis([-1., 1., 0, 3000])
	plt.xlabel('weight value')
	plt.figure(2)
	plt.subplot(3,1,1)
	plt.hist(w_original[2].data.numpy().flatten(), bins=100)
	plt.title('hidden to output')
	# plt.axis([-1.5, 1.6, 0, 40])
	plt.subplot(3,1,2)
	plt.hist(w_pruned[2].data[1-pruned_inds_by_layer[2]].numpy(), bins=100)
	# plt.axis([-1.5, 1.6, 0, 40])
	plt.subplot(3,1,3)
	plt.hist(w_retrained[2].data[1-pruned_inds_by_layer[2]].numpy(), bins=100)
	# plt.axis([-1.5, 1.6, 0, 40])
	plt.xlabel('weight value')
	plt.show()


def plot_dist2(model, layers, f_name, pruning_percentage):
	# plot weight distribution
	with open('model/'+model+'_7_bin'+f_name+'_retrained_conv2.pkl','r') as f:
		data = pkl.load(f)

	w_original = data['w_original']
	w_pruned = data['w_pruned']
	w_retrained = data['w_retrained']
	pruned_inds_by_layer = data['pruned_inds_by_layer']
	losses = data['losses']
	accuracies = data['accuracies']

	plt.figure(0)
	plt.subplot(3,1,1)
	plt.hist(w_original[0].data.numpy().flatten(), bins=100)
	plt.title('input to hidden')
	# plt.axis([-1.5, 1.5, 0, 750])
	plt.subplot(3,1,2)
	plt.hist(w_pruned[0].data.numpy().flatten(), bins=100)
	# plt.axis([-1.5, 1.5, 0, 750])
	plt.subplot(3,1,3)
	plt.hist(w_retrained[0].data.numpy().flatten(), bins=100)
	# plt.axis([-1.5, 1.5, 0, 750])
	plt.xlabel('weight value')
	plt.figure(1)
	plt.subplot(3,1,1)
	plt.hist(w_original[1].data.numpy().flatten(), bins=100)
	plt.title('hidden to hidden')
	# plt.axis([-1., 1., 0, 3000])
	plt.subplot(3,1,2)
	plt.hist(w_pruned[1].data.numpy().flatten(), bins=100)
	# plt.axis([-1., 1., 0, 3000])
	plt.subplot(3,1,3)
	plt.hist(w_retrained[1].data.numpy().flatten(), bins=100)
	# plt.axis([-1., 1., 0, 3000])
	plt.xlabel('weight value')
	plt.figure(2)
	plt.subplot(3,1,1)
	plt.hist(w_original[2].data.numpy().flatten(), bins=100)
	plt.title('hidden to output')
	# plt.axis([-1.5, 1.6, 0, 40])
	plt.subplot(3,1,2)
	plt.hist(w_pruned[2].data.numpy().flatten(), bins=100)
	# plt.axis([-1.5, 1.6, 0, 40])
	plt.subplot(3,1,3)
	plt.hist(w_retrained[2].data.numpy().flatten(), bins=100)
	# plt.axis([-1.5, 1.6, 0, 40])
	plt.xlabel('weight value')
	plt.show()



def perc_by_gate(model, layers, f_name, pruning_percentage):
	with open('model/'+model+'_'+pruning_percentage+f_name+'_retrained_conv2.pkl','r') as f:
		data = pkl.load(f)

	pruned_inds_by_layer = data['pruned_inds_by_layer']

	if model == 'rnn':
		for item in pruned_inds_by_layer:
			total = item.numpy().shape[0]*item.numpy().shape[1]
			print item.numpy().shape, total, item.numpy().sum()/float(total)
	elif model == 'lstm':
		print "input to hidden"
		for item in pruned_inds_by_layer[:-1:2]:
			w_i, w_f, w_c, w_o = item.chunk(4,0)
			total = item.numpy().shape[0]*item.numpy().shape[1]/4
			print 'input', w_i.numpy().shape, total, w_i.numpy().sum()/float(total)
			print 'forget', w_f.numpy().shape, total, w_f.numpy().sum()/float(total)
			print 'cell', w_c.numpy().shape, total, w_c.numpy().sum()/float(total)
			print 'output', w_o.numpy().shape, total, w_o.numpy().sum()/float(total)
		print "hidden to hidden"
		for item in pruned_inds_by_layer[1::2]:
			w_i, w_f, w_c, w_o = item.chunk(4,0)
			total = item.numpy().shape[0]*item.numpy().shape[1]/4
			print 'input', w_i.numpy().shape, total, w_i.numpy().sum()/float(total)
			print 'forget', w_f.numpy().shape, total, w_f.numpy().sum()/float(total)
			print 'cell', w_c.numpy().shape, total, w_c.numpy().sum()/float(total)
			print 'output', w_o.numpy().shape, total, w_o.numpy().sum()/float(total)
		print "hidden to output"
		total = pruned_inds_by_layer[-1].numpy().shape[0]*pruned_inds_by_layer[-1].numpy().shape[1]
		print pruned_inds_by_layer[-1].numpy().shape, total, pruned_inds_by_layer[-1].numpy().sum()/float(total)
	elif model == 'gru':
		print "intput to hidden"
		for item in pruned_inds_by_layer[:-1:2]:
			w_r, w_i, w_n = item.chunk(3,0)
			total = item.numpy().shape[0]*item.numpy().shape[1]/3
			print 'reset', w_r.numpy().shape, total, w_r.numpy().sum()/float(total)
			print 'input', w_i.numpy().shape, total, w_i.numpy().sum()/float(total)
			print 'new', w_n.numpy().shape, total, w_n.numpy().sum()/float(total)
		print "hidden to hidden"
		for item in pruned_inds_by_layer[1::2]:
			w_r, w_i, w_n = item.chunk(3,0)
			total = item.numpy().shape[0]*item.numpy().shape[1]/3
			print 'reset', w_r.numpy().shape, total, w_r.numpy().sum()/float(total)
			print 'input', w_i.numpy().shape, total, w_i.numpy().sum()/float(total)
			print 'new', w_n.numpy().shape, total, w_n.numpy().sum()/float(total)
		print "hidden to output"
		total = pruned_inds_by_layer[-1].numpy().shape[0]*pruned_inds_by_layer[-1].numpy().shape[1]
		print pruned_inds_by_layer[-1].numpy().shape, total, pruned_inds_by_layer[-1].numpy().sum()/float(total)




model = sys.argv[1]
layers = sys.argv[2]
prune_perc = sys.argv[3]

if layers == '1':
	f_name = ''
else:
	f_name = '_'+layers

pruning_percentage = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99]

# plot1(layers, f_name, pruning_percentage)
# plot2(layers, f_name, pruning_percentage)
plot_conv(model, layers, f_name, pruning_percentage)
# plot_dist2(model, layers, f_name, pruning_percentage)
# perc_by_gate(model, layers, f_name, prune_perc)





