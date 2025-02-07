import glob
import numpy as np
import matplotlib.pyplot as plt
import yaml
from dataclasses import dataclass
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec

def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize=16
    else:
        fontsize=int(fontsize)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)

    #plt.rcParams.update({'figure.facecolor':'w'})
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

    return

set_rc_params(fontsize=16)

@dataclass
class Metric:
    accuracy: np.array
    ECE: np.array
    group_orders: np.array
    sigma: float

files = glob.glob('no_noise_no_psf/sdss/*/metrics/c*best_model*.yaml')

metrics = []
for i in range(6):
    sigma = 0.5 * i 
    sigma_str = str(sigma) 

    accuracy = []
    ECE = []
    group_orders = []

    sigma_files = [file for file in files if sigma_str in file]

    for file in sigma_files:
        filename = file.split('/')[-1]
        if 'cnn' in filename:
            group_order = 0
        elif 'c10' in filename:
            group_order = 10
        elif 'c12' in filename:
            group_order = 12
        elif 'd10' in filename:
            group_order = 10
        elif 'd12' in filename:
            group_order = 12
        else:
            group_order = int(filename[1])
        group_orders.append(group_order)
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
            accuracy.append(data['accuracy'])
            ECE.append(data['ECE'])

    accuracy = np.array(accuracy)
    ECE = np.array(ECE)
    group_orders = np.array(group_orders)
    metric = Metric(accuracy=accuracy, ECE=ECE, group_orders=group_orders, sigma=sigma)
    metrics.append(metric)

marker_dict = {'0.0': 'o', '0.5': '^', '1.0': 's', '1.5': 'D', '2.0': 'v', '2.5': 'p'}
colors = ['b', 'g', 'r', 'c', 'm', 'y']
linestyles = ['-', '--']

fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
for metric in metrics:
    sorted_indices = np.argsort(metric.group_orders)  # Sort indices based on group_orders
    sorted_group_orders = np.array(metric.group_orders)[sorted_indices]
    sorted_accuracy = np.array(metric.accuracy)[sorted_indices]

    axes[0].plot(sorted_group_orders, sorted_accuracy, 
                 label=r'Accuracy, $\sigma = $ ' + str(metric.sigma), 
                 marker=marker_dict[str(metric.sigma)], 
                 color=colors[int(metric.sigma * 2)], 
                 linestyle=linestyles[0])

axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy vs Group Order under PSF Convolution on SDSS Galaxies')
axes[0].legend()

for metric in metrics:
    sorted_indices = np.argsort(metric.group_orders)  # Sort indices based on group_orders
    sorted_group_orders = np.array(metric.group_orders)[sorted_indices]
    sorted_ECE = np.array(metric.ECE)[sorted_indices]

    axes[1].plot(sorted_group_orders, sorted_ECE,
                 label=r'ECE, $\sigma = $ ' + str(metric.sigma), 
                 marker=marker_dict[str(metric.sigma)], 
                 color=colors[int(metric.sigma * 2)], 
                 linestyle=linestyles[1])

axes[1].set_xlabel('Cyclic Group Order')
axes[1].set_ylabel('ECE')
axes[1].set_title('ECE vs Group Order under PSF Convolution on SDSS Galaxies')
axes[1].legend()

plt.tight_layout()
plt.savefig('sdss_accuracy_ece_vs_group_order_cyclic_groups.pdf')

files = glob.glob('no_noise_no_psf/sdss/*/metrics/d*best_model*.yaml')
files += glob.glob('no_noise_no_psf/sdss/*/metrics/cnn*best_model*.yaml')

metrics = []
for i in range(6):
    sigma = 0.5 * i 
    sigma_str = str(sigma) 

    accuracy = []
    ECE = []
    group_orders = []

    sigma_files = [file for file in files if sigma_str in file]

    for file in sigma_files:
        filename = file.split('/')[-1]
        if 'cnn' in filename:
            group_order = 0
        elif 'c10' in filename:
            group_order = 10
        elif 'c12' in filename:
            group_order = 12
        elif 'd10' in filename:
            group_order = 10
        elif 'd12' in filename:
            group_order = 12
        else:
            group_order = int(filename[1])
        group_orders.append(group_order)
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
            accuracy.append(data['accuracy'])
            ECE.append(data['ECE'])

    accuracy = np.array(accuracy)
    ECE = np.array(ECE)
    group_orders = np.array(group_orders)
    metric = Metric(accuracy=accuracy, ECE=ECE, group_orders=group_orders, sigma=sigma)
    metrics.append(metric)

marker_dict = {'0.0': 'o', '0.5': '^', '1.0': 's', '1.5': 'D', '2.0': 'v', '2.5': 'p'}

fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
for metric in metrics:
    sorted_indices = np.argsort(metric.group_orders)  # Sort indices based on group_orders
    sorted_group_orders = np.array(metric.group_orders)[sorted_indices]
    sorted_accuracy = np.array(metric.accuracy)[sorted_indices]

    axes[0].plot(sorted_group_orders, sorted_accuracy,
                 label=r'Accuracy, $\sigma = $ ' + str(metric.sigma), 
                 marker=marker_dict[str(metric.sigma)], 
                 color=colors[int(metric.sigma * 2)], 
                 linestyle=linestyles[0])

axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy vs Group Order under PSF Convolution on SDSS Galaxies')
axes[0].legend()

for metric in metrics:
    sorted_indices = np.argsort(metric.group_orders)  # Sort indices based on group_orders
    sorted_group_orders = np.array(metric.group_orders)[sorted_indices]
    sorted_ECE = np.array(metric.ECE)[sorted_indices]

    axes[1].plot(sorted_group_orders, sorted_ECE,
                 label=r'ECE, $\sigma = $' + str(metric.sigma), 
                 marker=marker_dict[str(metric.sigma)], 
                 color=colors[int(metric.sigma * 2)], 
                 linestyle=linestyles[1])

axes[1].set_xlabel('Dihedral Group Order')
axes[1].set_ylabel('ECE')
axes[1].set_title('ECE vs Group Order under PSF Convolution on SDSS Galaxies')
axes[1].legend()

plt.tight_layout()
plt.savefig('sdss_accuracy_ece_vs_group_order_dihedral_groups.pdf')
