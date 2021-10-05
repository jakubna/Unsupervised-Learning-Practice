from read_data import read_data
from algorithms.minmaxkmeans import MinMaxKmeans
from algorithms.kmeans import Kmeans
from algorithms.kpifs import Kpifs
from algorithms.kmedians import Kmedians
from evaluation.plot2d import plot2d
from evaluation.utils import *
import pandas as pd
import datetime

# define dataset name
dataset_name = 'iris'
# define number of the restarts of the algorithms
it_num = 50

# read and pre-process dataset
X, y, K = read_data(dataset_name)

label_list = [y]
df_results1 = pd.DataFrame(columns=['Algorithm', 'Esum', 'Esum_std', 'Emax', 'Emax_std', 'AMI', 'AMI_std', 'Time'])
df_results2 = pd.DataFrame(columns=['Algorithm', 'Compl', 'Homo', 'V-meas', 'DB', 'Silhouette'])

algorithm_list = ['MinMaxKmeans_b0', 'MinMaxKmeans_b01', 'MinMaxKmeans_b03', 'Kmeans', 'Kmeans++', 'Kmedians', 'Kpifs']


centroids = {}
plot_centroids = {}
for alg_name in algorithm_list:
    centroids[alg_name] = []
# run experiments with multi-restarts of algorithms and initializing centroids randomly (except kmeans++)
print('Experiments: methods')
for alg_name in algorithm_list:
    print('     {}'.format(alg_name))
    r1 = {}
    r2 = {}
    alg_results = dict(time=[], emax=[], esum=[], AMI=[], ARI=[], compl=[], homo=[], vmeas=[], db=[], silhouette=[])
    for i in range(0, it_num):
        try:
            # initialize algorithms with selected parameters
            T1 = datetime.datetime.now()
            if alg_name == 'MinMaxKmeans_b0':
                algorithm = MinMaxKmeans(M=K, beta=0)
            elif alg_name == 'MinMaxKmeans_b01':
                algorithm = MinMaxKmeans(M=K, beta=0.1)
            elif alg_name == 'MinMaxKmeans_b03':
                algorithm = MinMaxKmeans(M=K, beta=0.3)
            elif alg_name == 'Kmeans':
                algorithm = Kmeans(M=K)
            elif alg_name == 'Kmeans++':
                algorithm = Kmeans(M=K, init='kpp')
            elif alg_name == 'Kmedians':
                algorithm = Kmedians(M=K)
            elif alg_name == 'Kpifs':
                algorithm = Kpifs(M=K)
            algorithm.fit(X)
            y_pred = algorithm.labels_
            T = datetime.datetime.now() - T1
            alg_results['time'].append((T.total_seconds()))
            alg_results['esum'].append(algorithm.esum)
            alg_results['emax'].append(algorithm.emax)
            # compute supervised metrics
            sup_results = evaluate_supervised_external(y, y_pred)
            for k, v in sup_results.items():
                alg_results[k].append(v)
            # compute unsupervised metrics
            unsup_results = evaluate_unsupervised_internal(X, y_pred)
            for k, v in unsup_results.items():
                alg_results[k].append(v)
        except:
            pass
    # store average results
    r1['Algorithm'] = alg_name
    r1['Esum'] = np.round(np.mean(alg_results['esum']), 3)
    r1['Esum_std'] = np.round(np.std(alg_results['esum']), 3)
    r1['Emax'] = np.round(np.mean(alg_results['emax']), 3)
    r1['Emax_std'] = np.round(np.std(alg_results['emax']), 3)
    r1['AMI'] = np.round(np.mean(alg_results['AMI']), 3)
    r1['AMI_std'] = np.round(np.std(alg_results['AMI']), 3)
    r1['Time'] = np.round(np.mean(alg_results['time']), 3)
    r2['Algorithm'] = alg_name
    r2['Compl'] = np.round(np.mean(alg_results['compl']), 3)
    r2['Homo'] = np.round(np.mean(alg_results['homo']), 3)
    r2['V-meas'] = np.round(np.mean(alg_results['vmeas']), 3)
    r2['DB'] = np.round(np.mean(alg_results['db']), 3)
    r2['Silhouette'] = np.round(np.mean(alg_results['silhouette']), 3)
    df_results1 = df_results1.append(r1, ignore_index=True)
    df_results2 = df_results2.append(r2, ignore_index=True)
    label_list.append(algorithm.labels_)
    centroids[alg_name].append(algorithm.centroids)
    plot_centroids[alg_name] = algorithm.centroids
# plot results
plot2d(X, label_list, ['Original'] + algorithm_list, dataset_name, plot_centroids)

# run experiments with multi-restarts of algorithms and using previously indicated centroids for initialization
print('Experiments: methods + k-Means')
for alg_name in ['MinMaxKmeans_b0', 'MinMaxKmeans_b01', 'MinMaxKmeans_b03', 'Kmeans', 'Kmedians', 'Kpifs']:
    print('     {}'.format(alg_name))
    r1 = {}
    r2 = {}
    alg_results = dict(time=[], emax=[], esum=[], AMI=[], ARI=[], compl=[], homo=[], vmeas=[], db=[], silhouette=[])
    for derived_centroids in centroids[alg_name]:
        T1 = datetime.datetime.now()
        kmeans = Kmeans(M=K, centroids=derived_centroids)
        kmeans.fit(X)
        T = datetime.datetime.now() - T1
        alg_results['time'].append((T.total_seconds()))
        alg_results['esum'].append(kmeans.esum)
        alg_results['emax'].append(kmeans.emax)
        sup_results = evaluate_supervised_external(y, kmeans.labels_)
        for k, v in sup_results.items():
            alg_results[k].append(v)
        unsup_results = evaluate_unsupervised_internal(X, kmeans.labels_)
        for k, v in unsup_results.items():
            alg_results[k].append(v)
    r1['Algorithm'] = '{}+k-Means'.format(alg_name)
    r1['Esum'] = np.round(np.mean(alg_results['esum']), 3)
    r1['Esum_std'] = np.round(np.std(alg_results['esum']), 3)
    r1['Emax'] = np.round(np.mean(alg_results['emax']), 3)
    r1['Emax_std'] = np.round(np.std(alg_results['emax']), 3)
    r1['AMI'] = np.round(np.mean(alg_results['AMI']), 3)
    r1['AMI_std'] = np.round(np.std(alg_results['AMI']), 3)
    r1['Time'] = np.round(np.mean(alg_results['time']), 3)
    r2['Algorithm'] = '{}+k-Means'.format(alg_name)
    r2['Compl'] = np.round(np.mean(alg_results['compl']), 3)
    r2['Homo'] = np.round(np.mean(alg_results['homo']), 3)
    r2['V-meas'] = np.round(np.mean(alg_results['vmeas']), 3)
    r2['DB'] = np.round(np.mean(alg_results['db']), 3)
    r2['Silhouette'] = np.round(np.mean(alg_results['silhouette']), 3)
    df_results1 = df_results1.append(r1, ignore_index=True)
    df_results2 = df_results2.append(r2, ignore_index=True)

# save results
outputdir = 'results/'
outputfilename1 = outputdir + "{}_1.csv".format(dataset_name)
df_results1.to_csv(outputfilename1)
outputfilename2 = outputdir + "{}_2.csv".format(dataset_name)
df_results2.to_csv(outputfilename2)
