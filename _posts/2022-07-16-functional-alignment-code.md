---
layout: post
title: Functional Alignment Notebook
date: 2022-07-16 20:09:00
description: tutorial for functional alignment
tags: neuroimaging psychology analysis-methods
categories: naturalistic-data-analysis
---

# Functional Alignment

```python
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from nltools.mask import create_sphere, expand_mask
from nltools.data import Brain_Data, Adjacency
from nltools.stats import align
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn.plotting import plot_stat_map
import datalad.api as dl
import warnings

warnings.simplefilter('ignore')
```

    It is highly recommended to configure Git before using DataLad. Set both 'user.name' and 'user.email' configuration variables.

## Get Dataset using datalad

```python
data_dir = '/Users/naeun-oh/Sherlock'

# If dataset hasn't been installed, clone from GIN repository
if not os.path.exists(data_dir):
    dl.clone(source='https://gin.g-node.org/ljchang/Sherlock', path=data_dir)

# Initialize dataset
ds = dl.Dataset(data_dir)

ds.status(annex='all')
```

    [1;31muntracked[0m: fmriprep/sub-02/.DS_Store ([1;35mfile[0m)
    1350 annex'd files (25.1 GB/109.0 GB present/total size)


```python
# Get Cropped & Denoised HDF5 Files  : takes a very very long time
result = ds.get(glob.glob(os.path.join(data_dir, 'fmriprep', '*', 'func', f'*crop*hdf5')))
```

    [1;1mget[0m([1;32mok[0m): fmriprep/sub-01/func/sub-01_denoise_crop_smooth6mm_task-sherlockPart2_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-05/func/sub-05_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-06/func/sub-06_denoise_crop_smooth6mm_task-sherlockPart2_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-11/func/sub-11_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-12/func/sub-12_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-08/func/sub-08_denoise_crop_smooth6mm_task-sherlockPart2_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-02/func/sub-02_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-05/func/sub-05_denoise_crop_smooth6mm_task-sherlockPart2_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-04/func/sub-04_denoise_crop_smooth6mm_task-sherlockPart2_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-12/func/sub-12_denoise_crop_smooth6mm_task-sherlockPart2_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-06/func/sub-06_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-09/func/sub-09_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-04/func/sub-04_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    [1;1mget[0m([1;32mok[0m): fmriprep/sub-03/func/sub-03_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5 ([1;35mfile[0m) [from origin...]
    action summary:
      get (notneeded: 18, ok: 14)


Hyperalignment was developed at Dartmouth College and is implemented in the PyMVPA toolbox. There is a tutorial on the PyMVPA website for how to implement different versions of hyperalignment. The Shared Response Model was developed at Princeton University and is implemented in the brainiak toolbox and I also encourage you to see their excellent tutorial.

## Reponse-based Hyperalignment
Functional alignment is usually performed within an ROI. The original hyperalignment papers align within searchlights over the whole brain. 
Here, we will align within regions of interest (ROI) from whole-brain functional parcellations. We will use a n=50 parcellation based on patterns of coactivation from the Neurosynth database (de la Vega et al.,2016)


```python
# ROI mask
mask = Brain_Data('https://neurovault.org/media/images/8423/k50_2mm.nii.gz')
mask_x = expand_mask(mask)
mask.plot()
```

![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_6_0.png){: width="100%"}

As an example, let's extract voxel activity within the early visual cortex (i.e. ROI 4) from the second half of Sherlock (i.e. Part2) using hdf5 files.
Brain_Data class of nltools loads data.

```python
# Example of single ROI
scan = 'Part2'
roi = 4

roi_mask = mask_x[roi]

file_list = glob.glob(os.path.join(data_dir, 'fmriprep', '*', 'func', f'*crop*{scan}*hdf5'))
all_data = []
for f in file_list:
    sub = os.path.basename(f).split('_')[0]
    print(sub)
    data = Brain_Data(f)
    all_data.append(data.apply_mask(roi_mask))
    
roi_mask.plot()
```

    sub-13
    sub-14
    sub-15
    sub-12
    sub-08
    sub-01
    sub-06
    sub-07
    sub-09
    sub-10
    sub-11
    sub-16
    sub-05
    sub-02
    sub-03
    sub-04

![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_8_1.png){: width="100%"}

```python
# Hyperalignment using procrustes transform
# align() input is a list of Brain_Data objects (numpy matrices)

# Here, we will exclude the last subject for now and add them later in the tutorial.

# align() output is dictionary with keys for a list of...
# 'transformed': transformed data, 
# 'transformation_matrix': corresponding transformation matrices, 
# 'common_model': the common model where all subjects are projected
# 'isc': Intersubject Correlations(ISC) for transformed data
# 'disparity': multivariate distance of the subject to common space
# 'scale' ?

hyperalign = align(all_data[:15], method='procrustes')

print(hyperalign.keys())
```

Let's plot the aligned voxel time course.
We will see how similar the activity is across participants within a random voxel using ISC.
ISC is the average pairwise correlation between subject voxel time courses.

```python
voxel_index = 50

voxel_unaligned = pd.DataFrame([x.data[:, voxel_index] for x in all_data]).T
voxel_aligned = pd.DataFrame([x.data[:, voxel_index] for x in hyperalign['transformed']]).T

f, a = plt.subplots(nrows=2, figsize=(15, 5), sharex=True)
a[0].plot(voxel_unaligned, linestyle='-', alpha=.2)
a[0].plot(np.mean(voxel_unaligned, axis=1), linestyle='-', color='navy')
a[0].set_ylabel('Unaligned Voxel', fontsize=16)
a[0].yaxis.set_ticks([])

a[1].plot(voxel_aligned, linestyle='-', alpha=.2)
a[1].plot(np.mean(voxel_aligned, axis=1), linestyle='-', color='navy')
a[1].set_ylabel('Aligned Voxel', fontsize=16)
a[1].yaxis.set_ticks([])

plt.xlabel('Voxel Time Course (TRs)', fontsize=16)
a[0].set_title(f"Unaligned Voxel ISC: r={Adjacency(voxel_unaligned.corr(), matrix_type='similarity').mean():.02}", fontsize=18)
a[1].set_title(f"Aligned Voxel ISC: r={Adjacency(voxel_aligned.corr(), matrix_type='similarity').mean():.02}", fontsize=18)
```




    Text(0.5, 1.0, 'Aligned Voxel ISC: r=0.4')



![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_11_1.png){: width="100%" }

The overall time course of both unaligned and aligned voxel activity is very similar.
However, participants have an overall higher degree of similarity after hyperalignment (r=0.4) compared to the unaligned data (r=0.34).

```python
# Plot the distribution of overall ISC across all voxels
plt.hist(hyperalign['isc'].values())
plt.axvline(x=np.mean(list(hyperalign['isc'].values())), linestyle='--', color='red', linewidth=2)
plt.ylabel('Frequency', fontsize=16)
plt.xlabel('Voxel ISC Values', fontsize=16)
plt.title('Hyperalignment ISC', fontsize=18)

print(f"Mean ISC: {np.mean(list(hyperalign['isc'].values())):.2}")
```

    Mean ISC: 0.36

![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_13_1.png){: width="100%" }

The overall ISC across voxels is pretty high, mean=0.36.
Note that the mean ISC value is biased b/c it's not cross-validated so it's likely slightly inflated.

Let's plot the transformed data for a random TR and check the impact of hyperalignment on spatial topography.

```python
tr_index = 100

f,a = plt.subplots(ncols=5, nrows=2, figsize=(15,6), sharex=True, sharey=True)
for i in range(5):
    sns.heatmap(np.rot90(all_data[i][tr_index].to_nifti().dataobj[30:60, 10:28, 37]), cbar=False, cmap='RdBu_r', ax=a[0,i])
    a[0,i].set_title(f'Subject: {i+1}', fontsize=18)
    a[0,i].axes.get_xaxis().set_visible(False)
    a[0,i].yaxis.set_ticks([])
    sns.heatmap(np.rot90(hyperalign['transformed'][i][tr_index].to_nifti().dataobj[30:60, 10:28, 37]), cbar=False, cmap='RdBu_r', ax=a[1,i])
    a[1,i].axes.get_xaxis().set_visible(False)
    a[1,i].yaxis.set_ticks([])

a[0,0].set_ylabel('Unaligned Voxels', fontsize=16)
a[1,0].set_ylabel('Aligned Voxels', fontsize=16)

plt.tight_layout()
```

![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_16_0.png){: width="100%" }

Are the subjects' patterns more spatially similar after alignment? With alignment, voxels are rearranged to maximize temporal synchrony. The algorithm picks a random subject and then projects every other subject into that space. This is averaged and then iteratively repeated.

## Shared Response Model
This model allows alignment into a lower dimensional functional space, rather than dimension size of the number of voxels (which can be alot).
This model learns a common latent space and the overall dimensionality is limited to the number of observations.
In this example, there are more voxels (n=2786) in the early visual anatomical mask relative to the number of observed TRs (n=1030). This means the max number of components we can estimate is 1030.
Here, we will align to a 100 dimensional feature space.

```python
# zscore the data and then train model with SRM
all_data_z = [x.standardize(method='zscore') for x in all_data]
srm = align(all_data_z, method='deterministic_srm', n_features=100)
```

```python
# Plot the average time course of a single latent component
# But we cannot directly compare SRMs to the unaligned voxels (unlike the examples above)

component_index = 0

component_aligned = pd.DataFrame([x[:, component_index] for x in srm['transformed']]).T

f, a = plt.subplots(nrows=1, figsize=(15, 5), sharex=True)
a.plot(component_aligned, linestyle='-', alpha=.2)
a.plot(np.mean(component_aligned, axis=1), linestyle='-', color='navy')
a.set_ylabel('Aligned Component', fontsize=16)
a.yaxis.set_ticks([])

plt.xlabel('Component Time Course (TRs)', fontsize=16)
a.set_title(f"Aligned Component ISC: r={Adjacency(component_aligned.corr(), matrix_type='similarity').mean():.02}", fontsize=18)
```




    Text(0.5, 1.0, 'Aligned Component ISC: r=0.45')



![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_20_1.png){: width="100%" }

```python
# Plot the distribution of overall ISC across all components
plt.hist(srm['isc'].values())
plt.axvline(x=np.mean(list(srm['isc'].values())), linestyle='--', color='red', linewidth=2)
plt.ylabel('Frequency', fontsize=16)
plt.xlabel('Voxel ISC Values', fontsize=16)
plt.title('Shared Response Model ISC', fontsize=18)
```




    Text(0.5, 1.0, 'Shared Response Model ISC')



![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_21_1.png){: width="100%" }

The consequence of this lower dimensional projection is that we can no longer maintain a voxel level representation. So we are unable to generate the same figure depicting how the cortical topographies change.
Instead, we will plot the weights that project each subject's data into a common latent time course.

```python
# Plot spatial pattern of weights that project subject data into a random latent component (time course)
component = 3

f = plt.figure(constrained_layout=True, figsize=(12,8))
spec = gridspec.GridSpec(ncols=4, nrows=4, figure=f)
for i in range(4):
    a0 = f.add_subplot(spec[i, 0])
    a0.imshow(np.rot90(srm['transformation_matrix'][i][component].to_nifti().dataobj[30:60, 10:28, 37]),cmap='RdBu_r')
    a0.set_ylabel(f'Subject {i+1}', fontsize=18)
    a0.yaxis.set_ticks([])
    a0.xaxis.set_visible(False)    
    
    a1 = f.add_subplot(spec[i, 1:])
    a1.plot(srm['transformed'][i][:,component])
    a1.xaxis.set_visible(False)
    a1.yaxis.set_visible(False)

    if i < 1:
        a0.set_title('Spatial Pattern', fontsize=20)
        a1.set_title('Latent Timecourse', fontsize=20)
```

![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_23_0.png){: width="100%" }

## Project New Subject Data into Common Space
We can align new subjects into the common model without retraining the entire model.
Here, we individually align subject 16 to the common space learned above using hyperalignment.

```python
# Align the leftout subject data to common model
new_data = all_data[-1]

new_data[0].plot()

aligned_sub_hyperalignment = new_data.align(hyperalign['common_model'], method='procrustes')

aligned_sub_hyperalignment['transformed'][0].plot()
```

![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_25_0.png){: width="100%" }

![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_25_1.png){: width="100%" }

The pattern of cortical activation has now changed after projecting this subject into the common space.

We can also project subject 16 into the SRM common space. Note that we also need to zscore the subject's data before alignment.

```python
aligned_sub_srm = new_data.standardize(method='zscore').align(srm['common_model'], method='deterministic_srm')

aligned_sub_srm['transformation_matrix'][0].plot()
```

![png]({{ site.baseurl }}/assets/img/Functional%20Alignment/Functional%20Alignment_27_0.png){: width="100%" }

Because the FA models were trained to maximize ISC, the ISC values are biased and will likely be inflated.
It is important to evaluate how well the model works on independent data.
You can either divide data into training and test datasets, or perform cross-validation.

The idea is to use the LEARNED subject-specific transformation matrices to project NEW independent data from that participant into the common space.

Here we will project Part1 Sherlock data into common space using models trained on Part2 data.

```python
# Load Part1 data and overwrite Part2 data to save RAM
scan = 'Part1'
roi = 4

roi_mask = mask_x[roi]

file_list = glob.glob(os.path.join(data_dir, 'fmriprep', '*', 'func', f'*crop*{scan}*hdf5'))
all_data = []
for f in file_list:
    sub = os.path.basename(f).split('_')[0]
    print(sub)
    data = Brain_Data(f)
    all_data.append(data.apply_mask(roi_mask))
```

    sub-13
    sub-14
    sub-15
    sub-12
    sub-08
    sub-01
    sub-06
    sub-07
    sub-09
    sub-10
    sub-11
    sub-16
    sub-05
    sub-02
    sub-03
    sub-04

Project data into the common space using hyperalignment.

```python
# Create a copy of subject 16's data variable
s16_pt1_hyp_transformed = all_data[-1].copy()

# Use np.dot() to perform a simple inner matrix multiplication to project the data into common space 
# using the subject's transformation matrix learned from Part2
s16_pt1_hyp_transformed.data = np.dot(s16_pt1_hyp_transformed.data, aligned_sub_hyperalignment['transformation_matrix'].data.T)

s16_pt1_hyp_transformed
```




    nltools.data.brain_data.Brain_Data(data=(946, 2786), Y=0, X=(0, 0), mask=MNI152_T1_2mm_brain_mask.nii.gz)



```python
# Repeat for the rest of the participants

hyperalign_transformed = []
for i,x in enumerate(all_data[:15]):
    new_x = x.copy()
    new_x.data = np.dot(x.data, hyperalign['transformation_matrix'][i].data.T)
    hyperalign_transformed.append(new_x)
```

This time, project data into the common space using SRM. Remember to zscore the data again.

```python
s16_pt1_srm_transformed = all_data[-1].copy().standardize(method='zscore')

s16_pt1_srm_transformed = np.dot(s16_pt1_srm_transformed.data, aligned_sub_srm['transformation_matrix'].data.T)

s16_pt1_srm_transformed.shape
```




    (946, 100)



```python
# Repeat for the rest of the participants

srm_transformed = []
for i,x in enumerate(all_data[:15]):
    srm_transformed.append(np.dot(x.standardize(method='zscore').data, srm['transformation_matrix'][i].data.T))
```