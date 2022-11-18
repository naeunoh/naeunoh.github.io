---
layout: post
title: Functional Alignment
date: 2022-07-16 15:09:00
description: minimize individual variation
tags: neuroimaging psychology analysis-methods
categories: naturalistic-data-analysis
---


Generally, neuroimaging studies implicitly assume that each person’s brain processes information in the same way as other people’s brains. However, it is a known fact that there is a wide variation in individual neuroanatomy. 

Thus, to identify the common brain processes, we must normalize each participant into a common stereotactic space. This would minimize the individual variation that we are not interested in.

One way is applying a small amount of Gaussian smoothing to mitigate small misalignments and increase voxel signal to noise ratios (SNR). This method is effective when performing mass univariate testing but problematic when using multivariate techniques that assume strong alignment of features (e.g. voxels) across participants.

Another newly developing approach lead by Jim Haxby and Peter Ramadge projects subjects into common space based on how voxels respond to stimuli or are connected to other voxels. This is called **functional alignment** or **hyperalignment**.

The basic idea behind hyperalignment is as follows.

First, the cortical patterns are treated as vectors corresponding to locations in a high dimensional space, where each axis reflects a measurement of that pattern (e.g. voxel activity). Rather than treating cortical functions in a 1D (average roi activity), 2D (cortical sheet), or 3D physical space, HA models information as being embedded in an n-D space, where n reflects the number of measurements (e.g. voxels). Simply put, a cortical pattern of a timepoint t (e.g. conditions, stimuli, timepoints) is vectorized or flattened into a vector and then expressed as a location point/position in the n-dimensional space where each axis is a voxel. 
Then, vector representations of individual cortical patterns can be transformed into a common dimensional space that is shared across participants (Haxby et al., 2000). Each cortical pattern has a unique transformation matrix (voxel x dimension). 

![hyperalignment]({{ site.baseurl }}/assets/img/fa/elife-56601-fig1-v1.jpg){: width="100%" }

**Note:** [3blue1brown video on vectors](https://youtu.be/fNk_zzaMoSs) might help understand the general concept of vectors in space.


There are many different models of functional alignment. The basic framework can be applied to cortical patterns of voxel responses (*response-based hyperalignment*, vector=timepoint x voxel) or functional connectivity (*connectivity-based hyperalignment*, vector=target x connected node) or latent feature space (*shared response model*, joint-SVD).
Overall, the lower dimensional projections (onto common dimensional space) remove redundant axes in the high dimensional space and can effectively denoise the signal, which also improves classification accuracy of subsequent analysis. It has been shown that all the different approaches can dramatically improve between subject classification accuracy in ventral temporal cortex.

(For detailed information, watch the [video by Dr. James Haxby, PhD](https://youtu.be/QX7sNaLyxdo) from the 2018 MIND Computational Summer School. I will add more on this) 
