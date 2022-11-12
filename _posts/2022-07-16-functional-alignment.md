---
layout: post
title: Functional Alignment
date: 2022-07-16 15:09:00
description: naturalistic data analysis
tags: neuroimaging psychology analysis-methods
categories: naturalistic data analysis
---


Generally, neuroimaging studies implicitly assumes that each person’s brain processes information in the same way as other people’s brains. However, it is a known fact that there is a wide variation in individual neuroanatomy. 

Thus, to identify the common brain processes, we must normalize each participant into a common stereotactic space. This would minimize the individual variation that we are not interested in.

One way is applying a small amount of Gaussian smoothing to mitigate small misalignments and increase voxel signal to noise ratios (SNR). This method is effective when performing mass univariate testing but problematic when using multivariate techniques that assume strong alignment of features (e.g. voxels) across participants.

Another newly developing approach lead by Jim Haxby and Peter Ramadge projects subjects into common space based on how voxels respond to stimuli or are connected to other voxels. This is called functional alignment or hyperalignment.

The basic idea behind hyperalignment is to...
(1) Treat cortical patterns as vectors corresponding to locations in a high dimensional space, where each axis reflects a measurement of that pattern (e.g. voxel activity). 

**Note:** <a href="https://youtu.be/fNk_zzaMoSs">3brown1blue video</a> on vectors might help understand the general concept.

Rather than treating cortical functions in a 1D (average roi activity), 2D (cortical sheet), or 3D physical space, HA models information as being embedded in an n-D space, where n reflects the number of measurements (e.g. voxels). Simply put, a cortical pattern of a timepoint t (e.g. conditions, stimuli, timepoints) is vectorized or flattened into a vector and then expressed as a location point/position in the n-dimensional space where each axis is a voxel. 

(2) Vector representations of individual cortical patterns can be transformed into a common dimensional space that is shared across participants (Haxby et al., 2000). Each cortical pattern has a unique transformation matrix (voxel x dimension). 
This framework can be applied to cortical patterns of voxel responses (response-based, vector=timepoint x voxel) or functional connectivity (connectivity-based, vector=target x connected node) or latent feature space (shared response model, joint-SVD).
Overall, the lower dimensional projections (onto common dimensional space) remove redundant axes in the high dimensional space and can effectively denoise the signal, which also improves classification accuracy of subsequent analysis. 
(For detailed information, watch <a href="https://youtu.be/QX7sNaLyxdo">video by Dr. James Haxby, PhD from the 2018 MIND Computational Summer School</a>. I will add more on this) 
