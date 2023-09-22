---
layout: post
title: Intersubject RSA
date: 2022-07-20 15:09:00
description: individual variations in brain activity
tags: neuroimaging psychology analysis-methods
categories: naturalistic-data-analysis
---

Whereas ISC and related approached were traditionally developed to detect responses shared at the group level, we know that brain activity during naturalistic stimuli also shows interesting individual differences. If ISC by definition operated at the level of subject pairs, how can we use this to measure behaviors at the level of single subjects with individual differences?
If we take the subjects-by-subjects ISC matrix to be a brain similarity matrix, we can **construct a behavioral similarity matrix and use RSA to find brain regions where subjects who are more similar in behavior are also more similar in their neural response**. 


![isrsa]({{ site.baseurl }}/assets/img/IS-RSA/Fig1_multilayer_figure_R1.jpg){: width="100%" }

**Fig. 1.** Schematic of inter-subject representational similarity analysis. (Finn et al., 2020). Each subject (bottom layer) is associated with a behavioral score (middle layer) and a pattern of brain activity (top layer, e.g., a time series from a given brain region during naturalistic stimulation). The middle and upper layers depict weighted graphs obtained using the similarity matrices as adjacency matrices, where thicker lines indicate increased similarity between nodes (subjects). In IS-RSA, we construct pairwise (i.e, subject-by-subject) similarity matrices for the behavioral data and the brain data, then compare these matrices using a Mantel test. Thus, we can leverage inter-subject analysis methods such as ISC to detect shared structure between brain data and behavioral data. This figure is a modified version of Fig. 1 in Glerean et al. (2016).

[This video by Emily Finn, PhD](https://youtu.be/vDrMuFJfsv8) shows how inter-subject approaches can be more sensitive to phenotypic differences between individuals than other approached for analyzing naturalistic data.

[This video by Carolyn Parkinson, PhD](https://youtu.be/roG9gkTOx_U) discusses practical considerations for analyzing naturalistic data.


## Measuring similarity

How do we measure behavioral similarity? This is a basic concept of RSA in general. In choosing a distance metric, particularly when our behavior is one-dimensional (e.g. age, trait score, accuracy on a cognitive task), we bake in some fundamental assumptions about the structure of the brain-behavior representational similarity that affect the ultimate results and how we interpret them. Also, there is some evidence that computing similarity using responses to individual questions as an embedding space can create a richer representational space than using univariate summary scores (Chen et al., 2020). To get a feel for some potential structures, imagine arranging the rows and columns of the ISC matrix such that subjects are ordered by their behavioral score. What would we expect the resulting matrix to look like?

If we use Euclidean distance or another relative distance metric, we implicitly assume that subjects with closer scores should be more similar to one another, regardless of where they fall on the scale. In other words, for a behavior that is measured on a scale from 0 to 100, a pair of subjects scoring 0 and 1 should be just as similar as a pair of subjects scoring 99 and 100 (since in both scaes the Euclidean distance is 1). We call this the Nearest Neighbors (NN) model, which assumes that a subject should always look most similar to his or her immediate neighbors, regardless of their absolute position on the scale. 

**Thoughts:** this eminds me of cognitive tests, does the metric assume??

The NN model may be appropriate for certain behaviors (e.g. age, accuracy on attention or vision task..), but we could imagine an equally if not more plausible scenario : that similarity between subjects increases or decreases as one moves up or down the scale, in an absolute rather than relative sense. For example, perhaps high-scoring subjects are more similar to other high scorers, while low-scoring subjects are less similar both to high scorers and other low scorers. In other words, brain responses cluster together for subjects at one end of the behavioral spectrum, white variability increases as one moves toward the opposite end of the spectrum. We call this the Anna Karenina (AnnaK) model, after the famous opening line of Leo Tolstoy’s novel, which reads “All happy families are alike; each unhappy family is unhappy in its own way” (or, in this context, “all high [low] scorers are alike; each low [high] scorer is different in his or her own way”). In this case, Euclidean distance would not be the most appropriate choice. Instead, we would want to model similarity using a metric that reflects absolute position on the scale, such as the mean $$(i+j)/2$$, minimum $$min(i,j)$$, or the product of the mean and minimum.

## Things to consider when analyzing brain data

With fMRI data, we have a choice to whether we'd like to work in the original resolution of the data (voxels) or to summarize across space in some way. 

We could calculate similarity across the **whole brain** at once, but there are probably some regions where the representational similarity with behavior is stronger than in other regions, and we'd like to be able to visualize and say something about which regions are contributing most to our effect. 

We could calculate similarity at **each individual voxel** separately. This has the advantage of maximizing spatial specificity, but it's also expensive in terms of time and computation, and we know the BOLD response tends to be smoother than single voxels.

Another option would be to take a **searchlight** approach, where we calculate similarity with a searchlight (relatively small sphere or cube of voxels centered around a voxel). This preserves some degree of spatial specificity while boosting signal relative to single voxels (which can be noisy) and recognizing the inherent smoothness of the local BOLD response. But it still requires us to loop through every voxel which takes a lot of time and memory.
Furthermore, both single-voxel and searchlight approaches also lead to larger penalties when it comes time to correct for multiple comparisons, since we've effectively done as many tests as there are voxels, and we need to stringently control for false positives.

A happy medium is to summarize voxelwise data into **nodes or parcels**. We can use **predefined ROIs** to group voxels into contiguous regions. At each TR, we average signal in all the voxels in a node to get one representative timecourse for that node. This way, we cut down on computational complexity by several orders of magnitude (~70,000 brain voxels in a typical $$3mm^2$$ whole-brain acquisition vs ~100-300 nodes in most parcellations).

We will use a functional parcellation called the Shen atlas (Shen et al., 2013), which has 268 nodes. I personally like this parcellation because it covers the whole brain including the subcortex and cerebellum, whereas other parcellations only over cortex. Also, in general, parcellations in the 200-300 node range provide a good balance of spatial specificity without having nodes so small that they amplify registration errors and partial voluming effects from slight misalignments across subjects. In general, there is no single “true” parcellation : **it’s more of a data-reduction step and it’s never a bad idea to make sure your results are robust to the choice of parcellation.**

Note that while using parcellations is convenient for computational purposes, this approach may obscure finer-grained individual differences that emerge when considering smaller spatial scales. Feilong et al. (2018) used an IS-RSA approach to quantify the reliability of individual differences in cortical functional architecture across alignment methods, spatial scales, and functional indices, and found that individuals reliably differ in fine-scale cortical functional architecture when data were first hyperaligned. Furthermore, these individual differences in fine-grained cortical architecture provide strong predictions of general intelligence. Thus, combing functional alignment with IS-RSA approaches is a promising avenue for future work to strengthen our understanding of brain-behavior relationships.

