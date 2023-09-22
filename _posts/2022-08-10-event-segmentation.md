---
layout: post
title: Event Segmentation
date: 2022-08-10 15:09:00
description: how the brain segments info
tags: neuroimaging psychology analysis-methods
categories: naturalistic-data-analysis
---

This section shows how to detect event boundaries in fMRI data, which is defined as shifts in spatial patterns of voxel activity, and align events across perception and recall. This is especially important for naturalistic tasks where various events fluctuate across time and it may be hard to clearly identify which events are happening at timepoints, compared to controlled tasks that have predefined structures (e.g. blocks, events). 

We’ll show how both **Hidden Markov Model (HMM)** and **Greedy State Boundary Search (GSBS)** can be used to find boundaries, and how the HMM or GSBS+HMM can be used to track reactivation of event patterns during free recall.

[This video](https://youtu.be/-iDMphdGVxo) by Dr. Chris Baldassano, PhD introduces event segmentation using HMMs.

[This video](https://youtu.be/KvwzjRtbJ6U) by Dr. Linda Geerligs, PhD will discuss event segmentation using GSBS.

I’ve summarized the content of the videos here.


#### **Event Segmentation - HMM**

We divide naturalistic environments into discrete events, separated by event boundaries. These event boundaries are known to influence the dynamics of perception, immediate recall, and long-term memory.

How do we identify event boundaries? First, human observers can annotate when event boundaries occur (coarse and fine timescale) and check whether brain regions show response at boundaries. The problem is that every brain region may have different event boundaries so it’s hard to generate all the possible hypotheses on where the boundaries might be.

Instead, can we find event boundaries directly from fMRI data? The basic idea is to identify event boundaries as shifts in fMRI activity patterns. For example, if we look at these three voxel time courses, we can identify two moments at which these voxels show shifts in activities, diving up the time courses into three events, each of which has a characteristic spatial pattern of activity. 

![eventseg]({{ site.baseurl }}/assets/img/es/eventsegexample.png){: width="50%" }

We can see this temporal structure by looking at the angular gyrus during the beginning of the Sherlock dataset. If we take the spatial patterns in the angular gyrus and measure the correlation between all pairs of timepoints, we can obtain a matrix with **blocky structure along the diagonal**. This is the type of structure we’re looking for in event boundaries.

![hmm-block]({{ site.baseurl }}/assets/img/es/hmm-blockpattern.png){: width="100%" }

**Hidden Markov Model (HMM)** assumes the brain moves through a sequence of latent states which correspond to events ($$s_t$$). We observe a sequence of brain activity ($$b_t$$). Every event has some characteristic pattern of activity ($$m_k$$). All brain activities occurring during that event should be correlated with that characteristic. Thus, given a sequence of brain activity ($$b_t$$), we can infer where event boundaries are (which event every time point belongs to, $$s_t$$) and what these characteristic event patterns are ($$m_k$$). The model alternates between estimating variables $$s_t$$ and $$m_k$$ until convergence. 

![hmm]({{ site.baseurl }}/assets/img/es/hmm.png){: width="100%" }

HMM requires choosing the number of boundaries we want to find. One option is to fit the model on N-1 subjects to find boundaries and look at spatial pattern correlations within and across boundaries. We can choose the number of events ($$k$$) that maximizes the **within vs across (all other states) event boundary correlation (WAC)**. Another approach is to look at the **log-likelihood** of the model, which is to look at the model fit. We can train the model on some subjects can measure the model fit on other subjects, determining the number of events for which this model log-likelihood is highest. 

After identifying event boundaries of the model in different brain regions, we can **compare those boundaries to those annotated by human observers**. In general, we found that regions like the angular gyrus and the posterior medial cortex have event boundaries that correspond well with human annotated boundaries. When we look at how the optimal number of event boundaries varies across regions, we find a gradient from the sensory cortex to higher-level cortex such that regions like the visual cortex and auditory cortex have a large number of event boundaries, whereas high-level cortex have longer events in the scale of up to a minute in length.


How do we use the event boundaries that we found? 
1. Correlation between the number of event boundaries and characteristics of the stimuli (states)

2. Find shared event patterns across datasets with potentially different timings in order to :
    - compare different modalities (e.g. movie vs audio of same story shows modality-independent pattern sequences in some brain regions)
    - compare perception and later free recall (e.g. reactivation of events during free recall)
    - detect temporal shifts between first and rewatch of movies (e.g. after rewatch, event boundaries shift earlier in time up to 12 seconds)
    - see if different stories have similar underlying temporal structure (e.g. find brain regions that can classify schematic event sequences of stories-mpfc)

This HMM approach has also been applied to other neuroimaging modalities such as EEG.


#### **Event Segmentation – GSBS**

Events organize our experience over time by transforming continuous inputs into units that can be understood and remembered. Events are shared across participants, organized hierarchically. How does the brain segment information over time into neural states? 

Testing the event segmentation method introduced by Baldassano et al. (2018), we found some issues : within vs across event boundary correlation leads to over-estimation of the number of states. Also, HMM takes a long time to fit. 

Thus, we built a state boundary detection method that: detects state boundaries and the optimal number of states, does not have any assumptions about where state boundaries should be (data-driven), and is fast. The greedy state boundary search (GSBS) using a greedy search approach to find boundaries iteratively. 

First, find the initial state boundary from all the timepoints :

1. Taking each timepoint as a boundary, calculate mean activity pattern per state that is divided by the boundary (i.e. states before and after boundaries)
2. For all the potential boundaries (timepoints), calculate how much the original voxel activity time course in a state can be explained by the mean activity pattern in the corresponding state (i.e. correlation)
3. average the correlations for all the boundaries (i.e. timepoints) thus getting a single estimate for the fit,
4. find the optimal boundary (timepoint) when average correlation is highest (max).

Now, the first state boundary is set (that’s why this is greedy). Then, we repeat the process to find the next boundary until we find the number of boundaries we aimed for. 

![gsbs]({{ site.baseurl }}/assets/img/es/gsbs.png){: width="100%" }


So, how do we determine the number of boundaries? In the time x time correlation matrix, we look at within state correlations and between consecutive state correlations. With the distribution of within state correlations and the distribution between consecutive state correlations, we find the optimal $$k$$ number of state boundaries where the distance between the two distributions is maximal (i.e., when within state correlation is highest and between consecutive state correlation is lowest). To measure the distance, we use **T-distance** by running t-tests contrasting the within and between state correlation for each potential number of states and finding the maximum T-distance.

![gsbs-num]({{ site.baseurl }}/assets/img/es/gsbs-numofboundaries.png){: width="100%" }


We compared HMM and GSBS methods on simulated data. As states had more varying lengths, GSBS maintained high accuracy but HMM’s accuracy dropped. When looking at the worst performing simulation for HMM, **HMM over-estimates the number of state boundaries as it detects states with approximately the same lengths.** When looking at GSBS with the same low performance has HMM, **GSBS detects all the boundaries with slight delays in some.** Note that the split-merge option in the latest implementation of the HMM-method is partially accounted for the over-estimation issue. 

As for determining the number of states, GSBS accurately estimates the number of states whereas HMM’s WAC tends to over-estimate the number of states especially as there are more states. However, GSBS does over-estimate when there is more noise in the data. While LOO CV still gave over-estimation, averaging the data across subjects accurately estimated the number of states (in simulated and real data).

The weakness of GSBS is that it is difficult to match neural states across datasets (because it's data driven?). Thus, we suggest a method combing GSBS and HMM by estimating the neural states in one dataset and using HMM to identify the same activity patterns in a different dataset.


Examining the duration of states identified by GSBS in the whole brain, we found a brain-wide hierarchy in the timescales of information processing, where sensory regions show short neural states and regions such as the medial prefrontal cortex show long periods of information integration. We also correlated the behaviorally identified event boundaries and neural state boundaries, considering the possible delays of 4-8 seconds and using permutation for significance testing.

Some state transitions are stronger than others. Using the correlation distance between mean neural activity pattern of states as the strength of boundaries, we find more areas that co-occur with subjective boundaries. Thus, stronger event boundaries (strong state transitions) are more likely to be accompanied with the subjective experience of boundaries.
