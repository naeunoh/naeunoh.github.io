---
layout: post
title: Intersubject Correlation
date: 2022-07-17 15:09:00
description: synchrony as reliability of brain activity
tags: neuroimaging psychology analysis-methods
categories: naturalistic-data-analysis
---

Synchrony of brain activity is associated with shared psychological perspectives toward a stimulus, friendship, and psychiatric conditions. **Intersubject Correlation (ISC)** (Hasson et al., 2004) calculates linear correlations between participants (pairwise or similarity to average) and derives summary statistics (overall level of synchrony) from these correlations to measure the level of similarity of brain activity. 

The brain activity measured with fMRI during naturalistic stimulation conditions (but also applies to controlled tasks or resting-state, honestly) can be thought to consist of four main sources : 
><ol>
>    <li> Stimulus-driven brain activity that is shared by most participants</li>
>    <li> Individual/idiosyncratic activity elicited by the stimulus</li>
>    <li> Intrinsic activity that is not time-locked to the stimulus</li>
>    <li> Noise from various sources</li>
></ol>

(1) The idea behind ISC is to identify brain activity that is shared by many. Thus, this method evaluates how much of an individual’s brain activity is explained by this shared component.

(2) By contrast, if smaller groups of participants (e.g. friends) share similar individual activity patterns, it may be better captured by the dyadic values in the pairwise matrices using techniques such as Intersubject Representational Similarity Analysis(IS-RSA). 

(3) The third category of activity is not readily detected by synchrony approaches, but in some innovative designs (Chen et al., 2017), it is still possible to extract shared brain activity patterns by temporally reorganizing the data (e.g. during verbal recall of previously experienced stimuli) even when the original experiences of participants were out of sync.    

**Thoughts:** How is (3) different from resting-state? 

Resting-state involves no external synchronizing factors apart from the repeating noise of the scanner gradients (and noise is not of our interest) and thus is ideal for demonstrating the true null distribution of no synchrony. I believe resting state involves both 3 and 4.



### Calculating ISC


ISC are mostly calculated locally within each voxel or region, but the method has been extended to functional connectivity (e.g. ISFC), which will be dealt with later. 

The first step of ISCs is *calculating individual synchrony* using one of two main approaches. First, one calculates pairwise correlations between all participant pairs to build a full intersubject correlation matrix. The second approach uses the average activity timecourse of other participants as a model for each individual left out participant. This produces individual, rather than pairwise, spatial maps of similarity (how typical one’s brain activation is) in the same way first level results of a traditional general linear model analysis would. However, some individual variability is lost with the average similarity approach and ISC values are typically higher than pairwise matrices.

The second step is to summarize the overall level of synchrony across participants. One can use the mean correlation. To make the correlation coefficients more normally distributed across the range of values, the Fisher’s Z transformation (inverse hyperbolic tangent) is applied before computing the mean. This transformation mainly affects the higher absolute correlation values, thus stretching the correlation coefficient 1 to infinity. However, as pairwise ISC values are typically not that high, the effects of this transformation are relatively small reaching less than 10% at the higher end of the scale of r=0.5. Recently, it has been suggested that computing the median, especially when using the pairwise approach, provides a more accurate summary of the correlation values (Chen et al., 2016).



### Hypothesis Tests with ISC


Now, we will perform hypothesis tests with ISC. Performing hypothesis tests that account for the false positive rate can be tricky with ISC because of the dependence between the pairwise correlation values and the inflated number of variables in the pairwise correlation matrices. Although there have been proposals to use mixed-effects models for a parametric solution (Chen et al., 2017), *non-parametric statistics* are recommended. 

The first non-parametric approach is permutation or randomization achieved by creating surrogate (artificial, fake) data and repeating the same analysis many times to build an empirical null distribution (e.g. 5-10k iterations). The null distribution represents the condition where any correlations in the data arise by chance (like a resting state condition). However, to meet the exchangeability assumption(?), it is important to consider the temporal dependence structure (because our data is sequential). Surrogate data can be created by circularly shifting the timecourses of the participants (circular shifting) or by scrambling the phases of the Fourier transform of the signals and transforming these signals back to the time domain (phase randomization). Various blockwise scrambling techniques and autoregressive models have been proposed to create artificial data for statistical inference. When properly designed, these methods can retain important characteristics of the original signal (e.g. frequency content and autocorrelation) while removing temporal synchrony in the data. 

The second non-parametric approach employs a subject-wise bootstrap on the pairwise similarity matrices. Participants are randomly sampled with replacement and then a new similarity matrix is computed with these resampled participants. Due to replacement sampling, sometimes the same subjects are sampled multiple times which introduces correlation values of 1 off the diagonal. Thus, summarizing the ISC with median can minimize the impact of these outliers. These values are then shifted by the real summary statistics to produce an approximately zero-centered distribution? Note that Brainiak and nltools convert these values to NaNs by default. 



### Intersubject Functional Connectivity (ISFC)


To address how brain regions coactivate due to naturalistic stimulation, ISC was recently extended to **intersubject functional connectivity (ISFC)** to measure brain connectivity between subjects (Simony et al., 2016). This method can identify connections that are activated consistently between participants by the stimulus while disregarding the intrinsic fluctuations as they are not time-locked between individuals. (This is very effective as FC fluctuate a lot) This can also illustrate how distant brain regions cooperate to make sense of the incoming stimulus streams. However, it can also highlight pairs of regions that show similar temporal activity patterns that are driven by the external stimulus (just happen to activate at the same time) rather than neural connections between the regions (causality in the brain), which should be taken into account in the interpretation. This is an intrinsic problem of FC due to the way connectivity is calculated (correlation).



### Dynamic ISC


Intersubject correlations give a summary statistic of synchrony over long periods of time. However, as the level of synchrony may change from one moment to the next ("dynamic"), **time-varying measures of synchrony are also employed (e.g. dynamic ISC)**. For instance, tools like correlation assume a constant statistical dependence between the variables over the entire imaging session and thus may not be the most appropriate way to analyze data gathered during complex naturalistic stimulation.

We want to calculate temporal variability of synchrony while limiting the effects of signal amplitudes. In other words, we don’t want the correlations between two signals to fluctuate depending on their amplitudes.

A simple way is to calculate correlations within sliding time windows. This allows the estimation of synchrony during time windows when the signals are close to their mean values as the amplitude within each time window is standardized when the correlation is calculated. However, the length of the temporal window forces a trade-off between temporal accuracy and stability of the correlation coefficient calculated in that window. Very short time windows allow one to follow precisely when correlations occur, but they also yield extremely unstable correlations with extreme correlation values that change signs wildly, which can be dominated completely by (unreliable signals like) single co-occuring spikes or slopes. 

Another option is to calculate the phase synchronization or phase locking of signals (Intersubject Phase Synchrony, ISPS). This has been used widely for electrophysiological measures such as EEG and MEG, and more recently also for fMRI (Glerean et al., 2012). Phase synchronization leverages the Hilbert transform to transform the real-valued signals into a complex valued, analytic signal, which is a generalization of the phasor notation of sinusoidal signals that are widely used in engineering applications. 

(Whoa… what does that even mean? Let me explain.)

The illustration below shows two examples of analytic signals with constant frequency and amplitude, plotted in three dimensions (real, imaginary, and time axes). We have used the cosine of the angular difference as a measure of pairwise synchrony (cosine similarity). This produces time-averaged values that are consistent with the ISCs in the regions. In contrast to a time-invariant phasor ([phase vector](https://en.wikipedia.org/wiki/Phasor)), an analytic signal has a time-varying amplitude envelope (wave) and frequency and can thus be used to track changes in synchrony over time. However, for meaningful separation of the envelope and phase of the signal, the original signal must be contained in a limited frequency band, which can be obtained through *band-pass filtering*. The smaller this frequency band is, the better the amplitude envelope is separated into a lower frequency than the phase of the signal in the pass-band. However, poorly designed filters may affect the shape of the signal considerably and even remove the signal of interest. For instance, some filters can cause non-linear phase shifts across the frequency spectrum, or an excessively tight pass-band may miss important frequencies completely.


![phase_synchrony]({{ site.baseurl }}/assets/img/isc/PhaseSynchronyAndCorrelation.gif){: width="100%" }


Compared to sliding-window correlations, phase synchronization has the benefit that no explicit time windows are required and synchronization is estimated at the original sampling frequency of the signals (though you need to choose a narrow frequency band). However, in a single pairwise comparison, phase synchrony can get extreme values by chance even when the two signals are independent. Accordingly, the estimate of mean synchrony oscillates with the phase of the signals, until eventually stabilizing around zero as expected for independent signals. Thus, phase synchrony of two signals does have the potential of producing extreme values like the sliding-window correlations. This can be mitigated by averaging over the timepoints of a full session, which will produce ISPS that is similar to group-level results of ISC. But then, this removes the benefit of the temporal accuracy of ISPS. By contrast, *averaging over (pairs of) subjects* improves the reliability of synchrony in a larger population while retaining the temporal accuracy.
