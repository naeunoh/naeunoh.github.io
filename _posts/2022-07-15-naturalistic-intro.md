---
layout: post
title: Neuroimaging - Naturalistic Data Analysis
date: 2022-07-15 15:09:00
description: intro to naturalistic methods
tags: neuroimaging psychology analysis-methods
categories: neuroimaging
---

The field of psychology and neuroscience is very close to each other. Both fields essentially want to understand the processes of psychological phenomena.
Why do we study the brain processes in neuroscience with the goal of understanding psychological processes?

Overall, neuroimaging tasks used to investigate psychological processes can be categorized into classic controlled tasks and naturalistic tasks. 
For many years, Classic controlled tasks have been used to study the visual system, decision-making, etc.
However, it has recently been noticed that for relatively abstract and endogenous process, such as affect or social cognition, controlled tasks with very specific predefined stimuli and controlled experimental environment may oversimplify the complex underlying processes that is actually occuring.

I want to note that controlled tasks have effectively identified certain "complex" processes and they also have many advantages compared to naturalistic tasks.

For the upcoming posts, I will be reviewing the various analysis methods of naturalistic tasks that reflect relatively abstract and endogenous psychological processes, such as affect and social cognition.
The tutorials are based on [the Naturalistic Data Analysis tutorial](https://naturalistic-data.org) built by multiple researchers in the social and affective neuroscience field.
I have reorganized and re-explained the tutorials to make it more comprehensive.

The methods described correspond to the questions asked in the various stages of naturalistic data analysis. Note that this only suggests that these methods can be used in these steps of naturalistic data analysis and their usage is not limited to the field of naturalistic tasks. In fact, they are commonly used in other types of tasks including block or event design controlled tasks, multivariate analyses, and Bayesian models.

>**Research Questions & Corresponding Methods for Naturalistic Tasks**
>
> 1. How do we build models using naturalistic designs?
>    1-1. Indirectly model by capturing “reliable” neural responses among subjects
>   - Minimize individual variations with Functional Alignment
>   - Predict one’s brain activity from another : Intersubject Correlation Intersubject Functional Connectivity
>   - Individual variations in brain activity : Intersubject Representational Similarity Analysis (IS-RSA)
>   - Dynamic changes in ISC : Intersubject Phase Synchrony
>   
>    1-2. Explicitly annotate features of the model
>   - Define stimuli using Automated Annotations
>   - Define stimuli using Natural Language Processing
> 2. How does the brain segment information from experiences? 
> Event Segmentation (Hidden Markov Model, Greedy State Boundary Search)
> 3. How do networks of brain regions dynamically reconfigure as thoughts and experiences change over time? 
> Hidden Semi-Markov Model
> 4. How do networks of brain regions (FC) interact in higher order patterns
> Dynamic Connectivity
> 5. How do we visualize complex high-dimensional data? 
> Embedding with Hypertools


**Naturalistic Datasets**

Here are some open-source fMRI datasets on naturalistic tasks.

[Human Connectome Project 7T movie-watching data](https://www.humanconnectome.org/study/hcp-young-adult) (n = 184, four 15-minute movie-watching runs, many cognitive, affective, and clinical measures)

[Child Mind Institute Health Brain Network project](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/index.html) (n = 2,500+ and growing, two movie-watching runs [one 10-minute and one 3.5-minute], many cognitive, affective, and clinical measures)

[Narratives: fMRI data for evaluating models of naturalistic language comprehension](https://openneuro.org/datasets/ds002345/versions/1.0.1) (n=315, auditory listening of naturalistic spoken stories [~3 to ~56 minutes], includes anatomical data)

[Paranoia: fMRI data for listening to audio narratives](https://openneuro.org/datasets/ds001338/versions/1.0.0) (n=22, auditory listening of original audio narrative designed to elicit individual variation of suspicion/paranoia)

There are also behavioral (only) datasets of naturalistic tasks.

[Standford Emotional Narratives Dataset](https://github.com/StanfordSocialNeuroscienceLab/SEND) (n=193, videos of people recounting important and emotional life stories, and valence ratings watching the videos)