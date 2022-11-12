---
layout: post
title: Neuroimaging - Naturalistic Data Analysis
date: 2022-07-15 15:09:00
description: naturalistic data analysis
tags: neuroimaging psychology analysis methods
categories: neuroimaging
---

The field of psychology and neuroscience is very close to each other. Both fields essentially want to understand the processes of psychological phenomena.
Why do we study the brain processes in neuroscience with the goal of understanding psychological processes?

Overall, neuroimaging tasks used to investigate psychological processes can be categorized into classic controlled tasks and naturalistic tasks. 
For many years, Classic controlled tasks have been used to study the visual system, decision-making, etc.
However, it has recently been noticed that for relatively abstract and endogenous process, such as affect or social cognition, controlled tasks with very specific predefined stimuli and controlled experimental environment may oversimplify the complex underlying processes that is actually occuring.

I want to note that controlled tasks have effectively identified certain "complex" processes and they also have many advantages compared to naturalistic tasks.

For the upcoming posts, I will be reviewing the various analysis methods of naturalistic tasks that reflect relatively abstract and endogenous psychological processes, such as affect and social cognition.
The tutorials are based on <a href="https://naturalistic-data.org">the Naturalistic Data Analysis tutorial</a> built by multiple researchers in the social and affective neuroscience field.
I have reorganized and re-explained the tutorials to make it more comprehensive, personally. :)
These methods correspond to the questions asked in the various stages of naturalistic data analysis. This only suggests that these methods can be used in these steps of naturalistic data analysis and their usage is not limited to the field of naturalistic tasks. In fact, they are commonly used in other types of tasks including block or event design controlled tasks, multivariate analyses, and Bayesian models.

Research Questions for Naturalistic Tasks
<ol>
    <li>How are stimuli defined in naturalistic tasks?
        <ol>
            <li>Use the reliability of neural responses : Intersubject Correlation</li>
            <li>Minimize individual variation with functional alignment</li>
            <li>Define stimuli using automated annotations</li>
            <li>Define stimuli using natural language processing</li>
        </ol>
    </li>
    <li>How does the brain segment information from experiences? Hidden Markov Model</li>
    <li>How do networks of brain regions dynamically reconfigure as thoughts and experiences change over time? Hidden Semi-Markov Model</li>
    <li>How do networks of brain regions (FC) interact with other networks? Dynamic Connectivity</li>
    <li> How do we visualize complex high-dimensional data? Embedding with Hypertools</li>
</ol>