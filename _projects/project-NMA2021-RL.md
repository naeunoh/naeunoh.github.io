---
layout: page
title: Building a controller in model-free and model-based RL in two-step task
description: NMA2021 Deep Learning Project
img: /assets/img/project_NMA_DL/Falling Humans_title.jpg
importance: 1
category: work
---

Unveiling the fundamental cognitive process of maximizing reward in a naturalistic environment has been a long-going challenge in both cognitive neuroscience and computer science. Recently, there are many attempts to use reinforcement learning to better understand the relevant mechanisms of the human brain.The human brain employs both exploration and exploitation strategies in learning about a new environment, where they generally apply model-free and model-based RL algorithms. The standard view is that both algorithms run in parallel either through integration or competition. However, previous work assumes unidirectional transition from model-free to model-based algorithms but not vice versa, which is more likely to occur in a changing environment. Moreover, the fundamental structure that directly employs algorithms in parallel before sufficient evidence accumulation may not adequately explain human behavior of exploration under uncertain environments.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project_NMA_DL/Falling Humans_intro.jpg" title="reinforcement learning models" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Reinforcement Learning Models : Model-free, Model-based, and our model.
</div>

This research intends to investigate how the transition between model-free and model-based learning contributes to adaptive learning under uncertainty, especially how different models represent exploitation and exploration. We simulate various agents of RL playing a two-step bandit task with shifting probabilities of rewards and state transitions. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project_NMA_DL/Falling Humans_methods.jpg" title="methods-task" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Two-step bandit task with shifting probabilities of rewards and state transitions.
</div>

We define a model that shifts between model-free and model-based RL with an independent controller module based on uncertainty. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project_NMA_DL/Falling Humans_functions.jpg" title="methods-task" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Algorithms of the model-free and model-based components of our model.
</div>

This is how we defined the independent controller.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project_NMA_DL/Falling Humans_controller.jpg" title="methods-task" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The controller of our RL model that decides whether to shift between model-free and model-based learning.
</div>


We expect the shifting model to outperform the independent RLs on the fraction of rewarded trials and total reward. In addition, the shifting model will exhibit patterns of stay probabilities that differ from independent RLs. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project_NMA_DL/Falling Humans_results.jpg" title="methods-task" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


The results show that ...
Average reward
Mf : does not learn the task -> does not take into account state transition probabilities
Mb : learns the task and gets rewards at an average of 0.6
Controller : takes longer to learn reward but gets higher rewards than mb of average of 0.7

Model switching helps the agent to maximize reward under uncertain naturalistic environment.
Functional model switching structure can be made with event memory and entropy calculation.


This research suggests the adaptive transition between learning models provides a better understanding of the decision making process and its underlying architecture.


