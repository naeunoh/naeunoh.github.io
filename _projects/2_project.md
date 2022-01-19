---
layout: page
title: Building a controller in model-free and model-based RL in two-step task
description: NMA2021 Deep Learning Project
img: /assets/img/3.jpg
importance: 1
category: work
---

Unveiling the fundamental cognitive process of maximizing reward in a naturalistic environment has been a long-going challenge in both cognitive neuroscience and computer science. Recently, there are many attempts to use reinforcement learning to better understand the relevant mechanisms of the human brain.The human brain employs both exploration and exploitation strategies in learning about a new environment, where they generally apply model-free and model-based RL algorithms. The standard view is that both algorithms run in parallel either through integration or competition. However, previous work assumes unidirectional transition from model-free to model-based algorithms but not vice versa, which is more likely to occur in a changing environment. Moreover, the fundamental structure that directly employs algorithms in parallel before sufficient evidence accumulation may not adequately explain human behavior of exploration under uncertain environments.

This research intends to investigate how the transition between model-free and model-based learning contributes to adaptive learning under uncertainty, especially how different models represent exploitation and exploration. We simulate various agents of RL playing a two-step bandit task with shifting probabilities of rewards and state transitions. We define a model that shifts between model-free and model-based RL with an independent controller module based on uncertainty. We expect the shifting model to outperform the independent RLs on the fraction of rewarded trials and total reward. In addition, the shifting model will exhibit patterns of stay probabilities that differ from independent RLs. This research suggests the adaptive transition between learning models provides a better understanding of the decision making process and its underlying architecture.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, *bled* for your project, and then... you reveal it's glory in the next row of images.


<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>


The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}
```html
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
```
{% endraw %}
