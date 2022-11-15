---
layout: page
title: The Hierarchical Architecture of the Visual Cortex?
description: NMA2021 Computational Neuroscience Project
img: /assets/img/12.jpg
importance: 2
category: work
---

Human visual system is known to have an anatomical and functional hierarchical structure. Different regions of the visual cortex process different visual information in a hierarchy from basic features to object recognition. Recent advances in computer vision allow deep neural network models (DNN) to recognize visual stimulation with high accuracy. Does the DNN capture feature information of images in a way that the brain does? If so, what does this imply about the hierarchy of the visual system as a network?
Using fMRI responses while viewing images, we explore whether DNN models can explain the hierarchy of representations in the visual cortex.

There are two hypotheses we want to test.

First, we want to examine the correspondance of the hierarchy of DNN model to the visual cortex by testing whether fMRI responses in the visual cortex can be predicted by the layers of a DNN model. We will extract each layer activations of a image classification DNN model (e.g. AlexNet, CorNet) for all the images of Kay et al. fMRI dataset. Then, we will train voxel-wise encoding models to predict fMRI responses in different regions of the visual cortex (7 regions including V1, V2, V3, V3A, V3B, V4, LatOcc) from the activations of each model layer (8 layers), and compute prediction performance on an separate test dataset.  If our hypothesis is correct, we will find that activations of the initial layers in the model accurately predict the neural reponses of lower regions in visual cortex, and vice-versa. Also, we will compare the results with a null model performance, i.e. the prediction performance of a randomly initialized DNN model not trained on image classification.

Second, we will examine the representations of the hierarchical architecture of visual processing. We hypothesize that the lower regions in the visual cortex represent low-level information (e.g., spatial information) as do the initial layers of the DNN model, whereas the higher regions in the visual cortex represent high-level information (e.g., categorial information) as do the deeper layers of the DNN model. First, to explore which brain region preserves spatial information, we will feed spatially transformed images (e.g. translation, scale, rotation, clutter) to the DNN model and observe how it changes the performance of layers predicting fMRI responses in each region. The layers and regions where encoding performance suffers most is considered to be more relevant to spatial information. Next, to find which region encodes categorical information, we will calculate representational dissimilarity matrices of fMRI responses to images in each brain region. If a brain region encodes categorical information, we will see patterns of similarity for images in the same category in the dissimilarity matrix.


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
