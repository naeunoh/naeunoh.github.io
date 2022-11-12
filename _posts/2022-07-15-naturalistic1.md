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

Question 1. How are stimuli defined in naturalistic tasks?
1-1. Use the reliability of neural responses : Intersubject Correlation
1-2. Minimize individual variation with functional alignment 
1-3. Define stimuli using automated annotations
1-4. Define stimuli using natural language processing
Question 2. How does the brain segment information from experiences? Hidden Markov Model
Question 3. How do networks of brain regions dynamically reconfigure as thoughts and experiences change over time? Hidden Semi-Markov Model
Question 4. How do networks of brain regions (FC) interact with other networks? Dynamic Connectivity
Question 5. How do we visualize complex high-dimensional data? embedding with Hypertools

{% raw  %}
{% highlight c++ linenos %}  <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers.
Produces something like this:

{% highlight c++ linenos %}

int main(int argc, char const \*argv[])
{
    string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}

{% endhighlight %}
