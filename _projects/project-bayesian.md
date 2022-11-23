---
layout: page
title: Bayesian Model of Social Learning
description: Coomputational Modeling Coursework Project
img: /assets/img/12.jpg
importance: 4
category: work
---

Project on the effects of personal preferences on learning and choice process in social learning task. Tested Additional effects of relative item popularity on these personal preferences. Used stan for Bayesian modeling and inference. Used PyMC3 to fit model with MCMC algorithms.

Tarantola et al.(2017) found that our personal preferences affect the way we learn the preferences of other people. Through computational models combining inter-trial Bayesian learning and intra-trial choice process, they found the effects of participants’ preferences on both the learning and choice process. The preferences were reflected on the learning process through the influence of priors and on the choice process through the influence of the choice bias. These effect generalized to non-social learning experiment. When they modeled the influence of relative item popularity on the prior in the Bayesian learning process, they could find that only the participants in the social learning experiment additionally benefit by using their knowledge about the popularity of certain preferences.
However, they didn’t take into account the possible influence of item popularity on the choice bias. Therefore, we modified the author’s code and fitted two new models. 


References
• Tarantola, T., Kumaran, D., Dayan, P., & De Martino, B. (2017). Prior preferences beneficially influence social and non-social learning. Nature communications, 8(1), 817.