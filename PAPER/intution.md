# Notes for a TFS Paper on Fuzzy Symmetry in MARL

I am writing a paper targeting IEEE Transactions on Fuzzy Systems (TFS). The topic is modeling and exploiting fuzzy symmetry in multi-agent reinforcement learning (MARL). This document preserves the main research ideas while organizing them in a form closer to a paper draft, so that it can be directly expanded into the abstract, introduction, problem formulation, method, and theory sections.

## Core Idea

Existing symmetry-aware MARL methods usually treat symmetry as either exact or partially valid under bounded deviations. In real-world environments, however, the symmetry of multi-agent systems is often affected by wind, uneven terrain, sensing noise, actuation errors, and other disturbances, which makes it imperfect rather than ideal. The AAAI 2024 work on partial symmetry already shows that this kind of imperfect symmetry can be characterized by reward and transition deviations under transformation, and can therefore be quantified as a continuous degree rather than a binary condition. From a TFS perspective, this means that partial symmetry already contains the essential ingredients of fuzzy symmetry: symmetry is not simply valid or invalid, but has degree, local reliability, and adaptive exploitable strength.

More importantly, most existing studies still exploit symmetry at a relatively shallow level, such as through data augmentation, consistency losses, or replay-based mechanisms. Comparatively fewer works embed symmetry deeply into model structure and policy learning. Even when symmetry-aware architectures are designed, they often still assume exact symmetry or globally approximate symmetry, which makes it difficult to achieve a robust balance between expressive flexibility and generalization. This motivates us to reformulate the core question from whether symmetry holds to where symmetry holds, to what degree it holds, and how strongly it should be exploited.

## Contributions

- Reinterpret partial symmetry from a fuzzy-systems perspective and formulate it as fuzzy symmetry with continuous membership, so that local symmetry satisfaction can be described in a degree-aware way.
- Derive a performance error bound showing that the error induced by symmetry exploitation decreases as symmetry membership increases.
- Design a general fuzzy symmetry exploitation framework that augments an equivariant structural backbone with a residual bypass branch, enabling the model to handle imperfect symmetry.
- Validate the framework in both simulation and real-robot settings, covering aggregation, pursuit, navigation, and StarCraft scenarios.

## Method Overview

We propose a fuzzy symmetry-aware framework that combines a symmetry-constrained equivariant branch with an unconstrained residual branch, and adaptively balances the two according to the degree of local symmetry.

From a modeling perspective, the framework provides a continuously adjustable unified form between a strictly symmetry-constrained model and an unconstrained model. As a result, the policy network can retain the generalization advantage induced by symmetry while remaining adaptive to real-world disturbances and asymmetric structures. More concretely, a structured relaxation is introduced between the strict symmetry hypothesis space and a larger function space, yielding a model form of symmetric structural backbone plus asymmetric compensation, with adaptive control over how strongly symmetry is embedded.

## Writing Requirements

### Abstract

The abstract should contain 6-8 sentences and clearly include:

- The importance of symmetry in MARL.
- The limitation of existing methods under exact symmetry or partial symmetry assumptions.
- The key observation that symmetry in real environments has degree, rather than being simply present or absent.
- The proposed idea of modeling symmetry as a fuzzy structural property with continuous membership.
- The theoretical contribution, namely an error bound related to symmetry membership.
- The method overview, namely a unified framework that adaptively combines an equivariant branch with a residual branch.
- The empirical benefits, including improved robustness and sample efficiency in both simulation and real-world settings.

### Introduction

The introduction should contain around 6-8 paragraphs with a clear logical progression.

1. Explain why symmetry is important in MARL and why it benefits sample efficiency and generalization.
2. Review existing approaches, including hard equivariant methods and soft regularization methods.
3. Point out their common limitation: most assume perfect symmetry or uniformly approximate symmetry.
4. Emphasize that real-world symmetry is typically non-uniform, imperfect, and degree-varying.
5. Discuss the value of existing partial symmetry studies while noting that they still mainly rely on augmentation or global approximation assumptions and often incur additional computational cost.
6. Introduce our perspective: fuzzy symmetry as a continuous structural prior.
7. Explain how we build a continuously adjustable unified framework between strictly symmetry-constrained and unconstrained models, so that the policy retains structural generalization while adapting to local asymmetry.
8. Summarize the theory and method, then conclude with bullet-point contributions.

## Style Requirements

- Use a formal academic tone suitable for TFS.
- Avoid exaggerated claims.
- Maintain clear logical transitions between paragraphs.
- Avoid unnecessary jargon.
- Emphasize problem reformulation rather than merely proposing a new method.

## Key Message

The main contribution of this work is not merely a new architecture, but a new way to model symmetry in MARL: the idea of partial symmetry is further reformulated from approximate symmetry with relaxation bounds into a fuzzy structural property with continuous membership, and this property is deeply embedded into representation learning and policy optimization in an adaptive manner.

# Problem Formulation

## Background

Symmetry has been widely recognized as an effective structural prior in multi-agent reinforcement learning because it improves generalization and sample efficiency. Existing approaches typically exploit symmetry either through hard architectural constraints, such as equivariant networks, or through soft regularization techniques, such as data augmentation and consistency losses. These methods, however, usually assume that symmetry is either strictly satisfied or only violated within some globally bounded deviation.

In practical multi-agent systems, symmetry is rarely homogeneous. Environmental disturbances, agent heterogeneity, and task-dependent asymmetries all introduce non-uniform and context-dependent symmetry violations. More specifically, wind variation, uneven terrain, obstacles, and sensing or actuation noise may cause the same transformation to be highly reliable in some regions but much less reliable in others. As a result, symmetry cannot be adequately characterized as a binary property or a globally bounded property.

To address this limitation, we propose to model symmetry in MARL as a fuzzy structural property characterized by a continuous degree of satisfaction. More precisely, we do not introduce a completely disconnected notion from prior work. Instead, we build on the theoretical foundation of partial symmetry and reinterpret this approximately valid but non-binary structural regularity as a fuzzy membership structure.

## Preliminaries

We consider a Markov game defined as

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, r, G \rangle,
$$

where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the joint action space, $P$ is the transition function, $r$ is the reward function, and $G$ is a group acting on the state-action space.

For $g \in G$, the group action is denoted as

$$
g \cdot (s, a) = (g \cdot s, g \cdot a).
$$

Classical symmetry requires

$$
P(g \cdot s' \mid g \cdot s, g \cdot a) = P(s' \mid s, a), \quad
r(g \cdot s, g \cdot a) = r(s, a).
$$

## Fuzzy Symmetry

We generalize symmetry into a fuzzy concept. The starting point is aligned with the AAAI 2024 definition of partial symmetry: realistic environments no longer satisfy strict equivariance or invariance, but only approximately satisfy the corresponding transformation relationships within tolerable deviations. In this sense, partial symmetry can be viewed as a direct predecessor of fuzzy symmetry. The difference is that the former is mainly expressed by deviation bounds and approximation radii, whereas the latter further reformulates such approximation into continuous membership.

**Definition (Fuzzy Symmetry Membership).** For a transformation $g \in G$, the fuzzy symmetry at a state-action pair $(s, a)$ is defined as a membership function

$$
\mu(s, a; g) \in [0,1],
$$

which quantifies the degree to which symmetry is satisfied under transformation $g$.

Following the modeling logic of partial symmetry, one may first require relaxed consistency constraints between the original and transformed samples:

$$
|r(s,a)-r(g \cdot s, g \cdot a)| \le \varepsilon,
$$

and

$$
D\big(P(\cdot|s,a),\, g^{-1}P(\cdot|g \cdot s, g \cdot a)\big)\le \delta.
$$

These conditions already show that symmetry is no longer a discrete proposition, but a continuous property determined by the size of reward and transition deviations. We therefore further express it as a fuzzy membership function. A practical construction is given by

$$
\mu(s, a; g) =
\phi_r\!\left(|r(s,a) - r(g \cdot s, g \cdot a)|\right)
\cdot
\phi_p\!\left(D\big(P(\cdot|s,a),\, g^{-1}P(\cdot|g \cdot s, g \cdot a)\big)\right),
$$

where $\phi_r(\cdot)$ and $\phi_p(\cdot)$ are monotonically decreasing mappings, and $D(\cdot,\cdot)$ is a divergence measure. Under this formulation, partial symmetry can be understood as a concrete realization of fuzzy symmetry: $\varepsilon$ and $\delta$ characterize the extent of deviation from strict symmetry, while $\mu(s,a;g)$ further normalizes this extent into a continuous membership value in $[0,1]$.

We further define the fuzzy symmetry of a policy-induced distribution as

$$
\mu_G(\pi) = \mathbb{E}_{(s,a) \sim \rho^\pi} \left[ \operatorname{Agg}_{g \in G} \mu(s,a; g) \right],
$$

where $\rho^\pi$ denotes the state-action visitation distribution, and $\operatorname{Agg}$ is an aggregation operator such as the mean or minimum.

This formulation unifies several classical cases:

- Exact symmetry: $\mu = 1$.
- Broken symmetry: $\mu \approx 0$.
- Partial symmetry: corresponding to continuous intermediate membership.

Therefore, from a fuzzy-systems viewpoint, partial symmetry and fuzzy symmetry are not two unrelated notions. A more accurate statement is that partial symmetry provides the RL/MARL formal foundation for fuzzy symmetry, while fuzzy symmetry offers a more natural degree-aware language, a unified interpretation, and a better interface for explainable modeling.

## Problem Statement

Given a Markov game with unknown and non-uniform symmetry structure, our goal is to learn a policy $\pi$ that maximizes expected return while adaptively exploiting fuzzy symmetry:

$$
\max_\pi \; J(\pi),
$$

where the strength of symmetry exploitation is modulated by $\mu(s,a;g)$.

This introduces a fundamental trade-off: enforcing symmetry in low-symmetry regions leads to bias, while ignoring symmetry in high-symmetry regions leads to inefficiency. Our objective is to balance this trade-off in a continuous manner and thereby achieve a better compromise between structural bias and model flexibility.

# Fuzzy Symmetry-Aware MARL Framework

## Overview

We propose a unified framework that integrates fuzzy symmetry estimation, adaptive structural modeling, and symmetry-aware learning. The framework consists of three components:

1. Fuzzy symmetry estimation.
2. Fuzzy-structured policy or value networks.
3. Fuzzy symmetry-regularized learning objective.

## Fuzzy Symmetry Estimation

We estimate the membership function

$$
\mu_\psi(s,a,g) \in [0,1],
$$

which characterizes the reliability of symmetry under transformation $g$.

The estimator can be constructed using:

- Analytical deviation measures based on reward and transition.
- Learned discriminators via contrastive learning.
- Model-based consistency evaluation.

## Fuzzy-Structured Networks

Inspired by residual pathway priors, we construct a policy or value function of the form

$$
f_\theta(x) =
f_{\text{sym}}(x)
+
\big(1 - w(x)\big)\, f_{\text{res}}(x),
$$

where

$$
w(x) = \operatorname{Agg}_{g \in G} \mu_\psi(x,g).
$$

Here:

- $f_{\text{sym}}$ is a symmetry-constrained equivariant component.
- $f_{\text{res}}$ is an unconstrained residual component.

This structure allows the model to continuously interpolate between structured and flexible representations according to the degree of symmetry. In other words, the framework provides a continuously adjustable unified form between a strictly symmetry-constrained model and an unconstrained model: highly symmetric regions rely mainly on the structural backbone, whereas low-symmetry regions are handled through the asymmetric compensation branch.

## Fuzzy Symmetry-Regularized Learning

We introduce a symmetry consistency loss weighted by fuzzy symmetry:

$$
\mathcal{L}_{sym} =
\mathbb{E}_{(s,a)} \left[
w(s,a) \cdot
\| f(g \cdot x) - \rho(g)f(x) \|^2
\right].
$$

The overall objective becomes

$$
\mathcal{L} = \mathcal{L}_{RL} + \lambda \mathcal{L}_{sym}.
$$

This formulation enforces symmetry only when it is reliable, thereby avoiding the negative effects of incorrect inductive bias and preventing the model from imposing structural priors indiscriminately across the entire state space.

## Discussion

Compared with existing approaches, the proposed framework:

- Generalizes symmetry from discrete representations to continuous representations.
- Extends symmetry exploitation from shallow augmentation-based mechanisms to deep structural embedding in the model.
- Enables adaptive use of structural priors and achieves a more robust balance between expressiveness and generalization.
- Provides stronger robustness in heterogeneous and partially symmetric environments.

## Theoretical Result

**Theorem (Unified Fuzzy Symmetry Error Bound).** Suppose that for a transformation $g \in G$, the fuzzy symmetry membership satisfies

$$
\mu(s,a;g)\ge \alpha,\qquad \forall (s,a)\in\Psi.
$$

Assume further that this implies

$$
|r(s,a)-r(g\!\cdot\! s,g\!\cdot\! a)|\le \bar{\varepsilon}_\alpha,
$$

and

$$
D\!\left(P(\cdot|s,a),\,g^{-1}P(\cdot|g\!\cdot\! s,g\!\cdot\! a)\right)\le \bar{\delta}_\alpha.
$$

Then

$$
\big|Q^*(s,a)-Q^*(g\!\cdot\! s,g\!\cdot\! a)\big|
\le
\frac{\bar{\varepsilon}_\alpha+\gamma \bar{\delta}_\alpha}{1-\gamma}.
$$

This result shows that as symmetry membership increases, the error upper bound induced by structural exploitation decreases accordingly. In fact, this is consistent with the theoretical conclusion of partial symmetry: when reward and transition deviations are small, the error induced by exploiting transformed samples remains bounded and controllable. Our key difference is that we no longer stop at the language of bounded deviations, but reinterpret it as a fuzzy modeling view in which higher continuous membership implies more reliable structural exploitation.

