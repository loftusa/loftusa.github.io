---
title:  "Research directions in interpretability"
date:   2024-04-28
permalink: /posts/2024/04/research_in_interp/
mathjax: true
---

# Research Directions in Interpretability

I am about to start a PhD at Northeastern University with [David Bau's lab](https://baulab.info/), and the beginning of a PhD is a great time to think carefully about what areas I can maximally have an impact in.

So, here are some collected thoughts on research directions in interpretability I think would be impactful. A good starting point is to begin with application areas and end goals, then backpropagate into research that could potentially be fruitful for those areas and end goals. Therefore I will begin with downstream research areas I'm interested in, starting with general philosophy, and then move into more specific project ideas.

## Philosophy
In general, it seems clear that large foundation models learn useful things about the world. This is true for language, but I think vision and multimodal (and video, which I think will develop over the coming years!) is just as important, and possibly often easier from a research perspective because visual information is just so much more semantically rich from a (human) cognitive perspective than language information. Using prompts to access that information feels to me like an incredibly coarse method of information acquisition, and I think I want one of the central pushes of my PhD to be towards finding richer ways of accessing that information.

## Motivations
This is useful for a number of reasons not necessarily transparent to computer scientists with a background entirely in deep learning. I'm coming from a biomedical engineering / statistical machine learning / data science background. In that world, a big source of hesitation in using deep learning - besides the obvious bulkiness involved in training a big model over running `sklearn.decomposition.PCA.fit_transform()` and the lack of usefulness in low-data regimes - is that it's much more difficult to find interpretable features. I'm coming out of a very computational drug discovery startup, for instance. In that world, language models trained on RNA are starting to make their appearance (for instance, [profluent](https://www.profluent.bio/) is active in that space, and my company has started to play with [SpliceBERT](https://github.com/chenkenbio/SpliceBERT?tab=readme-ov-file)). Biologists and chemists can't access the information in these models from prompting, since the whole point is that the model has learned something rich about how RNA sequences are organized in a language that they don't know. However, these models are clearly learning interesting and nontrivial information about biology, as evidenced by the downstream tasks (branchpoint prediction, splice site prediction, variant effect inference, nucleotide clustering...) they were able to get surprisingly good results on. A microscope to peer at this information directly would be critically useful for scientists.

In computer vision, my [girlfriend's](https://ainatersol.github.io/) company is another great example. They are using straightforward classification architectures - think EfficientNets and ResNets - to classify veterinary X-rays. One of their big challenges, particularly in moving into human data, is that doctors need to be able to trust the model. In order to trust the model, there needs to be a great deal of transparency in how the model is making its decisions. They've been able to take advantage of pixel attribution methods, e.g., SHAP, LIME, and GradCAM, to get some insight. But these methods don't tell the whole story.

Furthermore, sometimes (e.g., GradCAM, according to her) don't have easily-accessible implementations that can hook into existing codebases. For cases where the problem is wild-west levels of implementation for useful techniques, it is useful to collect all of them into a single codebase with a methods paper attached. Ideally this codebase would become popular, but trying to get that codebase PR'd into a major library in something like huggingface or something torch-adjacent might have more impact. This worked well during my master's in the graph world for [graspologic](https://github.com/microsoft/graspologic), during which time a labmate PR'd his work into [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multiscale_graphcorr.html#r882c1b4c2283-2). 

In general, here are a few areas in which better interpretability would be very useful:
- customer-facing medical applications
- legal applications
- self-driving
- drug discovery 
- all of science
- people who want to know why their model isn't training

Basically, anywhere in which a central push is to *learn something* rather than *do something*, and any application area in which there's a high cost for mistakes.

**The biggest, most useful big-picture goal would be: can I create a set of tools which let users tell the extent to which a model is hallucinating, based on what's happening in its activation space?**

## Interests
With the above impact motivations in mind, there are also things that just *strike my fancy*, and which I think are interesting and fun to think about. Good projects, in my opinion, will also contain elements of this. A somewhat representative but not at all exhaustive list:

- Constrained optimization methods which create model edits that respect the data manifold. Riemannian optimization methods, for instance, or using spatial derivatives to move around on manifolds. Methods which take advantage of the tangent space and local linearity. 
- In the same vein, activation steering and vector arithmetic in the latent space. Anything that takes advantage of the geometric properties of algebraic spaces to perform some useful and unintuitive task.
- The whole set of LoRA/DoRA/ReLoRA etc. PEFT methods are interesting largely because I have a geometric intuition for low-rank updates -- e.g., making linear updates which are constrained to a low-dimensional hyperplane on the original subspace. 
- I have a side-interest in explorable explanations and interactive documents, e.g., [distill.pub](distill.pub), [betterexplained](www.betterexplained.com), anything by [brett victor](https://worrydream.com/ExplorableExplanations/), anything by [flowingdata](https://flowingdata.com/made-by-flowingdata/).
- Petar Veličković and friends have been working on [mathematical frameworks](https://cats.for.ai/) for [categorical deep learning](https://www.youtube.com/watch?v=QEL4djGIliw)  recently, extending geometric deep learning to use the tools of category theory to attempt to build a language of architectures. It looks interesting; I don't know at first glance how it would be useful in interpretability.
- My areas of mathematical interest include causal inference (e.g., judea pearl's work), high-dimensional statistical inference, information theory, information geometry, and differential geometry. My intuition is that all of these areas contain useful ideas for interpretability research.

## Research Directions
With all of the above in mind, here is an equally non-exhaustive list of directions and/or projects. Some of the below is more fleshed out than others; I basically just threw something down whenever I had an idea. My hope is that ideas will be quickly whittled down when they aren't feasible and/or don't have clear impact and/or are too vague and ill-defined, and useful ideas can be expanded into concrete projects with clear goals.

- Starting out by simply helping out labmates with their current work seems like a great way to get myself involved quickly with existing research. It seems reasonable to have a first-year goal be to at least touch everybody else's work in the lab at least once, and contribute something useful for them.
- Papers which explore the same idea, but under a lot of different model architectures, implementation details, etc. Finding a set of golden retriever ear detecting neurons for a ConvNext trained on exactly ImageNet using exactly AdamW at exactly a learning rate of 1e-4 is much more interesting if it's also true for vision transformers and whatever else, using different optimizers and hyperparameters. Which results generalize most strongly?
    - Example: efforts to transfer interp results in GANs to diffusion models, efforts to transfer results in transformers to mamba...
    - addendum to this: to what extent do the same concepts apply when we change modalities? Can we discover ideas in interpretability in text, and do they also apply to vision, video, audio...
- Volumetric and video models are cool and largely underexplored right now, primarily (as far as I can tell) because of compute constraints. Possibly soon, particularly with video, this will be interesting to explore (although moving from $O(n^2)$ to $O(n^3)$ is a big jump). A paper which applies existing image-interpretability techniques to video models could be fun to write.
- For LLMs: Can we find or create a 'factuality' or 'confidence' direction in activation space, in a way that lets us add that information into model responses? 
    - for instance, you ask why some code doesn't work, and the LLM responds, and above the response there is something that says "the model is 60% sure that this answer is correct".
    - The loss function for this particular example would involve measuring the empirical distribution of the actual amount of times it was correct, and then looking at the difference.
- What kinds of interpretability questions can we ask about the kinds of updates PEFT methods are constrained to make under the low-rank regime? Is the difference between a low rank update and a full rank update just a quantitative change, or is there a qualitative difference in the kinds of updates that can be made? 
    - In the same vein: What are we actually doing geometrically to weight matrices by adding low-rank information into them? I suppose (via geometric intuition rather than formal proof, so possibly wrong) that higher-magnitude low-rank updates causes weight matrices to be closer to low rank themselves. Do we care about this?
- Something that would be very useful if solved is "densifying" context windows. Say you're at the edge of a very large context window. Is there a way to map the set of token embeddings you're conditioning on to a lower rank matrix, or to a single information-rich vector, or just anything more compressed, so that you can still condition on the context without running out of space? This is not necessarily interpretability research, but it is interesting.
- the attention operation, viewed as a linear transformation, has access to (e.g., not in the nullspace) at most a hyperplane of dimension `m` when operating on an embedding matrix where `m` is the number of tokens.
    - Multi-headed attention takes subsets of dimensions, and moves the `m` vectors around on the hyperplane within those subsets. Then they are concatenated.
    - is this linear? The softmax in the attention matrix isn't, but can you pull out all the softmaxes?
    - If the above is true, can you shuffle things around algebraically so that the splitting, attention-updating, and concatenating all happen in a single matrix multiplication (block attention matrices * head concatenations), since it's all one linear transformation?
    - if you can do the above, is there a technique that makes doing it this way faster, since then you're taking advantage of optimized matrix multiplication on the gpu as much as possible?
- My competitive advantage is a geometric intuition for linear algebra and being able to take ideas from the [graph stats textbook](https://alex-loftus.com/files/textbook.pdf). How many ideas that I know well from textbook could possibly be useful in deep learning for interpretability research?
    - MASE and OMNI embeddings - joint representation learning for multiple graphs (project nodes from different graphs into the same subspace) - possibly useful for comparing attention matrices
    - I wonder if the spectral embedding of the Laplacians of attention matrices tells us anything
    - graph matching - maybe not useful? I could potentially see an experiment where we look at whether attention matrices become harder or easier to graph-match (e.g., the loss function gets bigger or smaller) in deeper layers.
    - random graph models and their properties, e.g., erdos-renyi, SBMs, RDPGs, etc - probably not useful here. Maybe as a statistical framework. Not sure. Don't see it right now.
    - Network summary statistics: network density, clustering coefficient, shortest path length, average node degree, etc: would have to binarize attention matrices with a weight threshold for most of these, but looking through these statistics across attention matrices might tell us something interesting. For example, do attention matrices tend to be more or less dense than random graphs? 
    - signal subgraph estimation and anomaly detection in timeseries of networks: There is an interesting interpretability question here: which signal subgraph, e.g., subsets of token-nodes, tend to change the most over time?
    - Scree plots and automatic elbow selection: Could also be interesting. This helps answer the question: how does the latent dimensionality of attention matrices change across heads and layers? is it fixed, or does it change?
    - Community detection: how do nodes in different attention matrices cluster? Do they tend to form the same clusters across heads, or different clusters? what about across layers?
    - vertex nomination: find communities for community-less nodes when we know the communities of the other nodes. Don't think this is super interesting here but maybe I'm wrong.
    - out-of-sample embedding: Estimate the embedding for a new node given the embeddings of existing nodes. You need to have already decomposed a graph with SVD in order to be able to do this. I suppose you could measure the extent to which you can estimate what the embedding for a token should be given the embeddings of other tokens, but would have to think about downstream use cases for this.
    - two-sample testing: test whether two attention matrices are drawn from the same distribution. Could be interesting for comparing attention matrices across heads and layers.
- [Mixture of Depths](https://arxiv.org/abs/2404.02258) is an interesting paper which commits to the idea: "why apply every transformer block to every token when presumably some tokens need more computation than others?". An interesting interpretability follow-up to this would be: Which tokens ended up needing more compute than others, and what does that tell us about what is easy and difficult to compute for a language model?
- It feels to me like PEFT methods would work better if the weight matrices they are updating are low-rank. There is a more rigorous way to think about this: how much of the mass of singular values is concentrated in the top singular vectors? You can think of this sort of the entropy of the singular values. If the entropy is low, e.g., the magnitude of all the singular values is concentrated in the highest ones, then PEFT methods should work better, because the weight matrices are a closer approximation to low rank.


## Whittling it Down
So, after discussing with David, there are a few potential ways these directions are harmonious with existing work:

### Erasing
Low rank updates have overlap with **unlearning**. There are two ways in which you can get a model to unlearn through gradients updates or whatever.  

First, you can use a model's knowledge against itself. So, you ask a model what it knows about X, save the resulting activation vectors, and fine-tune against those vectors. This
Second, you can localize the model's knowledge about X using probes or causal mediation analysis or whatever, and then once you've localized, change the model at that location so that the probe can no longer see X. That's what LEACE does, for instance.

Rohit has a paper in review in this general area called sliders. The idea is that you can modify outputs of diffusion models by using an unlearning loss function to dynamically control stuff like age; erase concepts like 'youngness', for instance, so that it always makes people old or whatever. This is already published. The new idea is, instaed of using the erasing loss to permanently change a model, use the erasing loss to train a lora; and then you can just slide the lora up and down.

There is also a visiting HCI student named Imca Grabe, who is training a bunch of sliders. One idea here is: you have a left-hand vector and a right-hand vector, right? What if they were from different sliders? Then you're composing sliders together and maybe they can do interesting composed things. 

### Reverse-Engineering PEFT updates
Another line of research is in **reverse-engineering**. The idea is: Say you edit a concept in a model with ROME or whatever. You do this with a rank-one change.

Anybody can figure out that you've changed the model, that's easy - you just take the difference in weights and see that it's nonzero. You can also look at the rank of the difference matrix and see whether the change is just a low-rank update or something major.

What you can't do is figure out what's actually changed. So the question is: How do you figure it out?

Koyena is working on this right now, so I'd reach out to her and brainstorm.

### Function vectors
This is related to the context windows densification idea. There is also an industry application here.

So, the idea is essentially compression. Say you have a huge prompt. Like, you're a company and you've done a bunch of prompt engineering and now you have this gargantuan thing that's half the context window. It turns out that, under the hood, the model essentially represents this whole thing with a single vector. 

So, you can take this vector out, jam it back in later, and voila, the model will do the same thing it was going to do originally; you don't need all these examples.

Eric is working on this. He's mostly been looking at it for theoretical reasons, but there's room for turning it into an industrial-strength method that people would want to use. So, for instance, you have these chipmaking companies that need knowledge in these old programming languages that people don't know how to use anymore. And copilot doesn't work either because it wasn't trained on these languages. So they have these super long prompts that cost lots of money (in token cost) to run inference with.

Instead of these companies writing in these super wrong prompts, we can basically compress their big prompts into a single vector, and then they can just use that vector to get the same results for cost savings. There are various techniques for compressing context instructions, we'd do a literature review. But here we'd try to beat a benchmark: can we use function vectors to compress in ways that beat the current state of the art?

### Mechanisms of pause tokens
Related to the question about Mixture of Depths, as well as the question about ranks of attention matrices when viewed as linear transformations, is the question: what are the mechanisms of pause tokens? Why do they work and what do they do? What kinds of extra computations become available when we add pause tokens?

A guy named Rihanna Brinkman is doing some stuff related to this. There's also a student in Chris Manning's lab at Stanford, Shikar, who is doing a research project around this. Learning why pause tokens work seems potentially fruitful; it seems like the type of thing that, if figured out, would open more research doors.

### Linear Relations and graphs
This has to do with the linear relations paper.

Take the phrase "The Eiffel Tower is located in the city of Paris". Run this through a forward pass and get the activations. Take the hidden state for Eiffel Tower in an early layer, and for Paris in a later layer. What is the relationship from one token to the other?

Well, take the Jacobian of the thing and you have this locally linear mapping between the two. But the cool thing is that then, using this Jacobian, you can, say, replace "eiffel tower" with "Great Wall", and the later hidden state will now output "China" -- with the same Jacobian, using the same linear transformation matrix!

So this tells us that transformers are not just locally linear, but they are linear over a pretty broad area.

Evan Hernandez at MIT wrote the linear relation embeddings paper, but he just graduated.

## Blog Post Ideas
- How much longer would it take to train a transformer if you expressed everything in residual stream format?

## More Ideas
- Not interpretability, but initialization seems weirdly underexplored. Is it possible to distil a transformer down to an MLP, and then reverse-engineer what the nearest easy initialization is? If someone could figure out how to encode the attention mechanism inside an MLP, for instance, that would be a massive breakthrough.
- what if instead of next word prediction, you predicted the embedding for the next word, using the loss of some embedding distance metric instead of cross entropy?
	- Could test a bunch of different distance metrics, using cross-entropy as a baseline, and see if any of them is better
- Nobody seems to care about proper model initialization because it isn't flashy. Is correct model initialization solved for every type of common layer? Is there a guide somewhere that says "for this type of layer, use this type of initialization" for all common layer types? If not, could I write a review paper?
    - by 'correct model initialization', I mean: 'model initialization such that, when data with mean 0 and variance 1 is passed through the untrained model, mean and variance statistically stay 0 and 1 respectively at every layer'
    - I ask this because many implementations I've seen only change the initialization for linear layers, even relatively well-known models (dinov2, for instance)
- Anthropic recently came out with some great [sparse autoencoders](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) work which seems to open a lot of doors. Near the end they track activations for the features which combine to result in a particular output, with examples being a bunch of sentences and how strongly these sparse features they created (from the SAE) fire for each of the tokens in the sentences.
    - I wonder if we could use this to figure out what in the training data created that feature in the first place. Could you use the strength of activations in training tokens to create a 'map' of where in the training data the model is using to make its decisions?
    - This could be very broadly useful: companies training models could use this to get much more specific with what their model ends up learning, by ablating sections of training data they don't want (based on this map).
- Method: Using only activations as input, train a network to predict what the output will be. Which layers contain the most output-relevant enformation?
- Method: Use SAE features to build better search pipelines. A k-nearest neighbor search on an SAE feature space might be much better than one on a polysemantic feature space.
- Method: initialization technique --> warm up for some number of epochs on a loss function that encourages a weight space such that mean is 0, variance is 1 for activations. Then replace with the real loss function.
- Benchmark / Dataset: Pieces of code followed by what happens if you execute the code. Rather than next-token prediction, it'd be: predict terminal output given code input. Problems:
	- What do you predict? You're not predicting one token at a time, so it would need to be something other than a probability distribution over tokens. Predicting every token of output seems computationally intractable, because of combinatorial explosion when you try to predict n tokens at once. Maybe there's a way I don't know about.
- The extent to which linear updating methods work depends on the curvature of the concept-space corresponding to that feature. Quantifying the curvature using methods outlined [here](https://www.youtube.com/watch?v=UYiAlYlSwBo) should tell you the extent to which editing methods work.
- Crazy idea: what if I recreated this paper, but for Starcraft 2, and AlphaStar? https://arxiv.org/abs/2310.16410
- An algebra of editing methods: take n model-editing techniques. What algebraic properties do they have? e.g., are they commutative? associative? invertible? transitive?
- One of the most famous studies of all time in neuroscience was the discovery of parrahipocampal place cells. Rats were led through a maze, and their location in the maze was reconstructable from place cell firing patterns. Can I do an equivalent study on video transformers? e.g., are there neurons or activation directions responsible for where objects in the video are located?
	- proof of concept for this would be: changing one of these activation directions or whatever and showing that it moves objects around in the video that gets generated (this would work for image transformers as well).
	- the end goal of this system would be precision editing. Change activations to move objects in a picture around (in a diffusion model). Change activations to change where objects are predicted to be via bounding boxes (in an image transformer trained on ImageNet).
- Fuzzy idea for another way to get function vectors. I was using Claude to rewrite text. At the beginning of the session, I said something like "I want to improve the sentence structure in this text, ...". Then I pasted in some text and it improved the structure. Then I pasted in more text and it kept improving the structure. I wonder if you could marginalize out all the times that it improved structure to get at the instruction itself, across all the responses. 
- To what extent is [this video](https://www.youtube.com/watch?v=eMlx5fFNoYc) true? e.g., could I try to find an attention head whose query/key matrix searches for adjectives behind nouns, and then updates the vectors for the nouns according to those adjectives? (If not a full paper, this would make for a great blog post!)
- Feature importance technique: Quantifying which features in an LLM are the most monosemantic is likely a good feature importance metric, because features with their own orthogonal directions contain the least interference with other features when matrix-multiplied (and thus, are more 'valuable real-estate')
- Creating a Priviliged Basis: Can you train an orthogonal procrustes iteratively at each layer of the residual stream to map everything to the same basis? (this thought may be half-formed because I haven't thought very deeply about what 'privileged basis' means yet)
