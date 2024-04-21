---
layout: post
title:  "Is it worth buying a house in Boston in 2024?"
date: 2024-04-07
mathjax: true
---

My girlfriend and I are about to move to Boston, and so I figured it would be fun (and useful) to play around with some neighborhood-level data to see what the market looks like. The ideal goal is to find a place bikeable from Northeastern University, where I'll be studying. 

It turns out that Zillow has a few [reasonably decent datasets](https://www.zillow.com/research/data/) that give a birds-eye view of the market. Our goal is to find a 2-bedroom home, one of which we'd turn into an office, and the other into a bedroom. Zillow has a median home values for two-bedroom houses by neighborhood, so let's start there.

<div class="imgcap">
<img src="/images/boston-housing/neighborhood-zillow-dataset.png">
<div class="thecap">Zillow dataset</a>
</div>

The dataset is a bit messy, but it's a good starting point. We can clean it up a bit and plot the data to get a sense of the distribution of home values in Boston. 

```python