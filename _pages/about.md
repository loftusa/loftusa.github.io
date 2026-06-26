---
permalink: /
title: ""
excerpt: "About me"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

<script>
function switchTab(tabName) {
  document.querySelectorAll('.tab-bar button').forEach(function(btn) {
    btn.classList.remove('active');
  });
  document.querySelectorAll('.tab-content').forEach(function(div) {
    div.classList.remove('active');
  });
  document.querySelector('.tab-bar button[data-tab="' + tabName + '"]').classList.add('active');
  document.getElementById('tab-' + tabName).classList.add('active');
  history.replaceState(null, null, '#' + tabName);
}
</script>

<style>
.tab-bar {
  display: flex;
  flex-wrap: nowrap;
  gap: 0;
  border-bottom: 2px solid #111;
  margin-bottom: 1.5em;
}
.tab-bar button,
.tab-bar a {
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  padding: 0.5em 1em;
  font-family: "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
  font-variant: small-caps;
  letter-spacing: 0.04em;
  font-size: 0.95em;
  white-space: nowrap;
  color: #666;
  cursor: pointer;
  transition: color 0.15s;
  text-decoration: none;
}
.tab-bar button:hover,
.tab-bar a:hover {
  color: #111;
}
.tab-bar button.active {
  color: #111;
  border-bottom-color: #a00;
}
.tab-content {
  display: none;
}
.tab-content.active {
  display: block;
}
</style>

<div class="tab-bar">
  <button class="active" data-tab="about" onclick="switchTab('about')">About</button>
  <button data-tab="publications" onclick="switchTab('publications')">Publications</button>
  <button data-tab="blog" onclick="switchTab('blog')">Blog</button>
  <a href="/files/cv.pdf" target="_blank">CV ↗</a>
  <a href="/files/submitted_thesis.pdf" target="_blank">Thesis ↗</a>
  <a href="https://youtube.com/playlist?list=PLlP-93ntHnnu-ETNlIfelO9C6T8VrADAh&si=oR1Ool9xr3zrc_NN" target="_blank">Linear Algebra ↗</a>
</div>

<div id="tab-about" class="tab-content active">

<div id="chat-container">
  <hr class="chat-rule" />
  <form id="chat-form">
    <input
      id="chat-input"
      type="text"
      placeholder="Ask me anything about my work…"
      autocomplete="off"
    />
    <button id="chat-send" type="submit" aria-label="Send">→</button>
  </form>
  <div id="chat-messages"></div>
  <p class="chat-privacy">
    <a href="/chat" style="color: #a00; text-decoration: none;">Open in full page ↗</a>
    &nbsp;·&nbsp; conversations are logged
  </p>
</div>

<div markdown="1">

Hi, I'm **Alex Loftus**. I build and run 0-to-1 technical programs at the intersection of AI safety research and engineering — most recently leading multi-agent red-teaming campaigns for OpenAI: recruiting the team, designing the research, building the infrastructure, and running live operations end-to-end. I'm also a textbook author, Kaggle competition winner, and PhD researcher with [David Bau's group](https://baulab.info/), where I study interpretability and evaluation for large language models. Before this I worked as a data scientist, a machine learning engineer, and a master's student in biomedical machine learning at Johns Hopkins University.

I've been fortunate to work with a number of brilliant people over the years. Here are some fun projects which resulted:
 - I designed and led an internal red-teaming campaign for OpenAI. This included writing a campaign proposal, recruiting a 16-person team, and building out infrastructure for the campaign (~17,000 lines of code in two weeks!). The project was successful, and a second campaign is now underway, with double the team size!
 - I'm a co-author of [Agents of Chaos](https://agentsofchaos.baulab.info/), which was covered by [Science](https://www.science.org/content/article/ai-algorithms-can-become-agents-chaos) and WIRED, featured in [Anthropic co-founder Jack Clark's Import AI](https://jack-clark.net/), and went viral on X (millions of views).
 - I helped organize the [New England Mechanistic Interpretability (NEMI) conference](https://www.khoury.northeastern.edu/northeastern-mechanistic-interpretability-workshop-aims-to-make-sense-of-ai-systems/) to help build the interpretability community in the Northeast.
 - I helped out with some work on [Subliminal Learning](https://www.lesswrong.com/posts/m5XzhbZjEuF9uRgGR/it-s-owl-in-the-numbers-token-entanglement-in-subliminal-1), which got featured in a [Welsh labs YouTube video](https://www.youtube.com/watch?v=NUAb6zHXqdI)
 - I led the publication effort for [NDIF](https://www.ndif.us), a large-scale AI interpretability infrastructure project, which led to a paper at ICLR.
 - I've [given talks](https://www.youtube.com/watch?v=V_hcmfdJzF8) for the San Diego Machine Learning meetup group, where I joined a team competing in the [Vesuvius Ink Detection Kaggle Competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection). We won 1st place against 1,249 teams for a competition prize pool of $100,000. Our work was featured in [Scientific American](https://x.com/AlexLoftus19/status/1828158652018237536)!
 - I worked with [Professor Joshua Vogelstein](https://www.neurodata.io) in the Johns Hopkins Biomedical Engineering department on a [pipeline to create graphs from MRI data](https://github.com/neurodata/m2g), which led to a paper in-review at Nature Methods.
 - We developed an open-source project, [Graspologic](https://www.github.com/microsoft/graspologic), which was acquired by Microsoft and used to measure collaboration changes in their workforce during COVID.
 - I made a [linear algebra tutoring series](https://youtube.com/playlist?list=PLlP-93ntHnnu-ETNlIfelO9C6T8VrADAh&si=iYEkHZXhZbq2jrQC) for my friend, which builds up the mathematical machinery of neural networks starting from the absolute foundations: dot product geometry and linear algebra.

I have a number of academic side-interests, including spectral theory, information geometry, the history of science and mathematics, the mechanics of the visual system, constitutional law, various causal relationships between geography and history, and ethics (I am a big fan of Kant, Hume, Ross, and some modern ethicists like Susan Wolf). I am an avid traveler and am (slowly) learning Spanish.

### Misc

I grew up in Seattle, WA. I was a competitive Starcraft 2 player in high school (grandmaster league - competed/won in seattle-area tournaments!). I studied behavioral neuroscience during my undergraduate years, with a philosophy minor focused on ethics. I got interested in math and programming and started a computational neuroscience club, where I taught weekly seminars. I also spent a lot of time partner dancing and playing guitar at open mic nights!

Set up a meeting with me here: [calendly.com/alexloftus2004](https://calendly.com/alexloftus2004/new-meeting-1)

</div>
</div>

<div id="tab-publications" class="tab-content" markdown="1">

## Talks & Publications

- [Agents of Chaos](https://agentsofchaos.baulab.info): arXiv preprint, 2026 (1 citation)
- [NNsight and NDIF: Democratizing Access to Foundation Model Internals](https://openreview.net/forum?id=MxbEiFRf39): Co-first authorship, ICLR 2025. Explore large model internals easily.
- [A Saliency-based Clustering Framework for Identifying Aberrant Predictions](https://arxiv.org/pdf/2311.06454.pdf): NeurIPS Workshop Paper, 2023 - won best poster
- [1st Place Solution - Vesuvius Ink Competition](https://www.youtube.com/watch?v=IWySc8s00P0): Presentation, 2023, for 60 people. Presenting on our winning solution to a \$100,000 Kaggle competition.
- [Hands-On Network Machine Learning with Python](https://a.co/d/4IDM8d3): Textbook, 2025, Cambridge University Press. Second author. 524 pages, 147 figures. (2 citations)
- [A low-resource reliable pipeline to democratize multi-modal connectome estimation and analysis](https://www.biorxiv.org/content/10.1101/2021.11.01.466686v1): Paper, 2022, Nature Methods, under review. Second author, wrote most of the infrastructure for the codebase. (79 citations)
- [ICML Conference Highlights](https://www.youtube.com/watch?v=V_hcmfdJzF8): Talk about new machine learning techniques in drug discovery and medicine presented at ICML
- [Working with LLMs](https://lu.ma/aisd1): Talk, 2023, for 100 people at the AIML San Diego meetup
- [Role of CAMKII in Associative Conditioning and GLR-1 Expression in C. Elegans](https://imgur.com/a/f2TxUt9): Poster, presented at Society for Neuroscience, 2017, Washington, DC. Later author, conducted most of the later experiments.
- [Effects of an unc-43 (CaMKII) Gene Deletion on Short-Term Memory for Associative Conditioning in C. elegans](): Talk, presented at Psychfest, 2017, Bellingham, WA.

</div>

<div id="tab-blog" class="tab-content" markdown="1">

## Recent Posts

{% for post in site.posts limit:8 %}
<div style="margin-bottom: 1em;">
  <span style="color: #666; font-size: 0.85em;">{{ post.date | date: "%B %d, %Y" }}</span><br>
  <a href="{{ post.url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  {% if post.excerpt %}<br><span style="color: #444; font-size: 0.9em;">{{ post.excerpt | strip_html | truncate: 160 }}</span>{% endif %}
</div>
{% endfor %}

<a href="/year-archive/" style="font-variant: small-caps; letter-spacing: 0.04em;">View all posts →</a>

</div>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<script src="/assets/js/chat.js"></script>

<script>
// On load, check hash for tab state
(function() {
  var hash = window.location.hash.replace('#', '');
  if (['about', 'publications', 'blog'].indexOf(hash) !== -1) {
    switchTab(hash);
  }
})();
</script>
