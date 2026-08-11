---
layout: post
title: "About"
author: "Paulina Moskwa"
permalink: /about/
---

<center>
Hello, I'm Paulina Moskwa 👋
</center>

<br>

<br>

## Can you beat an AI at Super Mario Bros?

Play Super Mario Bros level 1 and try to win against the ghost. The ghost racing you is a PPO agent I trained as a hands-on part for [Part II](../2026-08-08/rl-pt2) of my Reinforcement Learning saga, playing live rather than from a recording: all 1.7 million weights are running on this page, so no two runs are the same. Press play and you start together. It reaches the flag in about 18 seconds, so hold **B** to keep up. It has no idea you are there.

<div class="race"
     data-rom="{{ site.baseurl }}/assets/data/smb.nes"
     data-policy="{{ site.baseurl }}/assets/data/policy.bin"
     data-policy-meta="{{ site.baseurl }}/assets/data/policy.json"></div>

<script src="{{ site.baseurl }}/assets/js/jsnes.js"></script>
<script src="{{ site.baseurl }}/assets/js/mario-policy.js"></script>
<script src="{{ site.baseurl }}/assets/js/mario-race.js"></script>
