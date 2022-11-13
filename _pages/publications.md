---
layout: page
permalink: /publications/
title: publications
description: Published and working papers. Note that PDFs can be downloaded for noncommercial, information purposes only. They may not be reposted without permission.
years: [2022, 2020, 2019, 2018]
nav: true
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>