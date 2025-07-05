---
layout: page
home: true
---
<p id="mission">
  We are rethinking the next generation of <strong>AI-centric computing systems</strong>, 
  by developing DNN compression methods, efficient hardware architectures, and end-to-end AI systems optimization.
</p>

<h2 id="news-header">News</h2>
  <div id="pinned">
    {% assign pinned = site.data.news | where: "pinned", true %}
    {{pinned[0].desc | markdownify}}
  </div>
  <div id="news">
      <div id="news-items">
        {% assign unpinned = site.data.news | where_exp: "item", "item.date" %}
        {% for item in unpinned limit: 200 %}
          <div class="item">
            <p class="date">{{item.date}}</p>
            {{item.desc | markdownify}}
          </div>
        {% endfor %}
      </div>
    </div>

<div id="home" class="pure-g">
  <div id="themes" class="pure-u-1 pure-u-md-3-5">
    <h2>Research Themes</h2>
    {% for theme in site.data.research_themes %}
      <div id="theme-{{theme.key}}" class="theme" data-url="{{theme.url}}" data-people="{{theme.people}}">
        <!-- <img src="/themes/{{theme.key}}.png" style="max-width: 100%; height: auto; display: block; margin-top: 0;"> -->
          <div style="padding-top: 0px; border-radius: 5px; margin-bottom: 0px;">
            <div class="content">
              <h3>{{theme.name}}</h3>
              {{theme.desc | markdownify}}
            </div>
          </div>
      </div>
    {% endfor %}

  <!-- --- Sponsors pane ---------------------------------------------------- -->
  <details id="sponsors-pane">
    <!-- ▸ arrow rotates open/closed via CSS -->
    <summary>
      <span class="toggle-arrow">▸</span>
      Our research is generously supported by amazing collaborators and sponsors, listed here.
    </summary>

    <!-- Logos grid — manual list with per-logo width -------------------------------->
    <div class="sponsor-logos pure-g">
      <div class="logo-cell pure-u-1-2 pure-u-sm-1-3 pure-u-md-1-4">
        <img src="/imgs/sponsors/nsf.png"     alt="NSF"     width="120">
      </div>
      <div class="logo-cell pure-u-1-2 pure-u-sm-1-3 pure-u-md-1-4">
        <img src="/imgs/sponsors/intel.png"   alt="Intel"   width="100">
      </div>
      <div class="logo-cell pure-u-1-2 pure-u-sm-1-3 pure-u-md-1-4">
        <img src="/imgs/sponsors/altera.png"  alt="Altera"  width="95">
      </div>
      <div class="logo-cell pure-u-1-2 pure-u-sm-1-3 pure-u-md-1-4">
        <img src="/imgs/sponsors/lg.png"      alt="LG"      width="85">
      </div>
      <div class="logo-cell pure-u-1-2 pure-u-sm-1-3 pure-u-md-1-4">
        <img src="/imgs/sponsors/meta.png"    alt="Meta"    width="95">
      </div>
      <div class="logo-cell pure-u-1-2 pure-u-sm-1-3 pure-u-md-1-4">
        <img src="/imgs/sponsors/nvidia.png"  alt="NVIDIA"  width="105">
      </div>
      <div class="logo-cell pure-u-1-2 pure-u-sm-1-3 pure-u-md-1-4">
        <img src="/imgs/sponsors/tcs.png"     alt="TCS"     width="90">
      </div>
      <div class="logo-cell pure-u-1-2 pure-u-sm-1-3 pure-u-md-1-4">
        <img src="/imgs/sponsors/xilinx.svg"     alt="Xilinx"     width="90">
      </div>
    </div>

    <!--  Explanatory text --------------------------------------------------- -->
    <div class="sponsor-text">
      <p>
        <!-- -->
      </p>
    </div>
  </details>
  <!-- --- end Sponsors pane ----------------------------------------------- -->
  </div>

  <div class="pure-u-1 pure-u-md-2-5">

    <h2 id="people-header">People</h2>
    <div id="people" class="pure-g">
      {% assign members = site.data.people | filter_alumni: nil | filter_collab: nil | sort_people: 'Professor, PhD, Visiting, Researcher, Undergraduate Student', false %}
      {% for person in members %}
        {% unless person[1].not_current %}
          <div id="{{person[0]}}" class="person pure-u-1-4">
            <a href="{{person[1].url}}">
              <p class="headshot"><img src="/imgs/people/{{person[0]}}.jpg" alt="" /></p>
              <p class="name">{{person[1].name}}</p>
              <p class="title">{{person[1].title}}</p>
            </a>
          </div>
        {% endunless %}
      {% endfor %}
    </div>
    <h3 id="alumni-header">Alumni</h3>
    <ul id="alumni" class="pure-g">
      {% assign alumni = site.data.people | filter_alumni: true | filter_collab: nil | sort_people: 'PhD, Postdoctoral, Scientist' %}
      {% for person in alumni  %}
        <li id="{{person[0]}}" class="person pure-u-1-2">
          <a href="{{person[1].url}}">
            <span class="headshot">
              <img src="/imgs/people/{{person[0]}}.jpg" alt="" />
            </span>
            <span>
              <span class="name">{{person[1].name}}</span><br> 
              <span class="title">
                {{person[1].title}}
                {% if person[1].next %}
                  <br>
                  ⤷ {{person[1].next}}
                {% endif %}
              </span>              
            </span>
          </a>
        </li>
      {% endfor %}
    </ul>
  </div>
</div>

<div>
  <h5>Design from <a href="https://vis.csail.mit.edu/">MIT Visualization Group</a></h5>
</div>
