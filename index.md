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
        {% for item in unpinned limit: 10 %}
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
  </div>

  <div class="pure-u-1 pure-u-md-2-5">

    <h2 id="people-header">People</h2>
    <div id="people" class="pure-g">
      {% assign members = site.data.people | filter_alumni: nil | sort_people: 'Professor, PhD, Visiting, Researcher, Undergraduate Student', false %}
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
      {% assign alumni = site.data.people | filter_alumni: true | sort_people: 'PhD, Postdoctoral, Scientist' %}
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
                  &rdca; {{person[1].next}}
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
