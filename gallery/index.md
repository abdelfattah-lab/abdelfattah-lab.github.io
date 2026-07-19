---
title: Gallery
layout: page
---
<div id="gallery">
  <h1 class="title">Gallery</h1>
  <p class="gallery-intro">Moments from the Abdelfattah Research Group.</p>

  {%- comment -%} Manifest is pre-sorted newest-first, undated photos last. {%- endcomment -%}
  {% assign photos = site.data.gallery %}
  <div id="photo-grid">
    {% for photo in photos %}
      <button type="button" class="tile"
              data-index="{{ forloop.index0 }}"
              data-full="{{ '/imgs/gallery/' | append: photo.file | relative_url }}"
              data-date="{% if photo.date and photo.date != '' %}{{ photo.date | date: '%B %-d, %Y' }}{% endif %}"
              {% if photo.caption %}data-caption="{{ photo.caption }}"{% endif %}
              aria-label="Open photo{% if photo.caption %}: {{ photo.caption }}{% endif %}">
        <img src="{{ '/imgs/gallery/thumbs/' | append: photo.file | relative_url }}"
             loading="lazy" alt="{{ photo.caption | default: 'Group photo' }}" />
      </button>
    {% endfor %}
  </div>

  {% if photos.size == 0 %}
    <p class="gallery-empty">No photos yet. Run <code>python3 update_gallery.py</code> to add some.</p>
  {% endif %}
</div>

<!-- Lightbox overlay -->
<div id="lightbox" class="lightbox" aria-hidden="true" role="dialog" aria-modal="true" aria-label="Photo viewer">
  <button class="lb-close" aria-label="Close (Esc)">&times;</button>
  <button class="lb-nav lb-prev" aria-label="Previous photo (←)">&#8249;</button>
  <figure class="lb-stage">
    <img class="lb-img" src="" alt="" />
    <figcaption class="lb-caption"></figcaption>
  </figure>
  <button class="lb-nav lb-next" aria-label="Next photo (→)">&#8250;</button>
</div>

<script>
(function () {
  var tiles = Array.prototype.slice.call(document.querySelectorAll('#photo-grid .tile'));
  if (!tiles.length) return;

  var lb        = document.getElementById('lightbox');
  var lbImg     = lb.querySelector('.lb-img');
  var lbCap     = lb.querySelector('.lb-caption');
  var btnClose  = lb.querySelector('.lb-close');
  var btnPrev   = lb.querySelector('.lb-prev');
  var btnNext   = lb.querySelector('.lb-next');
  var current   = -1;

  function show(i) {
    if (i < 0) i = tiles.length - 1;
    if (i >= tiles.length) i = 0;
    current = i;
    var tile = tiles[i];
    lbImg.src = tile.getAttribute('data-full');
    lbImg.alt = tile.getAttribute('data-caption') || 'Group photo';
    var date = tile.getAttribute('data-date') || '';
    var cap  = tile.getAttribute('data-caption') || '';
    lbCap.textContent = cap ? (cap + ' — ' + date) : date;

    // Preload neighbours for snappy browsing.
    [i - 1, i + 1].forEach(function (n) {
      var t = tiles[(n + tiles.length) % tiles.length];
      if (t) { var p = new Image(); p.src = t.getAttribute('data-full'); }
    });
  }

  function open(i) {
    show(i);
    lb.classList.add('open');
    lb.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  }

  function close() {
    lb.classList.remove('open');
    lb.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    lbImg.src = '';
    // Return focus to the tile we came from — resume browsing where we left off.
    if (current >= 0 && tiles[current]) tiles[current].focus();
  }

  tiles.forEach(function (tile) {
    tile.addEventListener('click', function () {
      open(parseInt(tile.getAttribute('data-index'), 10));
    });
  });

  btnClose.addEventListener('click', close);
  btnPrev.addEventListener('click', function (e) { e.stopPropagation(); show(current - 1); });
  btnNext.addEventListener('click', function (e) { e.stopPropagation(); show(current + 1); });

  // Click the dark backdrop (but not the image / buttons) to go back.
  lb.addEventListener('click', function (e) {
    if (e.target === lb || e.target.classList.contains('lb-stage')) close();
  });

  document.addEventListener('keydown', function (e) {
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowLeft') show(current - 1);
    else if (e.key === 'ArrowRight') show(current + 1);
  });
})();
</script>
