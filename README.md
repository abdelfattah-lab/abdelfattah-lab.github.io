# Abdelfattah Research Group Website

## Dev Instructions

- Install Jekyll ([instructions](https://jekyllrb.com/docs/installation/macos/))
- Install bundler
```
gem install jekyll bundler
```
- Install dependencies
```
bundle install
```
- Serve your content
```
bundle exec jekyll serve
```
- Check localhost:4000 for your content
  

## Content Instructions

- Add news by adding a new entry to `_data/news.yml`.
- Add blogs by adding a markdown file to `_blogs`.
- Add publications by adding a markdown file to `_pubs`, and a teaser image with the same name in `imgs/teasers`. You can also optionally add a thumbnail in `imgs/thumbs`, if empty, the teaser will be used as a thumbnail.
