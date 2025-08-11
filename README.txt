Quick start

1) Use content_template.csv as a model.
   One row per slide. Vary number of slides per post freely.
   Columns:
     post_id   : identifier for the carouse. This allows us to make multiple carousels all at once. Post 1, post 2. i.e. we can have multiple slides per carousel, and multiple carousels.
     slide     : 1-based slide index
     slide_type: "title" or "content"
     layout    : exact name from layouts.json, or leave empty to randomize within type
     title_line: for the first slide (single line)
     heading   : for content slides
     subheading: for content slides
     body      : for content slides. Use \n for line breaks.

6) Run:
   python render.py --content content_template.csv --layouts layouts.json --config config.json

Notes
- Size is 1080x1080.
- Each slide picks a random image from images/. The script moves it into images_used/ so it will not be reused next time.
- If images/ is empty, the canvas background color shows.
- To reset image pool, move files back from images_used/ to images/.
- To force a specific layout, fill the "layout" column. Otherwise the script will pick one at random within the slide type.
- Tweak sizes and boxes in layouts.json to taste.
