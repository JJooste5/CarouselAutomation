# Carousel Renderer

## Setup
1. Install dependencies.
   ~~~bash
   pip install -r requirements.txt
   ~~~
2. Put background images in the `images` folder.
3. Add your content in `content.csv` or `content.xlsx`.

## How to use
- Add rows to `content.csv`. Keep the header: `post_id,heading,subheading,body`.
- It also works with Excel files.
- Run:
  ~~~bash
  python render.py
  ~~~
- Images are saved to the `out` folder.

## How to customise the images
- Most of `layouts.json` is now redundant.
- Text placement is chosen automatically.
- The script looks for the darkest area, with some randomness.
- It will not place text in the same location twice in a row.

## How it works (brief)
- Picks a random background from `images`.
- Moves used files to `images_used`.
- Analyses brightness by region.
- Chooses the darkest region with randomness.
- Avoids repeating the last text location.
- Renders title and content slides.
- Writes JPEGs to `out`.

---

## Example `content.csv`
~~~csv
post_id,heading,subheading,body
1,"5 ways to save",,
1,"Track spending","See where money goes","Review your expenses weekly."
1,"Cut subscriptions","Remove what you do not use","Cancel unused services to save money."
1,"Cook at home","Lower daily costs","Plan simple meals for the week."
1,"Buy in bulk","Reduce unit prices","Stock essentials when prices are low."
1,"Set savings goal","Stay on track","Pick a clear amount and date."
~~~

---

## Prompt to turn any text into the format we need in the CSV
