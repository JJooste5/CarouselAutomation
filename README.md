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
- Each row is it's own slide.
- Post_id defines which of the slides are for which post.
- The first slide with a given post id will be assumed to be the title slide. So, subheading and body will be ignored for this one.
- The formatting of text will be automatically decided.
- It also works with Excel files.
- Run:
  ~~~bash
  python render.py
  ~~~
- Images are saved to the `out` folder.

## How to customise the images
- There are a whole load of paramters at the top of render.py
- Most of `layouts.json` is now redundant and doesn't actually define a lot of the formatting.
- Text placement is chosen automatically.
- The script looks for the darkest area, with some randomness.
- It will not place text in the same location twice in a row.

## How it works
- Picks a random background from `images`.
- Moves used files to `images_used`, so that we don't use the same image twice.
- Analyses brightness by region.
- Chooses the darkest region with randomness.
- Avoids repeating the last text location.
- Renders title and content slides.
- Writes JPEGs to `out`.

---

## Prompt to turn any text into the format we need in the CSV

~~~txt
Format the input into a CSV with this exact header:
post_id,heading,subheading,body

Post_id should just be a number
Rules:
- One row per slide, in order.
- First row for each post_id is the title slide. Use heading only. Leave subheading and body blank.
- No layout names or slide numbers.
- Short sentences. No metaphors. No emoji.
- No manual line breaks inside fields.
- Quote fields with commas. Escape quotes by doubling them.

Output CSV only. No notes.
~~~

## Prompt to turn an idea into a fully finished carousel in the format we need

~~~txt
Create [number] concise, useful Instagram carousels, that adds a lot of value.

Output format:
- CSV only with this exact header:
post_id,heading,subheading,body
- One row per slide, in order.
- Title slide uses heading only. Leave subheading and body blank.
- Quote fields with commas. Escape quotes by doubling them.
- No manual line breaks inside fields.

Inputs I will give:
- topic or idea

Style:
- Short sentences. Plain English. Direct voice.
- Use second person where helpful.
- Active verbs. Concrete advice.
- No metaphors, emoji, hype, or exclamation marks.
- Avoid filler and buzzwords/
- Keep numbers simple. Do not invent stats.

Field limits:
- heading: max 5 words. Clear benefit or action.
- subheading: max 10 words. Why it matters or what changes.
- body: 1 to 2 short sentences. Max 20 words.

Content quality:
- Every content slide must add value
- Each slide should cover one idea only.
- Avoid repeating points across slides.
- Prefer specifics over generalities

Structure:
Title slide, heading only, keep subheading and body blank
3-6 content slides: heading, subheading and body
Title should always be something like “Here are the 5 [things] that [make them seem useful]”

Goal:
- Produce the CSV only, with the exact header above. No extra commentary.

~~~


