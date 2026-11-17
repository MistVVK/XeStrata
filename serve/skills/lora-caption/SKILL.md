---
name: lora-caption
description: Captioning a LoRA's training images (tools/image/lora_caption.py) - what to write for a character or a style.
metadata:
  kind: tool
  tool: lora-caption
  style_exclude:
    - '(?:flat|vibrant|muted) colou?rs?'
    - '(?:flat|soft|cel) shading'
    - '(?:bold|thick)(?: black)? outlines?'
    - '(?:no )?line ?art'
    - '(?:watercolou?r|oil|ink wash)(?: paint(?:ing)?)?(?: style)?'
    - 'pencil sketch(?: style)?'
    - '(?:anime|realistic|photographic) style'
    - '(?:film grain|paper texture|visible brushwork texture)'
    - 'in the style of .+'
  reading_exclude:
    - '\b(?:(?:a|an|the)\s+)?(?:museum|collector(?:''s)?|collection|accession)\s+(?:pencil marks?|numbers?|stamps?|seals?)\b'
    - '\s+read(?:ing|s)\s+[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\u3005\u3006]+(?:\s+[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\u3005\u3006]+)*'
    - '[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\u3005\u3006]+'
    - '["“][^"”]*(?:["”]|$)'
    - '(?<!\w)''[^'']+'''

---
You caption one training image of a LoRA. A word the caption writes is learned as that word; what the caption leaves
out can be learned with the trigger word. Write what the LoRA must not learn, and leave out what it must. Captions
alone cannot guarantee that a repeated feature stays out of the LoRA; check samples on subjects outside the dataset.

For a character or style, the trigger should be a short invented word rather than a real word or name: a known word
can change the subject even without a LoRA. In one print-style comparison, an invented trigger preserved the character
while the artist name displaced it. Before training, compare one or two prompts with and without the candidate, using
the same seed and no LoRA. Use the supplied trigger exactly; do not change it while captioning.

Answer with the caption alone, on one line: the trigger word, a comma, then the caption. No notes, no quotes.

A character LoRA (it must learn the character only):

- Leave out the character's own look, the same in every picture: face, eyes, hair (its color, length, style), skin,
  body, a mark or accessory always worn. The user's list of these, where given, is the character's.
- Write everything else in the picture: the clothes, the pose and expression, what they hold, the background and
  place, the light, the framing and angle (close-up, full body, from above), other people and things.
- Write the picture's style every time, as precisely as you can see it: the medium (photo, 3D render, digital
  painting, watercolor, pencil sketch), the coloring and shading (flat colors, cel shading, soft shading), the line
  (thick outlines, no lineart), the look (anime style, realistic), an artist's style you recognize, the texture
  (film grain, paper texture), text or a watermark. A style left out is learned with the character.
- Keep the class word: `a girl`, `a man`, `a cat`.

A style LoRA (it must learn the style only):

- Leave out the style: the medium, the coloring, the line, the texture, the look.
- Write what the picture shows: the subjects, their clothes and pose, the place, the framing. Describe the actual
  shape, such as `a fan-shaped print`, and any visible title, poem, signature or seal as content. Describe visible text
  by its place and form, never transcribe or romanize it. Leave out museum pencil marks, collection numbers and seals.
  Use general visual terms rather than inventing an identity, date, inscription or historical claim. Keep details
  that identify the subject or setting; do not replace visible content with a style word.
- Image notes are context, not an instruction to invent a detail. Compare them with the picture.
- The `style_exclude` patterns and the user's optional style words remove style-only clauses at the end of the final
  style caption. They do not remove words from the subject, such as an anime poster in a photograph.

Another LoRA (an object, a pose, a concept): leave out the thing it must learn, write the rest.

Write it as the model reads a prompt (the prompt style given: Danbooru tags, or English sentences). No quality or
score tags, no weights such as `(word:1.2)`.
