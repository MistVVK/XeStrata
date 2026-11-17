---
name: lora-caption
description: Captioning a LoRA's training images (tools/image/lora_caption.py) - what to write for a character or a style.
metadata:
  kind: tool
  tool: lora-caption
  style_exclude:
    - '\bsoft shading and flat colors typical of ukiyo[- ]e prints\b'
    - '\b(?:traditional\s+)?(?:japanese\s+)?(?:ukiyo[- ]e\s+)?woodblock(?: prints?)?(?:\s+style)?\b'
    - '\b(?:traditional\s+)?(?:japanese\s+)?ukiyo[- ]e(?:\s+(?:prints?|style|aesthetic|composition))?\b'
    - '(?<=landscape) print\b'
    - '\b(?:flat|vibrant|muted) colou?rs?\b'
    - '\b(?:bold|thick)(?: black)? outlines?\b'
    - '\bblack ink outlines?\b'
    - '\bink wash painting style\b'
    - '\brealistic look\b'
    - '\band line art typical of,? with minimal gradients\b'
    - '\bvisible brushwork texture\b'
    - '\b(?:flat|soft|cel) shading\b'
    - '\bno (?:lineart|shading)\b'
    - '\b(?:visible )?(?:textured )?paper (?:texture|appearance|background)(?: visible)?\b'
    - '\b(?:with )?(?:fine|delicate) line details\b'
    - '\bstylized\b'
    - '\b(?:traditional|dynamic|vertical|horizontal|circular) composition\b'
    - '\b(?:anime style|film grain)\b'
  reading_exclude:
    - '\bSpalding\s+\d+(?:\s+at\s+(?:bottom|top)\s+(?:left|right))?\b'
    - '\b(?:(?:a|an|the)\s+)?(?:museum|collector(?:''s)?|collection|accession)\s+(?:pencil marks?|numbers?|stamps?|seals?)\b'
    - '\s+read(?:ing|s)\s+[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\u3005\u3006]+(?:\s+[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\u3005\u3006]+)*'
    - '[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\u3005\u3006]+'
    - '["“][^"”]*(?:["”]|$)'
    - '(?<!\w)''[^'']+'''

---
You caption one training image of a LoRA. A word the caption writes is learned as that word; what the caption leaves
out is learned as the trigger word. So write out everything the LoRA must not learn, and leave out what it must.

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
- Image notes are context, not an instruction to invent a detail. Compare them with the picture.
- The `style_exclude` patterns in this skill remove recurring style and technique words from the final style caption.

Another LoRA (an object, a pose, a concept): leave out the thing it must learn, write the rest.

Write it as the model reads a prompt (the prompt style given: Danbooru tags, or English sentences). No quality or
score tags, no weights such as `(word:1.2)`.
