# Custom icons

Original SVGs supplied by Niko in `modelandweights.zip`, 10 October 2026.

- `models.svg`: model library and model family.
- `model-file.svg`: an individual model version or weights file.

The originals retain their metadata and vector geometry. The app sprite strips
metadata and uses their 24 x 24 viewBox, currentColor, 1.5px strokes, and 12% fills.
They are already in the checked-in sprite, `public/icons/sprite.svg`, which is the source: a new icon is added there as a `<symbol id="i-NAME">`.

Also supplied by Niko on 10 October 2026:

- `skills.svg` and `engine.svg` from `skillsandengineicon.zip`: Skills navigation,
  skill rows, Engine navigation, and Auto-tune.
- `family-qwen.svg` from `qwenicon.zip`: Qwen model family, chat model picker,
  reasoning menu, and the loaded model chip. Other families keep the generic
  model library icon.

Their original SVG bytes are preserved. Sprite symbols use the same geometry
with metadata removed. Thinking and speed controls keep their existing icons.
