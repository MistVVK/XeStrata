---
name: find-region
description: Where something is in an image, or its mask for edit_image: region (English words) and image.
metadata:
  kind: tool
  tool: find_region
---

- `region`: short English words, as a label: `face`, `every cup`, `the dog`; every instance is marked.
- The result gives the marked picture and the mask's link: the mask goes to edit_image's `mask`, or edit_region
  takes the same words. A score below about 0.6: check the marked picture.
