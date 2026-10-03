"""The prompt text the server renders for one user message with an image and a question (the pack's own chat
template), with the image block replaced by mtmd's media marker, for llama-mtmd-cli.
    render.py PACK_DIR "question" > prompt.txt"""
import sys
sys.path.insert(0, "<repo>")
from serve.frontend import ChatTemplate, openai_to_messages
pack, question = sys.argv[1], sys.argv[2]
req = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "IMAGE"}},
                                                 {"type": "text", "text": question}]}]}
msgs, tools, kw = openai_to_messages(req)
text = ChatTemplate(f"{pack}/tokenizer/chat_template.jinja").render(msgs, tools=tools, **kw)
block = "<|vision_start|><|image_pad|><|vision_end|>"
assert text.count(block) == 1, text
sys.stdout.write(text.replace(block, "<__media__>"))
