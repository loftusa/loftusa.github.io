# Chat feature TODOs:

## Immediate
- Deploy FastAPI backend to production

## Future (major)
- make UI prettier (anthropic-style, rendered as a box inline, scrolling possible)
- Add RAG with vector embeddings. Should contain 
  - all links in the resume so that the chat model has access to them if it needs
  - the code used to create the model chat window so that the model can figure out how it is set up

## bug fixes
- make "send" button and input box lower w.r.t the conversation
- remove references to non-streaming chat