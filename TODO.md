# Chat feature TODOs:

## Immediate
~~get initial chat box HTML and JS frontend working (no LLM call yet)~~
~~set up fastAPI backend~~
~~set up cerebras LLM API call to gpt-oss-120B~~
~~deploy on production w/ Fly.io ~~
~~switch to streaming API~~
~~remove references to non-streaming chat~~
~~add conversation history support~~
~~add health check endpoint and CI/CD~~
~~- keep Fly.io machine warm~~
~~- add chat logging~~
    - stats on chat logging
    - notify me whenever a new user starts a chat

## Future (major)
- classifier head that predicts whether a user is asking a question about the resume or not
- build an evaluation harness for accuracy
    - Q&A questions about my resume
    - call backend on each question
    - log raw responses, latency, errors
    - output summary report
- add live monitoring (tokens/sec, first response latency)
- make faster
    - dont keep passing conversation history back and forth (store on Fly.io server)
    - Preload KV cache w/ background context whenever a new user accesses the website
        - make sure not to leak info to the HTML frontend
- make more cost-efficient
    - have fly.io machine start up only when at least one user is on the site (upon entering the site)
- make UI prettier (anthropic-style, rendered as a box inline, scrolling possible)
- Add RAG with vector embeddings. Should contain 
  - all links in the resume so that the chat model has access to them if it needs
  - the code used to create the model chat window so that the model can figure out how it is set up

## future (minor)
- make "send" button and input box lower w.r.t the conversation
- switch to typescript over javascript

