---
layout: default
title: "chat"
permalink: /chat/
author_profile: false
---

<div id="chat-container">
  <div id="chat-messages"></div>

  <form id="chat-form">
    <input
      id="chat-input"
      type="text"
      placeholder="Type a message..."
      autocomplete="off"
    />
    <button id="chat-send" type="submit">Send</button>
  </form>
</div>

<script src="/assets/js/chat.js"></script>