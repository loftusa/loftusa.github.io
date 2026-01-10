---
layout: default
title: "chat"
permalink: /chat/
author_profile: false
---

<div id="chat-container">
  <div id="name-input-container">
    <label for="name-input">Your name (optional):</label>
    <input
      id="name-input"
      type="text"
      placeholder="Enter your name"
      autocomplete="off"
    />
    <button id="name-save" type="button">Save</button>
    <span id="name-status"></span>
  </div>

  <div id="chat-messages"></div>

  <form id="chat-form">
    <input
      id="chat-input"
      type="text"
      placeholder="Talk to my resume!"
      autocomplete="off"
    />
    <button id="chat-send" type="submit">Send</button>
  </form>
</div>

<script src="/assets/js/chat.js"></script>