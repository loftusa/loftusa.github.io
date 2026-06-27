document.addEventListener("DOMContentLoaded", function () {
    const messagesEl = document.getElementById("chat-messages");
    const formEl = document.getElementById("chat-form");
    const inputEl = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send");

    if (!messagesEl || !formEl || !inputEl) {
        console.error("Chat elements not found on this page.");
        return;
    }

    // Wake up Fly.io machine on page load so it's warm by the time user types
    const isLocalhost = window.location.hostname === "localhost" ||
                        window.location.hostname === "127.0.0.1";
    const apiBase = isLocalhost
        ? "http://127.0.0.1:8000"
        : "https://llm-resume-restless-thunder-9259.fly.dev";
    fetch(apiBase + "/health").catch(() => {});

    function setInputEnabled(enabled) {
        inputEl.disabled = !enabled;
        if (sendBtn) sendBtn.disabled = !enabled;
    }

    formEl.addEventListener("submit", async function (event) {
        event.preventDefault();

        const text = inputEl.value.trim();
        if (text === "") {
            return;
        }

        addMessage("user", text, false);
        inputEl.value = "";

        const botMessageEl = addMessage("bot", "", true);
        botMessageEl.textContent = "...";
        let botText = "";
        let firstChunk = true;

        setInputEnabled(false);

        try {
            await streamBotReply(text, (chunk) => {
                if (firstChunk) {
                    botMessageEl.textContent = "";
                    firstChunk = false;
                }
                botText += chunk;
                botMessageEl.innerHTML = DOMPurify.sanitize(marked.parse(botText));
                messagesEl.scrollTop = messagesEl.scrollHeight;
            });
        } catch (err) {
            console.error("Chat error:", err);
            if (firstChunk) {
                botMessageEl.textContent = "";
            }
            if (err.message === "rate_limit") {
                botMessageEl.textContent = "This conversation has gotten pretty long! Please refresh the page to start a new one.";
            } else {
                botMessageEl.textContent = "Something went wrong. Please try again.";
            }
        } finally {
            setInputEnabled(true);
            inputEl.focus();
        }
    });

    function generateUUID() {
        // Fallback if crypto.randomUUID() not available
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
    function getOrCreateUserId() {
        let hashId = localStorage.getItem('chat_user_id');
        if (!hashId) {
            hashId = crypto.randomUUID ? crypto.randomUUID() : generateUUID();
            localStorage.setItem('chat_user_id', hashId);
        }
        return hashId;
    }
    async function streamBotReply(message, onChunk) {
        const isLocalhost = window.location.hostname === "localhost" ||
                            window.location.hostname === "127.0.0.1";
        const apiUrl = isLocalhost
            ? "http://127.0.0.1:8000/chat?logging=true"
            : "https://llm-resume-restless-thunder-9259.fly.dev/chat?logging=true";

        const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                user_id: getOrCreateUserId()
            }),
        });

        if (response.status === 429) {
            throw new Error("rate_limit");
        }
        if (!response.ok) {
            throw new Error("API returned " + response.status);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;

        while (!done) {
            const { value, done: streamDone } = await reader.read();
            done = streamDone;
            if (value) {
                const chunkText = decoder.decode(value, { stream: !done });
                onChunk(chunkText);
            }
        }
    }

    function addMessage(role, text, isMarkdown) {
        const messageEl = document.createElement("div");
        messageEl.className = "chat-msg chat-msg-" + role;

        const textEl = document.createElement("div");
        textEl.className = "chat-text";

        if (role === "user") {
            const labelEl = document.createElement("span");
            labelEl.className = "chat-label";
            labelEl.textContent = "you";
            messageEl.appendChild(labelEl);
        }

        if (isMarkdown && text) {
            textEl.innerHTML = DOMPurify.sanitize(marked.parse(text));
        } else {
            textEl.textContent = text;
        }

        messageEl.appendChild(textEl);
        messagesEl.appendChild(messageEl);

        return textEl;
    }

});
