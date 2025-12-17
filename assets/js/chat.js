document.addEventListener("DOMContentLoaded", function () {
    const messagesEl = document.getElementById("chat-messages");
    const formEl = document.getElementById("chat-form");
    const inputEl = document.getElementById("chat-input");

    if (!messagesEl || !formEl || !inputEl) {
        console.error("Chat elements not found on this page.");
        return;
    }

    const conversation = [];
    const userId = getOrCreateUserId();

    formEl.addEventListener("submit", async function (event) {
        event.preventDefault();

        const text = inputEl.value.trim();
        if (text === "") {
            return;
        }

        addMessage("You", text);
        conversation.push({ role: "user", content: text });

        const botMessageEl = addMessage("Resume", "");
        let botText = "";

        await streamBotReply(conversation, (chunk) => {
            botText += chunk;
            botMessageEl.textContent = botText;
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });

        conversation.push({ role: "assistant", content: botText });

        inputEl.value = "";
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
        let userId = localStorage.getItem('chat_user_id');
        if (!userId) {
            userId = crypto.randomUUID ? crypto.randomUUID() : generateUUID();
            localStorage.setItem('chat_user_id', userId)
        }
        return userId;
    }
    async function streamBotReply(messages, onChunk) {
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
                messages: messages,
                user_id: userId || getOrCreateUserId()
            }),
        });

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

    function addMessage(sender, text) {
        const messageEl = document.createElement("div");
        messageEl.className = "chat-message";

        const senderEl = document.createElement("strong");
        senderEl.textContent = sender + ": ";

        const textEl = document.createElement("div");
        textEl.textContent = text;

        messageEl.appendChild(senderEl);
        messageEl.appendChild(textEl);
        messagesEl.appendChild(messageEl);

        return textEl;
    }

});