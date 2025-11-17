document.addEventListener("DOMContentLoaded", function () {
    const messagesEl = document.getElementById("chat-messages");
    const formEl = document.getElementById("chat-form");
    const inputEl = document.getElementById("chat-input");

    if (!messagesEl || !formEl || !inputEl) {
        console.error("Chat elements not found on this page.");
        return;
    }

    const conversation = [];

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

    async function getBotReply(userText) {
    const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
        "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userText }),
    });

    const data = await response.json();
    return data.reply;
    }

    async function streamBotReply(messages, onChunk) {
        const response = await fetch("http://127.0.0.1:8000/chat-stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ messages }),
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