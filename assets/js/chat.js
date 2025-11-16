document.addEventListener("DOMContentLoaded", function () {
    const messagesEl = document.getElementById("chat-messages");
    const formEl = document.getElementById("chat-form");
    const inputEl = document.getElementById("chat-input");

    if (!messagesEl || !formEl || !inputEl) {
        console.error("Chat elements not found on this page.");
        return;
    }

    formEl.addEventListener("submit", async function (event) {
        event.preventDefault();

        const text = inputEl.value.trim();
        if (text === "") {
            return;
        }

        addMessage("You", text);
        const reply = await getBotReply(text);
        addMessage("Bot", reply);

        inputEl.value = "";
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });

    async function getBotReply(userText) {
        await new Promise((resolve) => setTimeout(resolve, 300)); // simulate latency
        return "hello world";
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
    }

});