document.addEventListener("DOMContentLoaded", function () {
    const messagesEl = document.getElementById("chat-messages");
    const formEl = document.getElementById("chat-form");
    const inputEl = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send");
    const nameInputEl = document.getElementById("name-input");
    const nameSaveEl = document.getElementById("name-save");
    const nameStatusEl = document.getElementById("name-status");

    if (!messagesEl || !formEl || !inputEl) {
        console.error("Chat elements not found on this page.");
        return;
    }

    // Check if on main page
    const nameContainerEl = document.getElementById("name-input-container");
    const isMainPage = window.location.pathname === "/" ||
                       window.location.pathname === "/about/" ||
                       window.location.pathname === "/about.html";

    // Load saved name if exists
    const savedName = localStorage.getItem('chat_user_name');
    if (savedName && nameInputEl) {
        nameInputEl.value = savedName;
        // Hide container on main page if name already saved
        if (isMainPage && nameContainerEl) {
            nameContainerEl.style.display = "none";
        } else if (nameStatusEl) {
            nameStatusEl.textContent = "✓ Saved";
        }
    }

    if (nameSaveEl && nameInputEl) {
        nameSaveEl.addEventListener("click", function () {
            const name = nameInputEl.value.trim();
            if (name) {
                localStorage.setItem('chat_user_name', name);
                // Hide container on main page after saving
                if (isMainPage && nameContainerEl) {
                    nameContainerEl.style.display = "none";
                } else if (nameStatusEl) {
                    nameStatusEl.textContent = "✓ Saved";
                }
            } else {
                localStorage.removeItem('chat_user_name');
                if (nameStatusEl) {
                    nameStatusEl.textContent = "";
                }
            }
        });
    }

    const conversation = [];

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

        addMessage("You", text, false);
        conversation.push({ role: "user", content: text });
        inputEl.value = "";

        const botMessageEl = addMessage("Resume", "", true);
        botMessageEl.textContent = "...";
        let botText = "";
        let firstChunk = true;

        setInputEnabled(false);

        try {
            await streamBotReply(conversation, (chunk) => {
                if (firstChunk) {
                    botMessageEl.textContent = "";
                    firstChunk = false;
                }
                botText += chunk;
                botMessageEl.innerHTML = marked.parse(botText);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            });
            conversation.push({ role: "assistant", content: botText });
        } catch (err) {
            console.error("Chat error:", err);
            if (firstChunk) {
                botMessageEl.textContent = "";
            }
            botMessageEl.textContent = "Something went wrong. Please try again.";
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
        // First check if user has saved a name
        const savedName = localStorage.getItem('chat_user_name');
        if (savedName) {
            return savedName;
        }
        // Fall back to random hash
        let hashId = localStorage.getItem('chat_user_id');
        if (!hashId) {
            hashId = crypto.randomUUID ? crypto.randomUUID() : generateUUID();
            localStorage.setItem('chat_user_id', hashId)
        }
        return hashId;
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
                user_id: getOrCreateUserId()
            }),
        });

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

    function addMessage(sender, text, isMarkdown) {
        const messageEl = document.createElement("div");
        messageEl.className = "chat-message";

        const senderEl = document.createElement("strong");
        senderEl.textContent = sender + ": ";

        const textEl = document.createElement("div");
        if (isMarkdown && text) {
            textEl.innerHTML = marked.parse(text);
        } else {
            textEl.textContent = text;
        }

        messageEl.appendChild(senderEl);
        messageEl.appendChild(textEl);
        messagesEl.appendChild(messageEl);

        return textEl;
    }

});
