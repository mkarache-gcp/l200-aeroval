// AeroEval Chat Frontend with Model Toggling

const messagesArea = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const providerSelect = document.getElementById("provider-select");
const sessionId = "session-" + Math.random().toString(36).substring(2, 9);

function appendMessage(role, text, providerName = null) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "assistant") {
    let formatted = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`(.*?)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br/>");
    
    const badge = providerName ? `<span style="font-size: 0.75rem; color: #38bdf8; margin-left: 6px;">[${providerName}]</span>` : "";
    bubble.innerHTML = `<strong>AeroEval</strong>${badge}: ${formatted}`;
  } else {
    bubble.textContent = text;
  }

  msgDiv.appendChild(bubble);
  messagesArea.appendChild(msgDiv);
  messagesArea.scrollTop = messagesArea.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  const selectedProvider = providerSelect ? providerSelect.value : "gemini";

  // Render user message
  appendMessage("user", text);
  chatInput.value = "";
  chatInput.disabled = true;
  sendButton.disabled = true;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: sessionId,
        provider: selectedProvider,
      }),
    });

    if (!res.ok) {
      throw new Error(`Server returned status ${res.status}`);
    }

    const data = await res.json();
    const providerLabel = data.provider === "anthropic" ? "Claude" : "Gemini";
    appendMessage("assistant", data.response, providerLabel);
  } catch (err) {
    appendMessage("assistant", `⚠️ Error contacting agent server: ${err.message}`);
  } finally {
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
  }
});
