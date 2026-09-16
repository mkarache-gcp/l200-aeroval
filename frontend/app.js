// AeroEval Chat Frontend with Model Toggling

const messagesArea = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const providerSelect = document.getElementById("provider-select");
const sessionId = "session-" + Math.random().toString(36).substring(2, 9);

function appendMessage(role, text, agentName = null, pendingDraft = null) {
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
    
    const badge = agentName ? `<span style="font-size: 0.75rem; color: #38bdf8; margin-left: 6px; background: rgba(56, 189, 248, 0.1); padding: 2px 6px; border-radius: 4px;">[${agentName}]</span>` : "";
    bubble.innerHTML = `<strong>AeroEval</strong>${badge}:<br/>${formatted}`;

    // If human-in-the-loop signoff is pending, display interactive action buttons
    if (pendingDraft && pendingDraft.draft_id) {
      const actionPanel = document.createElement("div");
      actionPanel.style.marginTop = "12px";
      actionPanel.style.display = "flex";
      actionPanel.style.gap = "10px";

      const approveBtn = document.createElement("button");
      approveBtn.textContent = "✅ Approve & File Ticket";
      approveBtn.style.padding = "6px 14px";
      approveBtn.style.background = "#22c55e";
      approveBtn.style.color = "#ffffff";
      approveBtn.style.border = "none";
      approveBtn.style.borderRadius = "4px";
      approveBtn.style.cursor = "pointer";
      approveBtn.style.fontWeight = "600";

      const rejectBtn = document.createElement("button");
      rejectBtn.textContent = "❌ Discard Draft";
      rejectBtn.style.padding = "6px 14px";
      rejectBtn.style.background = "#ef4444";
      rejectBtn.style.color = "#ffffff";
      rejectBtn.style.border = "none";
      rejectBtn.style.borderRadius = "4px";
      rejectBtn.style.cursor = "pointer";
      rejectBtn.style.fontWeight = "600";

      approveBtn.addEventListener("click", async () => {
        approveBtn.disabled = true;
        rejectBtn.disabled = true;
        await submitApproval(pendingDraft.draft_id, true);
      });

      rejectBtn.addEventListener("click", async () => {
        approveBtn.disabled = true;
        rejectBtn.disabled = true;
        await submitApproval(pendingDraft.draft_id, false);
      });

      actionPanel.appendChild(approveBtn);
      actionPanel.appendChild(rejectBtn);
      bubble.appendChild(actionPanel);
    }
  } else {
    bubble.textContent = text;
  }

  msgDiv.appendChild(bubble);
  messagesArea.appendChild(msgDiv);
  messagesArea.scrollTop = messagesArea.scrollHeight;
}

async function submitApproval(draftId, approved) {
  try {
    const res = await fetch("/approve-action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        draft_id: draftId,
        approved: approved,
      }),
    });
    const data = await res.json();
    if (approved && data.status === "TICKET_CREATED") {
      appendMessage(
        "assistant",
        `🎉 **Ticket Officially Filed to Issue Tracker!**\n\nTicket ID: \`${data.ticket_id}\`\nStatus: Registered with Human Signoff.`,
        "HITL Approval Hook"
      );
    } else {
      appendMessage("assistant", `Draft \`${draftId}\` discarded. No ticket was filed.`, "HITL Approval Hook");
    }
  } catch (err) {
    appendMessage("assistant", `⚠️ Failed to record approval: ${err.message}`, "HITL Approval Hook");
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  const selectedProvider = providerSelect ? providerSelect.value : "auto";

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
    const agentLabel = data.agent ? `${data.agent} · ${data.model || data.provider}` : (data.provider === "anthropic" ? "DocGen (Claude)" : "AeroEval (Gemini)");
    appendMessage("assistant", data.response, agentLabel, data.requires_approval ? data.pending_draft : null);
  } catch (err) {
    appendMessage("assistant", `⚠️ Error contacting agent server: ${err.message}`);
  } finally {
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
  }
});
