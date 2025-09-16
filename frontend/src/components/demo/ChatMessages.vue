<template>
  <div class="chat-messages" ref="messagesRef">
    <div v-if="messages.length === 0" class="empty-state">
      <div class="empty-icon">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
          />
        </svg>
      </div>
      <h3>Start a conversation</h3>
      <p>
        Ask questions about your documents and get AI-powered answers with
        source citations.
      </p>
    </div>

    <div
      v-for="message in messages"
      :key="message.id"
      class="message"
      :class="message.type"
    >
      <div class="message-avatar">
        <div class="avatar" :class="message.type">
          <svg
            v-if="message.type === 'user'"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"
            />
          </svg>
          <svg v-else fill="currentColor" viewBox="0 0 24 24">
            <path
              d="M12,2A2,2 0 0,1 14,4C14,4.74 13.6,5.39 13,5.73V7A1,1 0 0,0 14,8H18A1,1 0 0,0 19,7V5.73C18.4,5.39 18,4.74 18,4A2,2 0 0,1 20,2A2,2 0 0,1 22,4C22,4.74 21.6,5.39 21,5.73V7A3,3 0 0,1 18,10H14A3,3 0 0,1 11,7V5.73C10.4,5.39 10,4.74 10,4A2,2 0 0,1 12,2Z"
            />
          </svg>
        </div>
      </div>

      <div class="message-content">
        <div class="message-bubble">
          <div class="message-text">{{ displayMessageText(message) }}</div>
          <div class="message-time">
            {{ formatTime(message.timestamp) }}
          </div>
        </div>

        <!-- Show retrieved documents for assistant messages -->
        <div
          v-if="
            message.type === 'assistant' && message.retrieved_documents?.length
          "
          class="retrieved-docs-info"
        >
          <span class="docs-count">
            {{ message.retrieved_documents.length }} source{{
              message.retrieved_documents.length !== 1 ? "s" : ""
            }}
            retrieved
          </span>
        </div>
      </div>
    </div>

    <!-- Typing indicator when loading -->
    <div v-if="isLoading" class="message assistant">
      <div class="message-avatar">
        <div class="avatar assistant">
          <svg fill="currentColor" viewBox="0 0 24 24">
            <path
              d="M12,2A2,2 0 0,1 14,4C14,4.74 13.6,5.39 13,5.73V7A1,1 0 0,0 14,8H18A1,1 0 0,0 19,7V5.73C18.4,5.39 18,4.74 18,4A2,2 0 0,1 20,2A2,2 0 0,1 22,4C22,4.74 21.6,5.39 21,5.73V7A3,3 0 0,1 18,10H14A3,3 0 0,1 11,7V5.73C10.4,5.39 10,4.74 10,4A2,2 0 0,1 12,2Z"
            />
          </svg>
        </div>
      </div>
      <div class="message-content">
        <div class="message-bubble">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from "vue";
import type { ChatMessage } from "../../types/api";

interface Props {
  messages: ChatMessage[];
  isLoading?: boolean;
}

const props = defineProps<Props>();

const messagesRef = ref<HTMLElement>();

// Prefer showing the textual part of a response object rather than the whole object
const displayMessageText = (msg: ChatMessage): string => {
  const c: any = (msg as any).content;
  if (typeof c === "string") return c;
  if (c && typeof c === "object") {
    return c.response ?? c.text ?? c.message ?? c.output ?? c.content ?? "";
  }
  return c != null ? String(c) : "";
};

const formatTime = (timestamp: Date) => {
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
};

const scrollToBottom = async () => {
  await nextTick();
  if (messagesRef.value) {
    messagesRef.value.scrollTop = messagesRef.value.scrollHeight;
  }
};

// Auto-scroll when new messages are added
watch(() => props.messages.length, scrollToBottom);
watch(() => props.isLoading, scrollToBottom);
</script>

<style scoped>
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: var(--spacing-2xl);
  color: var(--color-text-muted);
  height: 100%;
}

.empty-icon {
  width: 4rem;
  height: 4rem;
  margin-bottom: var(--spacing-md);
  opacity: 0.5;
}

.empty-state h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: var(--spacing-sm);
  color: var(--color-text-primary);
}

.message {
  display: flex;
  gap: var(--spacing-sm);
  max-width: 85%;
}

.message.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.message.assistant {
  align-self: flex-start;
}

.message-avatar {
  flex-shrink: 0;
}

.avatar {
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.avatar.user {
  background-color: var(--color-primary);
}

.avatar.assistant {
  background-color: var(--color-secondary);
}

.avatar svg {
  width: 1rem;
  height: 1rem;
}

.message-content {
  flex: 1;
  min-width: 0;
}

.message-bubble {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
}

.message.user .message-bubble {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.message-text {
  font-size: 0.9rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.message-time {
  font-size: 0.75rem;
  opacity: 0.7;
  margin-top: var(--spacing-xs);
}

.retrieved-docs-info {
  margin-top: var(--spacing-xs);
  font-size: 0.75rem;
  color: var(--color-text-muted);
}

.docs-count {
  font-style: italic;
}

.typing-indicator {
  display: flex;
  gap: 4px;
  align-items: center;
}

.typing-indicator span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: var(--color-text-muted);
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) {
  animation-delay: -0.32s;
}

.typing-indicator span:nth-child(2) {
  animation-delay: -0.16s;
}

@keyframes typing {
  0%,
  80%,
  100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}
</style>
