<template>
  <div class="chat-input-container">
    <div class="chat-input-wrapper">
      <textarea
        ref="textareaRef"
        v-model="message"
        class="chat-input"
        placeholder="Ask a question about your documents..."
        rows="1"
        @keydown="handleKeydown"
        @input="adjustHeight"
        :disabled="loading"
      />
      <Button
        class="send-button"
        variant="primary"
        :disabled="!message.trim() || loading"
        :loading="loading"
        @click="sendMessage"
      >
        <svg
          v-if="!loading"
          class="send-icon"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
          />
        </svg>
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from "vue";
import Button from "../common/Button.vue";

interface Emits {
  (e: "send", message: string): void;
}

interface Props {
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
});

const emit = defineEmits<Emits>();

const message = ref("");
const textareaRef = ref<HTMLTextAreaElement>();

const sendMessage = () => {
  if (message.value.trim() && !props.loading) {
    emit("send", message.value.trim());
    message.value = "";
    adjustHeight();
  }
};

const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
};

const adjustHeight = async () => {
  await nextTick();
  if (textareaRef.value) {
    textareaRef.value.style.height = "auto";
    textareaRef.value.style.height = `${textareaRef.value.scrollHeight}px`;
  }
};
</script>

<style scoped>
.chat-input-container {
  border-top: 1px solid var(--color-border);
  padding: var(--spacing-md);
  background: var(--color-background);
}

.chat-input-wrapper {
  display: flex;
  gap: var(--spacing-sm);
  align-items: flex-end;
  max-width: 100%;
}

.chat-input {
  flex: 1;
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  font-size: 0.9rem;
  font-family: inherit;
  resize: none;
  max-height: 120px;
  overflow-y: auto;
  transition: border-color 0.2s ease;
}

.chat-input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.chat-input:disabled {
  background-color: var(--color-surface);
  color: var(--color-text-muted);
  cursor: not-allowed;
}

.send-button {
  flex-shrink: 0;
  padding: var(--spacing-sm);
}

.send-icon {
  width: 1.25rem;
  height: 1.25rem;
}
</style>
