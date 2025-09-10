<template>
  <div class="carousel-container">
    <div class="carousel-header">
      <h3 class="carousel-title">{{ title }}</h3>
      <div class="carousel-controls">
        <button class="carousel-btn carousel-btn-prev" @click="scrollLeft">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path
              d="M15 18L9 12L15 6"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
        </button>
        <button class="carousel-btn carousel-btn-next" @click="scrollRight">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path
              d="M9 18L15 12L9 6"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
        </button>
      </div>
    </div>

    <div class="carousel-track" ref="trackRef">
      <div class="carousel-items">
        <div
          v-for="item in items"
          :key="item.id"
          class="carousel-item"
          :style="{ '--item-color': item.color }"
          @click="$emit('itemClick', item)"
        >
          <div class="item-icon">{{ item.icon }}</div>
          <div class="item-content">
            <h4 class="item-name">{{ item.displayName }}</h4>
            <p class="item-category">{{ item.category }}</p>
            <p class="item-description">{{ item.description }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";

interface CarouselItem {
  id: string;
  name: string;
  displayName: string;
  category: string;
  description: string;
  icon: string;
  color: string;
  details?: any;
}

interface Props {
  title: string;
  items: CarouselItem[];
}

defineProps<Props>();
defineEmits(["itemClick"]);

const trackRef = ref<HTMLElement>();

const scrollLeft = () => {
  if (!trackRef.value) return;
  trackRef.value.scrollBy({ left: -300, behavior: "smooth" });
};

const scrollRight = () => {
  if (!trackRef.value) return;
  trackRef.value.scrollBy({ left: 300, behavior: "smooth" });
};
</script>

<style scoped>
.carousel-container {
  margin-bottom: var(--spacing-lg);
}

.carousel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
}

.carousel-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0;
}

.carousel-controls {
  display: flex;
  gap: var(--spacing-xs);
}

.carousel-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: 1px solid var(--color-border);
  background: var(--color-background);
  border-radius: 6px;
  color: var(--color-text-primary);
  cursor: pointer;
  transition: all 0.2s ease;
}

.carousel-btn:hover {
  background: var(--color-background-hover);
  border-color: var(--color-border-hover);
}

.carousel-track {
  overflow-x: auto;
  scrollbar-width: none;
  -ms-overflow-style: none;
}

.carousel-track::-webkit-scrollbar {
  display: none;
}

.carousel-items {
  display: flex;
  gap: var(--spacing-md);
  padding-bottom: var(--spacing-sm);
}

.carousel-item {
  flex: 0 0 280px;
  background: var(--color-background);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  padding: var(--spacing-md);
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.carousel-item::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--item-color);
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.carousel-item:hover::before {
  transform: scaleX(1);
}

.carousel-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--color-border-hover);
}

.item-icon {
  font-size: 2rem;
  margin-bottom: var(--spacing-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  background: rgba(var(--item-color-rgb, 76, 175, 80), 0.1);
  border-radius: 8px;
}

.item-content {
  flex: 1;
}

.item-name {
  font-size: 1rem;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 var(--spacing-xs) 0;
}

.item-category {
  font-size: 0.75rem;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 500;
  margin: 0 0 var(--spacing-sm) 0;
}

.item-description {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  line-height: 1.4;
  margin: 0;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .carousel-item {
    flex: 0 0 240px;
  }

  .carousel-controls {
    display: none;
  }
}
</style>
