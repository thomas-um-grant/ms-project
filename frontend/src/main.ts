import "./assets/main.css"; // central design system + app styles

import { createApp } from "vue";
import { createPinia } from "pinia";

import App from "./App.vue";
import router from "./router";
import { themeService } from "./services/ThemeService";

// Unified theme initialization (idempotent)
themeService.init();

const app = createApp(App);

app.use(createPinia());
app.use(router);

app.mount("#app");
