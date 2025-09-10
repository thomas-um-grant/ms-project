/// <reference types="vite/client" />

// Provide TypeScript module declarations for Vue SFCs
declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<{}, {}, any>;
  export default component;
}
