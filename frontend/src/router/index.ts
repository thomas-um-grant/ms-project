import { createRouter, createWebHistory } from "vue-router";
import DemoView from "../views/DemoView.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      redirect: "/demo",
    },
    {
      path: "/demo",
      name: "demo",
      component: DemoView,
    },
    {
      path: "/evaluation",
      name: "evaluation",
      // route level code-splitting
      // this generates a separate chunk (Evaluation.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import("../views/EvaluationView.vue"),
    },
  ],
});

export default router;
