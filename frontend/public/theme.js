"use strict";
// Loaded before first paint so a saved theme never flashes. No saved choice: follow the system.
try {
  const t = localStorage.getItem("reglens.theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch { /* storage blocked: system theme */ }
