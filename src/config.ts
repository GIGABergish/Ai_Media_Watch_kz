export const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";
export const USE_BACKEND = (((import.meta as any).env?.VITE_USE_BACKEND) ?? "true") !== "false";
