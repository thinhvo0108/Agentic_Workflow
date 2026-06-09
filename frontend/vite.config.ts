import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// When running inside Docker, docker-compose sets VITE_API_PROXY_TARGET=http://backend:8000.
// For local dev (outside Docker), the backend is on localhost:8000.
const apiTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,  // listen on 0.0.0.0 so the Docker port mapping works
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
