import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy development when running backend separately (optional, as Docker uses Nginx reverse proxy)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})