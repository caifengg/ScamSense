import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/login': 'http://localhost:5000',
      '/signup': 'http://localhost:5000',
      '/detect': 'http://localhost:5000',
      // Text Extractor's endpoint - forwards to the same Flask backend as
      // /detect above, just a different route (see app.py's detect_text()).
      '/detect-text': 'http://localhost:5000',
      '/admin/stats': 'http://localhost:5000',
    },
  },
})
