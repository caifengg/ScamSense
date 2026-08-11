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
      // Text Extractor's screenshot-upload endpoint - same backend, see
      // extract_text_from_image_route() in app.py.
      '/extract-text-from-image': 'http://localhost:5000',
      // Text Extractor's per-user history/flag endpoints - same backend,
      // see list_text_checks()/flag_text_check()/delete_text_check() in app.py.
      '/text-checks': 'http://localhost:5000',
      '/admin/stats': 'http://localhost:5000',
    },
  },
})
