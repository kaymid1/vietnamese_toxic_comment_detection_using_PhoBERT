import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = process.env.VITE_API_BASE_URL?.trim()
    || env.VITE_API_BASE_URL?.trim()
    || 'http://127.0.0.1:8000'

  return {
    plugins: [
      // The React and Tailwind plugins are both required for Make, even if
      // Tailwind is not being actively used – do not remove them
      react(),
      tailwindcss(),
    ],
    server: {
      host: true,
      port: 5173,
      allowedHosts: ['.ngrok-free.app', 'localhost', '127.0.0.1'],
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
        },
        '/health': {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    resolve: {
      alias: {
        // Alias @ to the src directory
        '@': path.resolve(__dirname, './src'),
      },
    },
  }
})
