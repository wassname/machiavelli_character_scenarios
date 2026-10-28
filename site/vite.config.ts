import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so the build works from any GitHub Pages sub-path. -- PI/OpenAI
export default defineConfig({ plugins: [react()], base: './' })
