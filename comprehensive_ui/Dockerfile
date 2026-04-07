# =============================================================================
# Frontend Dockerfile — multi-stage build
#   Stage 1 (builder): Node 20 → npm build → dist/
#   Stage 2 (serve):   nginx:alpine serves the static files (~30 MB final image)
# =============================================================================

# ---- Stage 1: Build ----
FROM node:20-slim AS builder
WORKDIR /app

# Copy dependency manifest first so Docker can cache the npm install layer
# independently of source changes.
COPY package.json ./

# TODO: commit a package-lock.json or pnpm-lock.yaml to git, then change this to:
#   npm ci                              (for npm)
#   pnpm install --frozen-lockfile      (for pnpm)
# Reproducible lockfile installs are faster and more secure than npm install.
RUN npm install

# Copy source and build — outputs to dist/
COPY . .
RUN npm run build

# ---- Stage 2: Serve ----
# nginx:1.27-alpine: ~7 MB, supports AMD64 and ARM64 out of the box.
FROM nginx:1.27-alpine

# Serve built React app
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config: SPA fallback + reverse proxy to backend service
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
