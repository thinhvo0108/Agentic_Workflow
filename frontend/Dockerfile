FROM node:22-alpine AS base
ENV NODE_ENV=development

WORKDIR /app

# ── Development ────────────────────────────────────────────────────────────────
# Installs all deps (including devDependencies) and runs the Vite dev server.
# Source is mounted as a volume at runtime; node_modules live in the image.
FROM base AS dev
COPY package.json package-lock.json* ./
RUN npm install
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# ── Builder ────────────────────────────────────────────────────────────────────
# Produces the static dist/ bundle.
# NODE_ENV is intentionally NOT set to production here — npm ci would skip
# devDependencies (including tsc/vite) if it were. The vite build output is
# production-optimised regardless of NODE_ENV during the install phase.
FROM base AS builder
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# ── Production ─────────────────────────────────────────────────────────────────
# Serves the built bundle through nginx. nginx.conf proxies /api to the backend.
FROM nginx:1.27-alpine AS prod
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
