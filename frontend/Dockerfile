# --- Build stage ---
FROM node:20-alpine AS build
WORKDIR /app

# Install deps first for better cache
COPY package.json package-lock.json* ./
RUN npm ci || npm i

# Copy the rest of the frontend source
COPY . .

# Build the static site
RUN npm run build

# --- Runtime stage ---
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80