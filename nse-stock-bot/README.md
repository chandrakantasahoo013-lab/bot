# NSE Stock Bot Dashboard

## Overview
This project is a Flask-based stock analysis dashboard for NSE stocks, featuring a Golden Cross strategy, technical indicators, and a responsive frontend. It runs in Docker containers.

## Prerequisites
- Docker Desktop (with Docker Compose)
- Windows 10/11 with WSL 2 and virtualization enabled

## Setup
1. Install Docker Desktop:
   - Download from https://www.docker.com/products/docker-desktop/
   - Run the installer and enable WSL 2/Hyper-V if prompted.
2. Clone or copy this repository to `D:\nse-stock-bot`.
3. Update `backend/golden_cross_bot_india_v6.py` with your Gmail `SENDER_EMAIL` and `EMAIL_PASSWORD` (use App Password).
4. Edit `docker-compose.yml` environment variables with the same email credentials.
5. Build and run:
   ```powershell
   cd D:\nse-stock-bot
   docker compose up --build -d