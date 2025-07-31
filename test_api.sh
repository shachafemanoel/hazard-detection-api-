#!/usr/bin/env bash

# Simple API health check script for the Hazard Detection FastAPI service

set -e

API_URL="${API_URL:-}"
if [ -z "$API_URL" ]; then
  read -rp "Enter API URL: " API_URL
fi
if [ -z "$API_URL" ]; then
  echo "API URL is required" >&2
  exit 1
fi

# Helper to fetch status code
check_status() {
  local url=$1
  local expected=$2
  local status
  status=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
  if [ "$status" != "$expected" ]; then
    echo "❌ $url returned HTTP $status, expected $expected" >&2
    return 1
  fi
  return 0
}

# 1. GET /health
check_status "$API_URL/health" 200 || exit 1
echo "✅ health OK"

# 2. GET /
check_status "$API_URL/" 200 || exit 1
echo "✅ root OK"

# 3. OPTIONS /health
headers=$(curl -s -o /dev/null -D - -X OPTIONS "$API_URL/health")
cors=$(echo "$headers" | tr -d '\r' | grep -i '^Access-Control-Allow-Origin:' | awk '{print $2}')
if [ -z "$cors" ]; then
  echo "❌ Access-Control-Allow-Origin header missing" >&2
  exit 1
fi

frontend="${FRONTEND_URL:-}" # optional allowed origin
if [ "$cors" != "*" ]; then
  if [ -n "$frontend" ]; then
    echo "$cors" | tr ',' '\n' | grep -Fxq "$frontend" || { echo "❌ CORS header '$cors' does not allow '$frontend'" >&2; exit 1; }
  else
    echo "❌ CORS header '$cors' is not '*' and FRONTEND_URL not set" >&2
    exit 1
  fi
fi

echo "✅ CORS OK"
exit 0

