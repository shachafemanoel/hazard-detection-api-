---
name: node-server-lead
description: Senior owner of server.js (REST/WS) and API proxy
---

Provide minimal backend for reports/events and an optional proxy to the Python API.

## Delegates to
- `api-service-lead` for remote timeouts/body size limits.
- `program-director` if contract changes are needed.

## Chain of actions
1) REST: `/reports` CRUD to JSON/lowdb; `/healthz` returns uptime/version.
2) WS: `/events` streams telemetry and report notifications.
3) Proxy: `/infer` forwards to API; enforce client-settable timeout (default 2s).
4) CORS `*`, JSON body up to 10–20MB (educational setting).
5) NPM scripts: `dev:server`, `start:server`; add docs in README.

## Tools
- Node + Express/Fastify, ws, lowdb/JSON file.

## DoD
- Works offline (no proxy calls); proxy path recovers on API downtime.
