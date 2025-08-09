# Service Integration

To integrate the Hazard Detection API into your service, configure the base URL.

For detailed client examples and response handling, see the [Client Guide](CLIENT_GUIDE.md).

## Base URL

- Production: `https://hazard-api-production-production.up.railway.app`
- Development: `http://localhost:8080`

Set the `HAZARD_API_URL` environment variable to point your service to the appropriate instance:

```bash
export HAZARD_API_URL=https://hazard-api-production-production.up.railway.app
# For local development
# export HAZARD_API_URL=http://localhost:8080
```

Use this variable when making requests, for example:

```bash
curl -X GET "$HAZARD_API_URL/health"
```

This allows different environments to target the correct API endpoint.
