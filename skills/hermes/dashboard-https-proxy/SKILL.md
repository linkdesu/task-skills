---
name: dashboard-https-proxy
description: Set up Caddy HTTPS reverse proxy for Hermes Dashboard with WebSocket support, HTTP→HTTPS redirect, and wildcard hostname access.
type: prompt
whenToUse: When you need HTTPS access to Hermes Dashboard from LAN devices, or WebSocket connections fail with code=1006 from remote devices, or you need HTTP→HTTPS redirect with wildcard hostname support for LAN IPs / Tailscale IPs.
---

# Dashboard HTTPS Proxy (Caddy)

Set up Caddy reverse proxy to expose Hermes Dashboard with HTTPS, WebSocket support, HTTP→HTTPS redirect, and LAN/Tailscale hostname access.

## Key pitfalls

1. **HTTP/2 blocks WebSocket** — browsers use HTTP/2 by default, which doesn't support WebSocket Upgrade. Must disable HTTP/2 on the proxy port: `servers :9119 { protocols h1 }`
2. **Origin header rejection** — dashboard bound to `127.0.0.1` rejects WebSocket connections with non-loopback Origin. Caddy must strip the Origin header: `header_up Origin ""`
3. **Host header** — dashboard checks Host header against its bound address. Caddy must rewrite Host: `header_up Host 127.0.0.1:19119`
4. **Certificate SAN** — self-signed cert must include the LAN IP as a Subject Alt Name, otherwise browser TLS rejection prevents WebSocket connection
5. **LibreSSL on macOS** — doesn't support `-addext` flag. Must use config file approach for SANs
6. **Caddyfile paths** — `~` is not expanded by Caddy; use absolute paths

## Steps

### 1. Generate certificate with SANs

Create a config file and generate:

```bash
HERMES_HOME="${HOME}/.hermes"
mkdir -p "$HERMES_HOME/certs"

cat > "$HERMES_HOME/certs/san.cnf" << 'EOF'
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = *.local

[v3_req]
subjectAltName = DNS:localhost,DNS:*.local,DNS:hermes-dashboard.local,IP:127.0.0.1,IP:0.0.0.0,IP:<YOUR_LAN_IP>
EOF

openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
  -keyout "$HERMES_HOME/certs/dashboard.key" \
  -out "$HERMES_HOME/certs/dashboard.crt" \
  -config "$HERMES_HOME/certs/san.cnf"

rm "$HERMES_HOME/certs/san.cnf"
```

> **Note**: Replace `<YOUR_LAN_IP>` with your machine's actual LAN IP (e.g. `192.168.1.100`), or omit it if you only need localhost access. If using Tailscale, also add your Tailscale IP.

### 2. Write Caddyfile

```caddy
{
    # HTTPS server: HTTP/1.1 only for WebSocket compatibility
    servers :9119 {
        protocols h1
    }
}

# HTTP → HTTPS redirect (port 9080)
:9080 {
    redir https://{host}:9119{uri} 301
}

# HTTPS reverse proxy (port 9119)
:9119 {
    tls /Users/<USER>/.hermes/certs/dashboard.crt /Users/<USER>/.hermes/certs/dashboard.key
    reverse_proxy 127.0.0.1:19119 {
        header_up Host 127.0.0.1:19119
        header_up Origin ""
    }
}
```

Save this file to `~/.hermes/Caddyfile` (note: use absolute paths, `~` does not work in Caddyfile).

### 3. Remove `dashboard.public_url` config

```bash
hermes config set dashboard.public_url ''
```

This lets the dashboard auto-detect the URL from Caddy's X-Forwarded headers.

### 4. Start scripts

Create `~/.hermes/bin/dashboard-start`:

```bash
#!/bin/bash
set -e
HERMES_HOME="${XDG_HOME_HOME:-$HOME}/.hermes"
CADDYFILE="$HERMES_HOME/Caddyfile"
DASHBOARD_PORT=19119
CADDY_PORT=9119
HTTP_REDIRECT_PORT=9080

echo "==> Stopping any previous instances..."
kill $(lsof -ti:"$DASHBOARD_PORT","$CADDY_PORT","$HTTP_REDIRECT_PORT") 2>/dev/null || true
sleep 1

echo "==> Starting Hermes Dashboard (internal port $DASHBOARD_PORT)..."
hermes dashboard --host 127.0.0.1 --port "$DASHBOARD_PORT" --skip-build --no-open &
DASHBOARD_PID=$!

echo "==> Waiting for dashboard to be ready..."
for i in $(seq 1 10); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$DASHBOARD_PORT/" 2>/dev/null | grep -q 200; then
        echo "==> Dashboard ready (PID $DASHBOARD_PID)"
        break
    fi
    sleep 1
done

echo "==> Starting Caddy reverse proxy with TLS (port $CADDY_PORT)..."
caddy run --config "$CADDYFILE" &
CADDY_PID=$!

sleep 1

echo ""
echo "✅ All services running:"
echo "   Dashboard (HTTP):  http://127.0.0.1:$DASHBOARD_PORT/"
echo "   Caddy HTTPS:       https://0.0.0.0:$CADDY_PORT/"
echo "   HTTP→HTTPS:       http://0.0.0.0:$HTTP_REDIRECT_PORT/ → https://...:$CADDY_PORT/"
echo ""
echo "   LAN access:        https://<YOUR_LAN_IP>:$CADDY_PORT/"
echo "   Stop with:         dashboard-stop"
```

Create `~/.hermes/bin/dashboard-stop`:

```bash
#!/bin/bash
echo "==> Stopping Hermes Dashboard and Caddy..."
hermes dashboard --stop 2>/dev/null
CADDY_PID=$(lsof -ti:9119 2>/dev/null)
if [ -n "$CADDY_PID" ]; then
    echo "==> Stopping Caddy (PID $CADDY_PID)..."
    kill "$CADDY_PID" 2>/dev/null
fi
echo "✅ Stopped."
```

Make them executable:

```bash
chmod +x ~/.hermes/bin/dashboard-{start,stop}
```

### 5. Verify

```bash
curl -sk https://127.0.0.1:9119/           # HTTPS works → 200
curl -sv http://127.0.0.1:9080/ | grep 301  # HTTP redirect → 301
curl -sk --resolve 'test.local:9119:127.0.0.1' https://test.local:9119/  # Wildcard → 200
```

For WebSocket, extract token from HTML and test with curl:

```bash
TOKEN=$(curl -sk https://127.0.0.1:9119/ | sed -n 's/.*__HERMES_SESSION_TOKEN__="\([^"]*\)".*/\1/p')
curl -sk -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Origin: https://<YOUR_LAN_IP>:9119" \
  "https://127.0.0.1:9119/api/events?channel=test&token=$TOKEN"
```

You should see WebSocket data streaming back. If you get `code=1006`, double-check the pitfalls above.

## Files created

- `~/.hermes/Caddyfile` — Caddy config
- `~/.hermes/certs/dashboard.crt` — self-signed certificate (10-year, with SANs)
- `~/.hermes/certs/dashboard.key` — private key
- `~/.hermes/bin/dashboard-start` — start script (executable)
- `~/.hermes/bin/dashboard-stop` — stop script (executable)
