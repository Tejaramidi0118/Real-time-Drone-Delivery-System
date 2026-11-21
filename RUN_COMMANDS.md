# Run the simulator

Tested on Windows with Python 3.14, Node.js 22.14 and npm. Run the following commands from the repository root in two separate PowerShell terminals.

## Terminal 1 — Flask API

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flask --app backend/app.py run --host 127.0.0.1 --port 5002
```

Check [backend health](http://127.0.0.1:5002/health) for `{"status":"ok"}`. On macOS or Linux, create the environment with `python3.14 -m venv .venv` and use `.venv/bin/python` for the two subsequent commands.

## Terminal 2 — React dashboard

```powershell
cd frontend
npm ci
npm run dev -- --host localhost --port 5173 --strictPort
```

Open [the dashboard](http://localhost:5173/) and select **Try a sample flight**. Order 2 should be delivered first, followed by order 1 and a return-to-base event. Leave both terminals running during the flight. Use Ctrl+C in each to stop.

## Build check

From the frontend directory, run `npm run build`. Vite writes the compiled site to `dist/`. The dashboard still needs the Flask API for map data and route planning.

## Troubleshooting

- If the site is unreachable, check that Vite is listening on port 5173.
- If the page opens but cannot load its base, check the Flask health endpoint and confirm port 5002 is available.
- Keep the bundled GeoJSON files in `backend/data/`. The sample uses these files. External map imagery needs a network connection.
