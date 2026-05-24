# LipTone Studio

Responsive virtual lipstick try-on app with:

- `frontend/`: Next.js mobile-first UI
- `backend/`: FastAPI image-processing API

## Structure

```text
lipstick_tryon/
├── backend/
│   └── app/
├── frontend/
│   ├── app/
│   └── lib/
├── face_landmarker.task
└── requirements.txt
```

## Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

API runs on `http://localhost:8000`.

## Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Frontend runs on `http://localhost:3000`.

## Frontend env

Set:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000
```
