# Client (Frontend)

React frontend for Fake Review Detection System.

## Status

⏳ **Coming Soon** - Ready to build React UI

## What Will Be Built

- Dashboard with statistics
- Single review analysis form
- Batch upload interface
- Prediction history
- Analytics charts
- Model information page
- Mobile responsive design
- Dark mode support

## Setup (When Ready)

```bash
cd client
npm install
npm start
```

## API Connection

Frontend will connect to backend API at:
```
http://localhost:8000/api/v1/*
```

## Technologies

- React 18
- TypeScript
- Tailwind CSS or Material-UI
- Axios for API calls
- React Query for data management
- Chart.js for analytics

## Folder Structure (To Be Created)

```
client/
├── src/
│   ├── components/
│   │   ├── PredictionForm.tsx
│   │   ├── ResultCard.tsx
│   │   ├── History.tsx
│   │   └── Dashboard.tsx
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── Analytics.tsx
│   │   └── ModelInfo.tsx
│   ├── services/
│   │   └── api.ts
│   ├── App.tsx
│   └── index.tsx
├── package.json
├── tsconfig.json
└── README.md
```

---

**Backend API is ready!** Start the server and test with Postman.

When ready, React UI will connect to the backend.

📡 Backend: `../server/`
🎨 Frontend: (To be built)
