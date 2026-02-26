# VoxProof Frontend

A premium, futuristic UI for the VoxProof AI Voice Detection Platform.

## Tech Stack

- **React 18**
- **Vite 5**
- **TypeScript**
- **React Router 6**
- **Tailwind CSS**
- **Framer Motion**
- **Lucide Icons**

## Features

- 🌓 Dark/Light theme with smooth transitions
- 🎨 Glassmorphism & neon accent design
- 🎬 Cinematic Framer Motion animations
- 📱 Fully responsive design
- 🔊 Drag & drop audio upload
- 📊 Animated result visualization

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Edit .env.local with your API URL
# VITE_API_BASE_URL=https://your-backend-url.onrender.com
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm run preview  # Preview production build
```

## Environment Variables

| Variable            | Description            | Default                 |
| ------------------- | ---------------------- | ----------------------- |
| `VITE_API_BASE_URL` | Backend API URL        | `http://localhost:8000` |
| `VITE_API_KEY`      | API authentication key | -                       |

## Deployment

### Vercel / Netlify / Any Static Host

1. Push code to GitHub
2. Import project
3. Set build command: `npm run build`
4. Set output directory: `dist`
5. Add environment variables:
   - `VITE_API_BASE_URL`: Your Render backend URL
   - `VITE_API_KEY`: Your API key
6. Deploy

```bash
# Or build locally and deploy dist folder
npm run build
```

## Project Structure

```
frontend/
├── src/
│   ├── main.tsx                # Entry point
│   ├── App.tsx                 # Root component with routes
│   ├── pages/
│   │   ├── Home.tsx            # Landing page
│   │   ├── Dashboard.tsx       # Upload & analysis
│   │   └── About.tsx           # Team & info
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Navbar.tsx      # Navigation
│   │   │   ├── Footer.tsx      # Footer
│   │   │   └── Background.tsx  # Animated background
│   │   ├── providers/
│   │   │   └── ThemeProvider.tsx
│   │   └── ui/
│   │       └── ThemeToggle.tsx
│   └── app/
│       └── globals.css         # Global styles
├── public/                     # Static assets
├── index.html                  # HTML entry point
├── vite.config.ts              # Vite configuration
├── tailwind.config.ts
└── package.json
```

## API Integration

The frontend connects to the FastAPI backend at `/api/voice-detection`:

```typescript
const response = await fetch(
  `${import.meta.env.VITE_API_BASE_URL}/api/voice-detection`,
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": import.meta.env.VITE_API_KEY,
    },
    body: JSON.stringify({
      language: "English",
      audioFormat: "mp3",
      audioBase64: base64EncodedAudio,
    }),
  },
);
```

## License

MIT License
