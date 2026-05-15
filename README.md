# Chef Mshia Recipe App

AI-powered recipe assistant project that helps users chat about recipes, save favorites, and keep chat history.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Current Features](#current-features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Run the Project](#run-the-project)
- [API and Backend Notes](#api-and-backend-notes)
- [Data Storage](#data-storage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

Chef Mshia is a recipe-focused chatbot application.  
The backend handles chat requests, calls the Groq API for AI responses, and stores user data like chat history and favorites.

## Project Structure

```text
mshia/
  backend/
    app.py
    requirements.txt
    README.md
    data/
      chat_hist.json
      favorites.json
      user_memory.json
  templates/
  README.md
```

## Current Features

- Conversational recipe assistant via Groq API
- Session-based user flow
- Save and load chat history
- Save and delete favorite recipes
- Lightweight user memory persistence

## Tech Stack

- Python
- Flask
- Flask-CORS
- Requests
- Python-Dotenv
- HTML templates (server-rendered pages)

## Getting Started

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies from `backend/requirements.txt`.
4. Add required environment variables.
5. Run the Flask backend.

### Install Dependencies

From project root:

```bash
python -m venv .venv
```

Activate virtual environment:

- Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

- macOS/Linux

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r backend/requirements.txt
```

## Environment Variables

Create `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=replace_with_a_secure_random_value
```

- `GROQ_API_KEY` is required for AI responses.
- `SECRET_KEY` is used for Flask sessions and should be strong in production.

## Run the Project

From project root:

```bash
python backend/app.py
```

Default app URL:

`http://127.0.0.1:5000`

## API and Backend Notes

- Backend routes and detailed backend documentation are in `backend/README.md`.
- Main backend entry point is `backend/app.py`.

## Data Storage

Current storage is JSON-file based inside `backend/data/`:

- `chat_hist.json`
- `favorites.json`
- `user_memory.json`

For production, consider replacing file storage with a database.

## Roadmap

- Improve prompt and response quality for recipe tasks
- Add stronger validation and error handling
- Add automated tests
- Add production deployment configuration

## Contributing

1. Create a feature branch.
2. Keep changes focused and small.
3. Test locally before opening a pull request.
4. Update docs when behavior changes.

## License

Add your chosen license here (for example, MIT).
