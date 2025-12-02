# Quick Start Guide

## Option 1: Demo Mode (No Backend Required)

Just run the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Login with:
- Username: `demo`
- Password: `demo`

## Option 2: Full Setup with Backend

### Step 1: Install Dependencies

```bash
pip install -r requirements-streamlit.txt
```

For full backend functionality:
```bash
pip install -r requirements.txt
```

### Step 2: Setup Database (Optional for full backend)

```bash
# Create .env file
copy .env.example .env

# Edit .env and set your database URL
# DATABASE_URL=postgresql://user:password@localhost:5432/ai_platform

# Run migrations
alembic upgrade head
```

### Step 3: Create First User (Backend Mode)

Start Python and run:

```python
from app.db.connection import get_db
from app.db.crud import user_crud
from app.api.middleware.auth import hash_password
import asyncio

async def create_admin():
    async for db in get_db():
        user_data = {
            "username": "admin",
            "email": "admin@example.com",
            "hashed_password": hash_password("admin123"),
            "full_name": "Admin User",
            "role": "admin",
            "is_active": True
        }
        user = await user_crud.create(db, user_data)
        await db.commit()
        print(f"Created user: {user.email}")
        break

asyncio.run(create_admin())
```

### Step 4: Start Backend

```bash
uvicorn app.main:app --reload
```

Or double-click `run_backend.bat`

Backend will be at: http://localhost:8000
API Docs: http://localhost:8000/docs

### Step 5: Start Dashboard

```bash
streamlit run streamlit_app.py
```

Or double-click `run_dashboard.bat`

Dashboard will be at: http://localhost:8501

Login with:
- Email: `admin@example.com`
- Password: `admin123`

## Troubleshooting

### Backend not connecting
- Make sure the backend is running on port 8000
- Check if database is configured correctly
- Use demo mode if you just want to see the UI

### Dependency conflicts
- Use `requirements-streamlit.txt` for minimal installation
- Or use: `pip install --use-deprecated=legacy-resolver -r requirements.txt`
