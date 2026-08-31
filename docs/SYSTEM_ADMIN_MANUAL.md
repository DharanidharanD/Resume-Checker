# System Administrator & User Operations Manual

## TalentMatrix AI™: Enterprise Resume Screening & Candidate Classification System

---

### 1. System Requirements

- **Operating System**: Windows 10/11, macOS 12+, or Ubuntu 20.04+ Linux
- **Python**: Version 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14
- **RAM**: Minimum 4 GB (8 GB Recommended)
- **Disk Space**: Minimum 1 GB available disk space

---

### 2. Installation & Setup

1. **Navigate to the Project Directory**:
   ```bash
   cd C:\Users\My\.gemini\antigravity\scratch\resume-screening-nlp
   ```

2. **Install Required Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Database & Train ML Models**:
   ```bash
   python scripts/train_pipeline.py
   ```

---

### 3. Launching the Software Application

#### Option A: Launch Interactive Web Application (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
Open your browser at: `http://localhost:8501`

#### Option B: Launch FastAPI REST API Server
```bash
uvicorn src.api.app:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

#### Option C: One-Click Desktop Launcher
Double-click `run_application.bat` or run:
```bash
python launch.py
```

---

### 4. Database Operations & Backups

- The primary database file is located at: `data/talentmatrix.db`
- **Backup**: Simply copy `data/talentmatrix.db` to a backup storage location.
- **Reset Database**: Delete `data/talentmatrix.db` and re-run the application; it will automatically re-create tables and seed default requisitions.

---

### 5. Running Automated Verification Tests
```bash
pytest -v tests/
```
