# Setup and portability

Use Python 3.12. Runtime dependency: Streamlit 1.65.0. The backend otherwise uses the standard library.

Create a fresh environment from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the shown local URL. The researcher screen seeds/selects learners and starts a session; the child screen then receives a backend-generated task. `COMPONENT2_DB_PATH` may override the default database for testing. Do not commit SQLite databases, virtual environments, Parquet data, joblib files, or student records.

Research reproduction is separate. Install `requirements-research.txt` only when the external frozen workspace is available. The copied Step 1–3 scripts are evidence snapshots: their original relative output paths are not bundled, and full reproduction requires the external public dataset and processed artifacts.

One genuine inference, without retraining:

```powershell
python research\assistments\public_model_demo.py --workspace "C:\path\to\ASSISTments Dataset"
```

The script verifies all three frozen artifact hashes and fails clearly if model files or required external Parquet inputs are missing or changed.
