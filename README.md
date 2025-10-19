# CTI-RAG

CTI-RAG is a Retrieval-Augmented Generation (RAG) system for Cyber Threat Intelligence analysis.  
It enables ingestion, storage, and analysis of CTI datasets with semantic search and chatbot capabilities.

---

## Project Structure

```
CTI_RAG/
├── chroma_db_v_rag/
│   └── ae721f7a-5a2d-4021-86a4-af37291ea417/
│       └── chroma.sqlite3
├── dataset/
│   ├── cve_data_raw.csv
│   ├── datadown.ipynb
│   ├── enterprise-attack-17.1.json
│   └── mitredat.ipynb
├── .gitignore
├── analyst_chatbot.py
├── rag_debugger.py
```
- **chroma_db_v_rag/**: Vector database files (auto-generated; you do not need to edit these).
- **dataset/**: Place all your CTI datasets here (CSV, JSON, etc.). This folder is ignored by git, so you must add your own files after cloning.
- **analyst_chatbot.py**: The analyst chatbot interface script (run second).
- **rag_debugger.py**: RAG pipeline setup and debugging script (run first).

---

## How to Clone and Set Up

1. **Clone the repository**
   ```bash
   git clone https://github.com/karash10/CTI-RAG.git
   cd CTI-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your dataset**
   Place your CTI dataset(s) in the `dataset/` folder.  
   Example: `dataset/cve_data_raw.csv`, `dataset/enterprise-attack-17.1.json`, etc.

   > The `dataset/` folder is ignored by git, so you must add your own files after cloning.

---

## Running the Project

**1. Step 1: Run the RAG Debugger**

```bash
python rag_debugger.py
```
- This script sets up the database and ingests your dataset.
- **On running, you will be prompted to enter your API key** (e.g., for OpenAI or other services).  
  Enter your API key to continue.

**2. Step 2: Run the Analyst Chatbot**

```bash
python analyst_chatbot.py
```
- This launches the chatbot interface for CTI analysis.
- You may be prompted for your API key again if required.

---

## Flow Summary

1. Clone the repository.
2. Install dependencies.
3. Place your dataset in the `dataset/` folder.
4. Run `rag_debugger.py` and enter your API key when prompted.
5. Run `analyst_chatbot.py` for interactive CTI analysis.

---

## Troubleshooting

- **Missing Dataset:** If you see errors about missing datasets, ensure your files are present in the `dataset/` folder.
- **API Key:** If prompted for an API key, make sure your key is valid for the required service.

---

## License

[Specify your license, e.g., MIT]

---

Created by [karash10](https://github.com/karash10)
