# 🚀 Lite Setup Guide (For Low-End Servers)

This guide is optimized for servers with limited RAM (1GB-4GB) and no GPU. It reduces the installation size from **~3GB to ~400MB** using CPU-only libraries.

## 1. Quick Install (Recommended)

Run this single command to install everything efficiently:

```bash
pip install -r requirements.txt
```

> **Note:** If the above command fails or tries to download CUDA (large files), use this specific command for PyTorch first:

```bash
# 1. Install lightweight PyTorch (CPU only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the requirements
pip install -r requirements.txt

# 3. (Optional) Trim Library Bloat
# Removes tests, docs, and examples from installed libraries to save space.
python clean_install.py
```

## 2. Server Optimization Tips

### ✅ Enable Swap Memory (CRITICAL for 512MB RAM)
**WARNING:** On a 512MB server, the application **WILL CRASH** without swap memory. You must create at least a 4GB swap file.

```bash
# 1. Check if you already have swap
free -h

# 2. Create a 4GB swap file (Recommended for 512MB RAM)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 3. Make it permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 4. Verify
free -h
```

### ✅ Run with Memory Limits & Lite Mode
We have optimized the app to hold only ONE model in memory at a time. Run with these flags:

```bash
# Run app with specific server optimizations
streamlit run app.py --server.maxUploadSize 5 --global.developmentMode false
```

## 3. Deployment Command
For production (e.g., on Render, AWS EC2, or DigitalOcean), use this start command:

```bash
pip install -r requirements.txt && streamlit run app.py
```
