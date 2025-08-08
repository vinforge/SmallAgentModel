# 🔐 SAM Master Password Setup Guide

**Complete guide for new users to set up their Master Password after downloading SAM from GitHub**

---

## 📋 **Quick Start for New Users**

After downloading SAM from GitHub, you need to create a Master Password to encrypt your data. This is a **one-time setup** that secures all your SAM data.

### **🚀 Option 1: Automated Setup Script (Recommended)**

Run the automated setup script:

```bash
python setup_encryption.py
```

**What this script does:**
- ✅ Checks all security modules are installed
- ✅ Creates necessary security directories
- ✅ Guides you through Master Password creation
- ✅ Generates encryption keys automatically
- ✅ Tests the encryption system
- ✅ Provides clear next steps

**Follow the prompts:**
1. Press `Y` to continue with setup
2. Enter a strong Master Password (8+ characters)
3. Confirm your password
4. Wait for encryption setup to complete

---

## 📋 **Option 2: Manual Setup Through SAM Interface**

If you prefer to set up through the web interface:

1. **Start SAM:**
   ```bash
   python start_sam.py
   ```

2. **Open your browser:**
   - Go to `http://localhost:8502`

3. **Follow the setup wizard:**
   - SAM will automatically detect this is your first time
   - You'll see a "Master Password Setup" interface
   - Enter your password and confirm
   - Click "Setup Security"

---

## 🔑 **Master Password Requirements**

Your Master Password must meet these security requirements:

### **Minimum Requirements:**
- ✅ **At least 8 characters** (12+ strongly recommended)
- ✅ **Mix of uppercase and lowercase letters**
- ✅ **Include numbers and symbols**
- ✅ **Unique password** (don't reuse from other accounts)

### **⚠️ CRITICAL WARNINGS:**
- 🚨 **Cannot be recovered if lost!**
- 🚨 **No password reset option available**
- 🚨 **All your SAM data will be permanently inaccessible**
- 🚨 **Write it down and store it safely**

### **Good Password Examples:**
```
MyS@mP@ssw0rd2024!
Secure#SAM$Agent789
AI*Assistant&2024#
```

---

## 🔧 **Troubleshooting Setup Issues**

### **Problem: "Security modules not found"**
**Solution:**
```bash
# Install required dependencies
pip install -r requirements.txt

# Verify you're in the SAM directory
ls -la  # Should see setup_encryption.py
```

### **Problem: "Permission denied" errors**
**Solution:**
```bash
# Make script executable (Linux/Mac)
chmod +x setup_encryption.py

# Or run with python explicitly
python setup_encryption.py
```

### **Problem: "Encryption already set up"**
**Options:**
1. **Test existing password** - if you remember it
2. **Reset encryption** - will delete all encrypted data
3. **Skip setup** - if you want to use existing setup

---

## 🔄 **Resetting Your Master Password**

If you forgot your Master Password or want to start fresh:

### **⚠️ WARNING: This deletes all encrypted data!**

1. **Run the setup script:**
   ```bash
   python setup_encryption.py
   ```

2. **Choose option 2:** "Reset encryption"

3. **Type 'RESET' to confirm** (this cannot be undone)

4. **Create a new Master Password**

### **Alternative: Manual Reset**
```bash
# Backup and remove keystore
mv security/keystore.json security/keystore_backup.json

# Remove encrypted data
rm -rf memory_store/encrypted/

# Run setup again
python setup_encryption.py
```

---

## 🚀 **After Setup: Starting SAM**

Once your Master Password is set up:

### **Start SAM Securely:**
```bash
python start_sam_secure.py --mode full
```

### **Access SAM:**
- Open browser to `http://localhost:8502`
- Enter your Master Password when prompted
- Start using SAM!

### **Alternative Start Methods:**
```bash
# Simple start
python start_sam.py

# Advanced start with options
python secure_streamlit_app.py
```

---

## 📁 **What Gets Created During Setup**

The setup process creates these secure files:

```
security/
├── keystore.json          # Encrypted master keys
├── setup_status.json      # Setup completion status
└── entitlements.json      # Security permissions

memory_store/
└── encrypted/             # Encrypted memory storage
    └── chroma_db/         # Encrypted vector database
```

**🔒 All files are encrypted with your Master Password**

---

## 🛡️ **Security Best Practices**

### **Password Management:**
- ✅ Use a unique, strong password
- ✅ Store it in a password manager
- ✅ Write it down and store physically
- ❌ Don't share it with anyone
- ❌ Don't store it in plain text files

### **Backup Strategy:**
- ✅ Backup your `security/` folder regularly
- ✅ Test your Master Password periodically
- ✅ Keep backups in multiple secure locations
- ❌ Don't backup to cloud storage unencrypted

### **Access Control:**
- ✅ Lock your computer when away
- ✅ Use SAM on trusted devices only
- ✅ Log out of SAM when finished
- ❌ Don't leave SAM running unattended

---

## ❓ **Frequently Asked Questions**

### **Q: Can I change my Master Password later?**
A: Currently, you need to reset encryption (losing data) to change passwords. A password change feature is planned for future releases.

### **Q: What happens if I forget my Master Password?**
A: Unfortunately, there's no recovery option. You'll need to reset encryption and lose all encrypted data. This is by design for maximum security.

### **Q: Can I use SAM without a Master Password?**
A: No, the Master Password is required for SAM's security features. It encrypts all your conversations, memories, and uploaded documents.

### **Q: Is my Master Password stored anywhere?**
A: No, only a cryptographic hash is stored. The actual password never touches the disk and cannot be recovered from the hash.

### **Q: Can I run multiple SAM instances with different passwords?**
A: Yes, but they would need separate directories. Each SAM installation has its own encryption setup.

---

## 🆘 **Getting Help**

If you encounter issues during setup:

1. **Check the logs:**
   ```bash
   tail -f logs/secure_streamlit.log
   ```

2. **Run diagnostics:**
   ```bash
   python security_diagnostic.py
   ```

3. **Community Support:**
   - GitHub Issues: [Report problems](https://github.com/your-repo/sam/issues)
   - Documentation: [Full docs](https://docs.sam-ai.com)
   - Community Forum: [Get help](https://community.sam-ai.com)

---

## ✅ **Setup Complete Checklist**

After successful setup, verify:

- [ ] Master Password created and confirmed
- [ ] `security/keystore.json` file exists
- [ ] SAM starts without errors
- [ ] You can log in with your Master Password
- [ ] Encryption test passes
- [ ] You've backed up your password safely

**🎉 Congratulations! Your SAM installation is now secure and ready to use!**

---

*This guide ensures new GitHub users can quickly and securely set up their SAM Master Password for a fully encrypted AI assistant experience.*