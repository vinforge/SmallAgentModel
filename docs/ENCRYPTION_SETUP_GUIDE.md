# 🔐 SAM Encryption Setup Guide

**Complete Guide for Setting Up AES-256-GCM Encryption for New Users**

---

## 🎯 **Overview**

SAM uses enterprise-grade **AES-256-GCM encryption** to protect all your data. This guide walks you through setting up encryption for the first time.

### **🔒 Security Features**
- **AES-256-GCM**: Authenticated encryption with 256-bit keys
- **Argon2id**: Enterprise-grade password-based key derivation
- **Zero-Knowledge**: Your master password is never stored
- **Local Processing**: All encryption happens on your device

---

## 🚀 **Quick Setup (First-Time Users)**

### **Step 1: Launch SAM Secure**
```bash
cd SAM
python start_sam_secure.py --mode full
```

### **Step 2: First-Run Security Setup**
SAM will automatically detect this is your first time and guide you through setup:

```
🔐 SAM Secure Enclave Setup
───────────────────────────

Welcome to SAM! This is your first time running SAM with 
security enabled. You need to create a Master Password.

⚠️ IMPORTANT:
- Choose a strong password you'll remember
- This password cannot be recovered if lost
- All your SAM data will be encrypted with this password

🔑 Create Master Password
Password: ****************
Confirm:  ****************

🚀 Initialize SAM Secure Enclave
✅ Master password setup successful!
✅ Encryption keys generated
✅ Secure storage initialized

🎉 SAM is ready! Access points:
• Secure Chat: http://localhost:8502
• Memory Center: http://localhost:8501
```

### **Step 3: Access SAM**
- **Primary Interface**: http://localhost:8502 (Secure Streamlit)
- **Memory Center**: http://localhost:8501 (After authentication)

---

## 🔧 **Manual Setup Process**

If you need to set up encryption manually or understand the process:

### **What Happens During Setup**

1. **Master Password Creation**
   - You create a strong password (8+ characters recommended)
   - Password is used for key derivation only, never stored

2. **Salt Generation**
   - SAM generates a 128-bit cryptographically random salt
   - Salt is unique to your installation

3. **Key Derivation**
   - Uses Argon2id with enterprise parameters:
     - Time cost: 3 iterations
     - Memory cost: 64MB
     - Parallelism: 4 threads
     - Output: 256-bit encryption key

4. **Keystore Creation**
   - Creates `security/keystore.json` with:
     - Salt (for future key derivation)
     - Verifier hash (to check passwords)
     - Security configuration
     - Installation metadata

5. **Session Key Setup**
   - Derived key becomes your session key
   - Used for all AES-256-GCM encryption/decryption
   - Stored in memory only, never on disk

---

## 🔑 **Master Password Guidelines**

### **Requirements**
- ✅ **Minimum 8 characters** (12+ strongly recommended)
- ✅ **Mix of uppercase, lowercase, numbers, symbols**
- ✅ **Unique password** (don't reuse from other accounts)
- ⚠️ **Cannot be recovered** - choose carefully!

### **Good Examples**
```
MyS3cur3P@ssw0rd!2024
Tr0ub4dor&3_SAM_Key
C0ff33&Cr3am_Encrypt!
```

### **What to Avoid**
- ❌ Dictionary words only
- ❌ Personal information (birthdays, names)
- ❌ Common patterns (123456, password)
- ❌ Passwords used elsewhere

---

## 🛠️ **Advanced Setup Options**

### **Custom Security Parameters**
You can modify security parameters in `security/__init__.py`:

```python
SECURITY_CONFIG = {
    'argon2': {
        'time_cost': 3,        # Increase for more security
        'memory_cost': 65536,  # 64MB - increase for more security
        'parallelism': 4,      # Match your CPU cores
        'salt_length': 16,     # 128-bit salt
        'hash_length': 32      # 256-bit output
    },
    'session': {
        'timeout_seconds': 3600,  # 1 hour auto-lock
        'max_unlock_attempts': 5  # Lock after failed attempts
    }
}
```

### **Environment Variables**
```bash
# Optional configuration
export SAM_SESSION_TIMEOUT=7200      # 2 hour timeout
export SAM_MAX_UNLOCK_ATTEMPTS=3     # Stricter security
export SAM_KEYSTORE_PATH="custom/path/keystore.json"
```

---

## 🔄 **For Existing SAM Users**

If you have an existing SAM installation without encryption:

### **Migration Process**
```bash
# Run migration to encrypt existing data
python start_sam_secure.py --mode migrate
```

**Migration Steps:**
1. **Backup Creation** - Automatic backup of existing data
2. **Master Password Setup** - Create your encryption password
3. **Data Encryption** - Convert all data to encrypted format
4. **Verification** - Ensure migration completed successfully
5. **Cleanup** - Optional removal of unencrypted data

### **What Gets Encrypted**
- ✅ **All conversations** and chat history
- ✅ **Uploaded documents** and their content
- ✅ **Memory entries** and metadata
- ✅ **Vector embeddings** metadata
- ✅ **User preferences** and settings

---

## 🚨 **Important Security Notes**

### **⚠️ Password Recovery**
- **NO PASSWORD RECOVERY** - If you forget your master password, all encrypted data is permanently lost
- **Backup Strategy**: Consider keeping encrypted backups of important documents separately
- **Password Manager**: Use a password manager to securely store your master password

### **🔒 Session Security**
- Sessions automatically lock after 1 hour of inactivity
- Failed unlock attempts trigger temporary lockouts
- All encryption keys are cleared from memory when locked

### **📁 File Permissions**
SAM automatically sets secure file permissions:
- `security/keystore.json`: 600 (owner read/write only)
- `security/` directory: 700 (owner access only)
- Memory store files: 600 (owner read/write only)

---

## 🆘 **Troubleshooting**

### **Common Issues**

**1. "Keystore not found" Error**
```bash
# First run - this is normal
python start_sam_secure.py --mode full
```

**2. "Wrong password" Error**
- Double-check your master password
- Ensure caps lock is off
- Try typing slowly to avoid typos

**3. "Keystore corrupted" Error**
```bash
# Reset security (LOSES ALL ENCRYPTED DATA)
rm security/keystore.json
python start_sam_secure.py --mode full
```

**4. Permission Errors**
```bash
# Fix file permissions
chmod 700 security/
chmod 600 security/keystore.json
```

**5. Migration Issues**
```bash
# Force migration with backup
python start_sam_secure.py --mode migrate --force-reset
```

### **Getting Help**
- 📖 **Documentation**: Check `docs/README_SECURE_INSTALLATION.md`
- 🐛 **Issues**: Report problems with detailed error messages
- 📧 **Support**: Include log files from `logs/security.log`

---

## 📊 **Security Verification**

### **Check Your Setup**
After setup, verify your encryption is working:

1. **Upload a test document** in the secure interface
2. **Check the keystore exists**: `ls -la security/keystore.json`
3. **Verify file permissions**: Should show `-rw-------`
4. **Test session lock/unlock** with your master password

### **Security Audit**
```bash
# Check security status
python -c "
from security import SecureStateManager
sm = SecureStateManager()
print(f'Setup required: {sm.is_setup_required()}')
print(f'Currently locked: {sm.is_locked()}')
"
```

---

## 🎉 **You're All Set!**

Your SAM installation is now protected with enterprise-grade encryption:

- 🔐 **AES-256-GCM** encrypts all your data
- 🔑 **Argon2id** protects your master password
- 🛡️ **Zero-knowledge** design keeps you in control
- 🏠 **Local processing** ensures privacy

**Next Steps:**
1. Start using SAM at http://localhost:8502
2. Upload some documents to test encryption
3. Explore the Memory Center for advanced features
4. Set up regular encrypted backups

**Remember**: Keep your master password safe - it's the key to all your encrypted data!
