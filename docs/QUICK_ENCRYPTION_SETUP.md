# 🔐 SAM Quick Encryption Setup Guide

**5-minute guide to enable enterprise-grade encryption in SAM**

## ⚡ **Quick Start**

### **Option 1: Automatic Setup (Recommended)**
```bash
python setup.py
# Choose option 4: Encryption Only Setup
```

### **Option 2: Direct Script**
```bash
python setup_encryption.py
```

### **Option 3: During First Launch**
```bash
python start_sam_secure.py --mode full
# Follow encryption prompts
```

## 🔑 **Master Password Creation**

### **When prompted, create a strong master password:**

#### **Requirements:**
- ✅ Minimum 8 characters (12+ recommended)
- ✅ Mix of letters, numbers, and symbols
- ✅ Something you'll remember
- ❌ Cannot be recovered if lost

#### **Example Strong Passwords:**
- `MySecure2024!SAM`
- `AI#Memory$2024`
- `SAM-Secure-Key-123`

### **Password Prompt:**
```
🔐 SAM Secure Enclave Setup
───────────────────────────

🔑 Create Master Password
Password: ****************
Confirm:  ****************

✅ Master password setup successful!
```

## 🛡️ **What Gets Encrypted**

### **Protected Data:**
- ✅ All conversation history
- ✅ Uploaded documents
- ✅ Memory vectors and embeddings
- ✅ User preferences and settings
- ✅ API keys and sensitive configuration

### **Encryption Details:**
- **Algorithm:** AES-256-GCM
- **Key Derivation:** Argon2id
- **Salt:** Unique per installation
- **Authentication:** Built-in integrity verification

## 🚀 **Quick Verification**

### **Test Encryption is Working:**

1. **Start SAM:**
   ```bash
   python start_sam_secure.py --mode full
   ```

2. **Enter your master password when prompted**

3. **Look for these indicators:**
   ```
   🔐 Secure Enclave Status: UNLOCKED
   🛡️ Encryption: ACTIVE
   ✅ Memory Store: ENCRYPTED
   ```

4. **Access secure interface:** http://localhost:8502

## 🔧 **Troubleshooting**

### **Common Issues:**

#### **"Master password setup failed"**
```bash
# Ensure security directory exists
mkdir -p security

# Try setup again
python setup_encryption.py
```

#### **"Encryption modules not found"**
```bash
# Install required dependencies
pip install cryptography argon2-cffi

# Try setup again
```

#### **"Cannot unlock secure enclave"**
- Double-check your master password
- Ensure you're using the same password you created
- Password is case-sensitive

#### **"Security setup required"**
- This is normal for first-time setup
- Follow the master password creation prompts
- Encryption will be configured automatically

## 📋 **Quick Commands Reference**

### **Setup Encryption:**
```bash
python setup_encryption.py
```

### **Start with Encryption:**
```bash
python start_sam_secure.py --mode full
```

### **Check Encryption Status:**
```bash
python -c "from security import SecureStateManager; print('Setup required:', SecureStateManager().is_setup_required())"
```

### **Reset Encryption (⚠️ Destroys all data):**
```bash
rm -rf security/keystore.json
rm -rf memory_store/encrypted/
python setup_encryption.py
```

## 🎯 **Next Steps After Setup**

### **1. Start SAM Securely:**
```bash
python start_sam_secure.py --mode full
```

### **2. Access Secure Interface:**
- Open: http://localhost:8502
- Enter your master password
- Start using SAM with full encryption

### **3. Upload Documents:**
- All uploads are automatically encrypted
- Documents are processed securely
- Memory storage is protected

### **4. Configure API Keys:**
- Access Memory Center for API key management
- All keys are encrypted at rest
- Secure configuration interface

## 🔒 **Security Best Practices**

### **Master Password:**
- ✅ Use a unique, strong password
- ✅ Store it securely (password manager)
- ✅ Don't share with others
- ❌ Don't write it down in plain text

### **System Security:**
- ✅ Keep SAM updated
- ✅ Use secure network connections
- ✅ Regular backups (encrypted)
- ✅ Monitor access logs

### **Data Protection:**
- ✅ All data encrypted at rest
- ✅ Secure memory handling
- ✅ Automatic key rotation
- ✅ Integrity verification

## 📚 **Additional Resources**

- **Complete Guide:** `docs/ENCRYPTION_SETUP_GUIDE.md`
- **Security Architecture:** `docs/SECURITY_ARCHITECTURE.md`
- **Troubleshooting:** `docs/TROUBLESHOOTING.md`

## ⚡ **Summary**

**Encryption setup in 3 steps:**
1. Run `python setup_encryption.py`
2. Create a strong master password
3. Start SAM with `python start_sam_secure.py --mode full`

**Your data is now protected with enterprise-grade encryption!** 🛡️

---

**Need help?** Check the complete encryption guide or visit the GitHub repository for support.
