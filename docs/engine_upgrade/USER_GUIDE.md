# SAM Engine Upgrade Framework - User Guide

## Overview

The SAM Engine Upgrade Framework allows you to seamlessly switch between different AI model engines while preserving your data and personalization. This guide walks you through the complete process of upgrading your SAM engine.

## 🎯 What is Engine Upgrade?

Engine Upgrade lets you:
- **Switch AI Models**: Move from DeepSeek to Llama, Qwen, or other supported models
- **Preserve Data**: Keep your knowledge base and conversation history
- **Maintain Personalization**: Retrain personalized models for the new engine
- **Rollback if Needed**: Restore previous engine if issues occur

## 🚀 Getting Started

### Prerequisites

- SAM Memory Control Center running
- At least 10GB free disk space
- Stable internet connection for model downloads

### Step 1: Access Core Engines

**Option A: Full SAM Application (Recommended)**
1. Launch SAM with: `python start_sam_secure.py --mode full`
2. Open SAM at `http://localhost:8502`
3. Authenticate with your master password
4. Navigate to **Memory Control Center**
5. Select **🔧 Core Engines** from the dropdown menu

**Option B: Memory Control Center Only**
1. Launch with: `streamlit run ui/memory_app.py`
2. Open at `http://localhost:8501`
3. Navigate to **🔧 Core Engines** from the dropdown menu

### Step 2: Download an Alternative Engine

1. In the **📚 Model Library** tab, browse available models
2. Look for models marked as "Recommended" for best compatibility
3. Click **Download** on your chosen model
4. Wait for download to complete (may take 10-30 minutes depending on model size)

### Step 3: Activate the New Engine

1. Once downloaded, click **Activate** on your new model
2. The **Migration Wizard** will launch with 5 steps:

#### Step 1: Warning & Backup
- **Review the warning** about LoRA adapter invalidation
- **Choose backup option** (recommended: keep enabled)
- Click **Continue**

#### Step 2: Re-embedding Option
- **Choose re-embedding** (recommended for best search quality)
- Or skip if you want faster migration
- Click **Continue**

#### Step 3: Prompt Templates
- **Update prompts** for optimal performance with new engine
- Or keep current templates
- Click **Continue**

#### Step 4: Final Confirmation
- **Review migration summary**
- **Check the confirmation box**
- Click **🚀 Start Migration**

#### Step 5: Migration Progress
- **Monitor progress** in real-time
- **Wait for completion** (usually 5-15 minutes)
- Click **🎉 Finish** when done

## 📋 What Happens During Migration?

### Automatic Actions
1. **Backup Creation**: Your LoRA adapters are safely backed up
2. **Configuration Update**: SAM switches to the new engine
3. **LoRA Invalidation**: Old adapters are marked as incompatible
4. **Re-embedding** (if selected): Knowledge base is re-embedded in background
5. **Prompt Optimization** (if selected): Templates updated for new engine

### Your Data Safety
- ✅ **Knowledge Base**: Preserved and optionally re-embedded
- ✅ **Conversation History**: Fully preserved
- ✅ **Preference Data**: Available for retraining personalized models
- ✅ **LoRA Adapters**: Backed up and can be restored

## 🧠 Retraining Personalized Models

After engine upgrade, you'll need to retrain personalized models:

1. Go to **🧠 Personalized Tuner**
2. Check the **Engine Compatibility Warning** at the top
3. Your old preference data is still available
4. Start new training with the **🎯 Training Controls** tab
5. Activate the new personalized model when training completes

## ⚠️ Important Considerations

### Before Upgrading
- **Backup Important Data**: While automatic backups are created, consider manual backups
- **Check Disk Space**: Ensure sufficient space for new model and backups
- **Plan Downtime**: Migration may take 15-30 minutes
- **Test Thoroughly**: Validate functionality after upgrade

### Engine Compatibility
- **LoRA Adapters**: Must be retrained for each engine
- **Knowledge Base**: May need re-embedding for optimal search
- **Response Style**: Different engines have different personalities
- **Performance**: Speed and quality may vary between engines

### Rollback Process
If you need to rollback:
1. Go to **🔧 Core Engines**
2. Find your previous engine in downloaded models
3. Click **Activate** to switch back
4. Follow the migration wizard again

## 🔧 Troubleshooting

### Common Issues

#### Migration Fails
- **Check disk space**: Ensure sufficient free space
- **Verify model integrity**: Re-download if corrupted
- **Check logs**: Look in `logs/` directory for error details
- **Try again**: Many issues resolve on retry

#### Poor Search Quality After Migration
- **Re-embed knowledge base**: Use Settings tab in Core Engines
- **Wait for completion**: Re-embedding runs in background
- **Check progress**: Monitor in Downloads tab

#### Personalized Models Not Working
- **Check compatibility**: Look for warnings in Personalized Tuner
- **Retrain models**: Use existing preference data
- **Verify activation**: Ensure new model is activated

#### UI Shows Wrong Engine
- **Refresh page**: Browser cache may be stale
- **Check configuration**: Verify in Core Engines tab
- **Restart SAM**: If issues persist

### Getting Help

1. **Check System Health**: Use 🔧 System Health tab
2. **Review Logs**: Check `logs/` directory for errors
3. **Benchmark Performance**: Run `scripts/benchmark_engine_upgrade.py`
4. **Reset if Needed**: Restore from backup and try again

## 📊 Performance Expectations

### Migration Times
- **Small Setup** (no LoRA, no re-embedding): 2-5 minutes
- **Medium Setup** (with LoRA, skip re-embedding): 5-10 minutes
- **Full Setup** (with LoRA and re-embedding): 15-30 minutes

### Resource Usage
- **CPU**: 20-60% during migration
- **Memory**: 2-8GB additional during process
- **Disk**: 5-15GB for new model and backups
- **Network**: 2-8GB for model download

### Quality Impact
- **Response Quality**: May improve or change style
- **Search Accuracy**: Temporarily reduced until re-embedding completes
- **Personalization**: Lost until retrained (data preserved)
- **Speed**: Varies by engine and hardware

## 🎉 Best Practices

### Planning Your Upgrade
1. **Choose the right time**: Low usage periods
2. **Prepare users**: Inform about temporary changes
3. **Test first**: Try with non-critical data
4. **Have rollback plan**: Know how to revert

### Optimizing Results
1. **Enable all options**: Backup, re-embedding, prompt updates
2. **Monitor progress**: Watch for errors or issues
3. **Test thoroughly**: Validate all functionality
4. **Retrain quickly**: Start personalization training soon after

### Maintaining Your Setup
1. **Regular backups**: Beyond automatic migration backups
2. **Monitor performance**: Use benchmarking tools
3. **Keep models updated**: Download newer versions when available
4. **Clean up old models**: Remove unused engines to save space

## 📚 Additional Resources

- **Developer Guide**: `docs/engine_upgrade/DEVELOPER_GUIDE.md`
- **API Reference**: `docs/engine_upgrade/API_REFERENCE.md`
- **Troubleshooting**: `docs/engine_upgrade/TROUBLESHOOTING.md`
- **Performance Benchmarks**: `benchmarks/` directory
- **Test Plans**: `tests/plans/engine_upgrade_e2e_plan.md`

---

**Need Help?** Check the troubleshooting section above or review the system logs for detailed error information.
