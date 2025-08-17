# SAM Engine Upgrade Framework - End-to-End Test Plan

## Overview

This document outlines the comprehensive end-to-end test plan for the SAM Engine Upgrade Framework. The test plan validates the entire lifecycle of the feature, including normal operation, edge cases, and failure recovery scenarios.

**Test Environment Requirements:**
- SAM installation with Memory Control Center
- At least 2 different model engines available for download
- Sufficient disk space for model downloads and backups
- Test user account with clean state

## Test Case 1: The Happy Path

**Objective:** Validate the complete successful engine upgrade journey with all features enabled.

### Pre-conditions:
- SAM is running with default DeepSeek engine
- No existing LoRA adapters
- Empty knowledge base
- Core Engines tab shows available models

### Test Steps:

#### Step 1: Baseline Setup
1. **Action:** Launch SAM and verify default engine
   - **Expected:** Main UI shows DeepSeek as active engine
   - **Verification:** Check Core Engines tab shows "Active" status for DeepSeek

2. **Action:** Add test documents to knowledge base
   - **Expected:** Documents are ingested and embedded successfully
   - **Verification:** Search functionality works with added documents

3. **Action:** Create test conversation data for DPO training
   - **Expected:** Preference data is collected
   - **Verification:** Personalized Tuner shows available training data

#### Step 2: Train Initial LoRA Adapter
4. **Action:** Navigate to Personalized Tuner and start DPO training
   - **Expected:** Training job starts successfully
   - **Verification:** Training progress is visible and completes

5. **Action:** Activate the trained LoRA adapter
   - **Expected:** Adapter loads and becomes active
   - **Verification:** Personalized Tuner shows "Active" adapter status

6. **Action:** Test personalized responses
   - **Expected:** Responses show personalization effects
   - **Verification:** Response quality differs from base model

#### Step 3: Download Alternative Engine
7. **Action:** Navigate to Core Engines tab
   - **Expected:** Available models are displayed with download options
   - **Verification:** Model catalog shows Llama 3.1 8B as available

8. **Action:** Download Llama 3.1 8B model
   - **Expected:** Download starts with progress indicator
   - **Verification:** Download completes successfully, status shows "Downloaded"

#### Step 4: Execute Engine Migration
9. **Action:** Click "Activate" on downloaded Llama model
   - **Expected:** Migration wizard launches
   - **Verification:** Step 1 warning screen appears

10. **Action:** Proceed through wizard with all options enabled
    - **Step 1:** Accept backup warning → Continue
    - **Step 2:** Select "Re-embed knowledge base" → Continue  
    - **Step 3:** Select "Update prompt templates" → Continue
    - **Step 4:** Confirm migration → Start Migration
    - **Expected:** Each step shows appropriate options and warnings
    - **Verification:** Final confirmation shows migration summary

11. **Action:** Execute migration
    - **Expected:** Migration progress screen shows real-time updates
    - **Verification:** Migration completes successfully (100% progress)

#### Step 5: Validate Post-Migration State
12. **Action:** Verify engine switch
    - **Expected:** Core Engines shows Llama as "Active", DeepSeek as "Downloaded"
    - **Verification:** Main UI indicates Llama engine is active

13. **Action:** Check LoRA adapter status
    - **Expected:** Personalized Tuner shows old adapter as "Archived"
    - **Verification:** UI prompts for new adapter training

14. **Action:** Test knowledge base functionality
    - **Expected:** Search still works (may show re-embedding progress)
    - **Verification:** Documents are retrievable with new embeddings

15. **Action:** Train new LoRA adapter on Llama engine
    - **Expected:** New training job starts successfully
    - **Verification:** New adapter trains and activates on Llama engine

#### Step 6: Final Validation
16. **Action:** Test complete system functionality
    - **Expected:** All features work with new engine
    - **Verification:** Chat, search, and personalization all functional

**Success Criteria:**
- ✅ Migration completes without errors
- ✅ Old LoRA adapter is properly archived
- ✅ New engine is active and functional
- ✅ Knowledge base re-embedding completes
- ✅ New LoRA adapter can be trained on new engine
- ✅ All UI components reflect new engine state

---

## Test Case 2: The Cautious User Path

**Objective:** Validate engine upgrade with minimal options (no re-embedding).

### Pre-conditions:
- SAM running with populated knowledge base
- Active LoRA adapter on current engine
- Alternative engine downloaded

### Test Steps:

#### Step 1: Conservative Migration
1. **Action:** Start migration wizard
2. **Action:** Proceed with these options:
   - **Step 1:** Accept backup → Continue
   - **Step 2:** Select "Skip re-embedding" → Continue
   - **Step 3:** Select "Keep current prompts" → Continue
   - **Step 4:** Confirm migration → Start Migration

#### Step 2: Validate Degraded Performance
3. **Action:** Test knowledge base search
   - **Expected:** Search works but may show reduced quality
   - **Verification:** Old embeddings are used with new engine

4. **Action:** Verify RAG functionality
   - **Expected:** Document retrieval still functions
   - **Verification:** System handles embedding/engine mismatch gracefully

**Success Criteria:**
- ✅ Migration completes with minimal options
- ✅ System remains functional with degraded search
- ✅ User can manually trigger re-embedding later

---

## Test Case 3: The Rollback Path

**Objective:** Validate failure recovery and rollback capabilities.

### Pre-conditions:
- SAM running with active LoRA adapter
- Corrupted or invalid model file prepared for testing

### Test Steps:

#### Step 1: Simulate Migration Failure
1. **Action:** Replace downloaded model with corrupted file
2. **Action:** Attempt migration to corrupted model
   - **Expected:** Migration fails during model loading
   - **Verification:** Error message is clear and helpful

#### Step 2: Validate Rollback
3. **Action:** Verify system state after failure
   - **Expected:** Original engine remains active
   - **Verification:** LoRA adapter is restored and functional

4. **Action:** Check backup integrity
   - **Expected:** Backup files are intact and accessible
   - **Verification:** System can restore from backup if needed

#### Step 3: Recovery Testing
5. **Action:** Fix corrupted model and retry migration
   - **Expected:** Migration succeeds after fixing issue
   - **Verification:** System recovers and completes migration

**Success Criteria:**
- ✅ Failed migration doesn't break system
- ✅ Rollback restores original state
- ✅ Recovery is possible after fixing issues
- ✅ Error messages are clear and actionable

---

## Performance Benchmarks

### Migration Performance Metrics
- **Migration Controller Switch Time:** < 30 seconds
- **Model Loading Time:** < 2 minutes per model
- **Backup Creation Time:** < 1 minute
- **Configuration Update Time:** < 5 seconds

### Resource Usage During Re-embedding
- **CPU Usage:** Monitor during background re-embedding
- **Memory Usage:** Track peak memory consumption
- **Disk I/O:** Measure read/write operations
- **Network Usage:** Monitor model download bandwidth

### User Experience Metrics
- **Wizard Completion Time:** < 5 minutes for experienced users
- **Error Recovery Time:** < 2 minutes to restore from failure
- **UI Response Time:** < 1 second for status updates

---

## Test Execution Checklist

### Pre-Test Setup
- [ ] Clean SAM installation
- [ ] Test data prepared
- [ ] Alternative models available
- [ ] Monitoring tools configured

### Test Execution
- [ ] Test Case 1: Happy Path - PASS/FAIL
- [ ] Test Case 2: Cautious User Path - PASS/FAIL  
- [ ] Test Case 3: Rollback Path - PASS/FAIL
- [ ] Performance benchmarks captured
- [ ] Resource usage documented

### Post-Test Validation
- [ ] All UI components engine-aware
- [ ] Error handling comprehensive
- [ ] Documentation updated
- [ ] Performance within acceptable limits

---

## Risk Assessment

### High Risk Areas
1. **Data Loss:** LoRA adapter invalidation
   - **Mitigation:** Comprehensive backup system
2. **System Instability:** Failed migrations
   - **Mitigation:** Rollback capabilities
3. **Performance Degradation:** Large re-embedding tasks
   - **Mitigation:** Background processing and progress tracking

### Medium Risk Areas
1. **UI Inconsistency:** Engine state not reflected
   - **Mitigation:** Comprehensive UI review
2. **Resource Exhaustion:** Multiple concurrent operations
   - **Mitigation:** Resource monitoring and limits

### Low Risk Areas
1. **Configuration Corruption:** Invalid settings
   - **Mitigation:** Configuration validation
2. **Network Issues:** Model download failures
   - **Mitigation:** Resume capability and error handling

---

## Success Criteria Summary

The Engine Upgrade Framework is considered ready for release when:

1. **All test cases pass** without critical failures
2. **Performance benchmarks** meet acceptable thresholds
3. **UI components** are fully engine-aware
4. **Error handling** is comprehensive and user-friendly
5. **Documentation** is complete and accurate
6. **Rollback capabilities** work reliably
7. **Resource usage** is within reasonable limits

**Go/No-Go Decision Factors:**
- Critical bugs: 0 (blocking)
- High severity bugs: < 2 (review required)
- Performance degradation: < 20% vs baseline
- User experience issues: < 3 (must be minor)
