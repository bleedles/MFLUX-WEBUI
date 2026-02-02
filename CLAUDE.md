# Claude Code Instructions - MFLUX WebUI

## CRITICAL: Keep FEATURES_INDEX.md Updated

`FEATURES_INDEX.md` is the navigation system for this codebase. Update it whenever you change code structure.

---

## When to Update

Update `FEATURES_INDEX.md` if you:

- ✅ Add/remove/rename files
- ✅ Add/remove features
- ✅ Rename documented functions
- ✅ Change file locations
- ✅ Add environment variables
- ✅ Modify API endpoints
- ✅ Update architecture/workflow

**Don't update for**: Simple bug fixes that don't change structure

---

## What to Update

1. **Feature sections** - Backend/frontend files, key functions
2. **Frontend Components table** - Add/remove UI tabs
3. **File Organization Summary** - New/moved files
4. **Quick Reference guides** - If patterns change
5. **Environment Variables table** - New config
6. **Last Updated date** - Always update at bottom

---

## Quick Process

1. Search `FEATURES_INDEX.md` for affected features/files
2. Update all relevant sections
3. Verify file paths are correct
4. Update "Last Updated" date

---

## Example: Adding New Feature

```markdown
### New Feature Name
**Description**: What it does
**Backend Files**:
- backend/new_manager.py - Purpose (key_function_name)
**Frontend Files**:
- frontend/components/new_tab.py - UI (create_new_tab)
**Key Functions**:
- key_function_name() - Description
```

Then update Frontend Components table and File Organization Summary.

---

**Remember**: When in doubt, update it. Keep the index accurate for all AI agents.
