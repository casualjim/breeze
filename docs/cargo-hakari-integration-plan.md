# Cargo-Hakari Integration Plan

## Phase 1: Workspace Preparation

### Step 1.1: Update Cargo Resolver
- Update root `Cargo.toml` to use resolver = "3" for better performance
- Update `.config/hakari.toml` after creation to match resolver = "3"

## Phase 2: Initialize Workspace-Hack

### Step 2.1: Choose Workspace-Hack Location
- Create `workspace-hack` directory at workspace root
- Initialize with descriptive name: `breeze-workspace-hack`

### Step 2.2: Initialize cargo-hakari
```bash
# Initialize workspace-hack crate
cargo hakari init workspace-hack

# This creates:
# - workspace-hack/Cargo.toml
# - workspace-hack/src/lib.rs
# - .config/hakari.toml
```

### Step 2.3: Configure hakari.toml
- Update `.config/hakari.toml` with:
  - `hakari-package = "workspace-hack"`
  - `resolver = "3"`
  - Add common development platforms:
    - `"x86_64-unknown-linux-gnu"`
    - `"x86_64-apple-darwin"`
    - `"aarch64-apple-darwin"`
    - `"x86_64-pc-windows-msvc"`

## Phase 3: Generate and Integrate

### Step 3.1: Generate Workspace-Hack Dependencies
```bash
# Generate workspace-hack Cargo.toml with all unified dependencies
cargo hakari generate

# Review generated Cargo.toml for correctness
cat workspace-hack/Cargo.toml
```

### Step 3.2: Verify Feature Unification
```bash
# Check that dependencies are properly unified
cargo hakari verify

# Explain why specific dependencies are included
cargo hakari explain <dependency-name>
```

### Step 3.3: Add Workspace-Hack to All Crates
```bash
# Add workspace-hack as dependency to all workspace crates
cargo hakari manage-deps

# This will add to all Cargo.toml files:
# workspace-hack = { path = "../workspace-hack" }
```

### Step 3.4: Update Root Cargo.toml
- Add `workspace-hack` to workspace members list
- Ensure proper ordering in the members array

## Phase 4: CI/CD Integration

### Step 4.1: Update GitHub Actions
- Modify `.github/workflows/CI.yml` to include:
```yaml
- name: Check workspace-hack is up-to-date
  run: |
    cargo hakari generate --diff
    cargo hakari manage-deps --dry-run
```

### Step 4.2: Update Development Scripts
- Add to `.mise/tasks/build`:
```bash
# Ensure workspace-hack is current before builds
cargo hakari generate --quiet
```

### Step 4.3: Update README and Documentation
- Document workspace-hack in development setup
- Explain impact on build times
- Include troubleshooting steps

## Phase 5: Testing and Validation

### Step 5.1: Performance Benchmarking
```bash
# Time builds before and after
time cargo build --features local-embeddings
time cargo check --features local-embeddings

# Test incremental builds
touch crates/breeze/src/main.rs
time cargo check --features local-embeddings
```

### Step 5.2: Comprehensive Testing
```bash
# Run ALL tests (Rust, Python, and Node.js) using mise
mise test
```

## Phase 6: Special Considerations

### Step 6.1: Address Feature-Flag Dependencies
- Verify `local-embeddings` feature works correctly
- Test conditional compilation scenarios
- Ensure no feature leakage between crates

### Step 6.2: Update Development Workflows
- Update any scripts that clean/build the workspace
- Consider adding `cargo hakari disable` for debugging
- Document performance expectations

## Phase 7: Rollback and Maintenance

### Step 7.1: Create Rollback Plan
```bash
# To temporarily disable workspace-hack:
cargo hakari disable

# To completely remove:
cargo hakari remove-deps
rm -rf workspace-hack
```

### Step 7.2: Establish Maintenance Procedures
- Run `cargo hakari generate` after adding dependencies
- Update CI to fail if workspace-hack is out of date
- Monitor build time improvements

### Step 7.3: Long-term Monitoring
- Track build time metrics in CI
- Monitor for dependency changes that affect unification
- Update cargo-hakari version periodically

## Expected Benefits for This Project

Given the complexity of this workspace with 7 crates, multiple language bindings, and performance-critical code:

- **Incremental builds**: 10-50x faster for small changes
- **Full builds**: 1.5-2x faster overall
- **CI time reduction**: Significant savings on clean builds
- **Developer experience**: Much faster `cargo check` cycles

## Rollback Procedures

If you need to temporarily disable workspace-hack for debugging or encounter issues:

### Temporary Disable
```bash
# Temporarily disable workspace-hack (reversible)
cargo hakari disable
```

### Complete Removal
```bash
# Remove workspace-hack dependency from all crates
cargo hakari remove-deps

# Delete workspace-hack directory and config
rm -rf workspace-hack
rm .config/hakari.toml

# Remove workspace-hack from root Cargo.toml members
# Edit Cargo.toml and remove "workspace-hack" from members array
```

### Re-enable After Disable
```bash
# Re-enable workspace-hack after temporary disable
cargo hakari generate
cargo hakari manage-deps
```

## Important: Package Manager Usage

**CRITICAL**: This repository uses **bun exclusively** as the package manager. Never use npm, yarn, or pnpm. All Node.js dependencies and scripts must use `bun` commands:

- Use `bun install` instead of `npm install`
- Use `bun run` instead of `npm run`
- Use `bun test` instead of `npm test`

The `mise test` command will handle running tests across all languages (Rust, Python, Node.js) in this monorepo.

## Maintenance Procedures

### Keeping Workspace-Hack Up-to-Date

**After adding new dependencies:**
```bash
# Regenerate workspace-hack with new dependencies
cargo hakari generate

# Update all crates with new workspace-hack version
cargo hakari manage-deps
```

**Before commits (automated in CI):**
- CI automatically runs `cargo hakari generate --quiet` and `cargo hakari manage-deps --dry-run`
- If CI fails, run the commands above and commit the changes

### Development Workflow Integration

- **mise build** automatically syncs workspace-hack before building
- **CI jobs** automatically verify workspace-hack is up-to-date
- **Local development** benefits from faster incremental builds immediately

### Monitoring Performance

Track build time improvements:
```bash
# Measure incremental build performance
touch crates/breeze/src/main.rs
time cargo check --features local-embeddings

# Full build performance
time cargo build --features local-embeddings
```

### Troubleshooting

**Build errors after dependency changes:**
1. Run `cargo hakari generate` to update workspace-hack
2. Run `cargo hakari manage-deps` to update all crates
3. Clean and rebuild: `cargo clean && cargo build`

**Workspace-hack verification fails:**
1. Check `cargo hakari verify` for conflicts
2. Review `cargo hakari explain <dependency>` for specific issues
3. Consider excluding problematic dependencies in `.config/hakari.toml`