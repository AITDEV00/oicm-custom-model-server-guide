# Licenses

License agreement bundles for OICM / OIK8S platform components.

Each `.zip` contains:
- `*.bundle` — an `oi_license:v1:` encrypted license bundle
- `password.txt` — the activation/decryption key for the bundle

| Zip | Product | Date | Password |
|-----|---------|------|----------|
| `OICM-20260831120924-V28Y02-license-agreement.zip` | OICM | 2026-08-31 | *(see `password.txt` inside zip)* |
| `OIK8S-20260831121602-8U5MTO-license-agreement.zip` | OIK8S | 2026-08-31 | *(see `password.txt` inside zip)* |

## Notes
- Bundle format header: `oi_license:v1:` followed by base64-encoded, encrypted payload.
- These are credential/activation files — keep them out of public or uncontrolled repositories.