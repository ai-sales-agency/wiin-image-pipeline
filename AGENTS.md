# Repository agent rules

## Secrets

Read `SECRETS.md` before adding examples, fixtures, or credential-dependent
automation.

## WIIN Vaultwarden authority

Read SECRETS.md. All WIIN values are retrieved from the exact WIIN Vaultwarden
organization through bin/wiin-vault and personal/scoped service authentication.
Use WIIN_BITWARDEN_TOKEN or native OS credential storage. No age-key onboarding,
owner bootstrap in CI, another developer identity or old ciphertext fallback.
