# WIIN secret handling

The sole active secret authority is the WIIN Vaultwarden organization at
https://pswd1.wiin.agency (`af4660b3-db52-4271-af19-28fefa368f83`).
`ai-sales-agency/secrets` owns the metadata manifest and provider runtime.

- Use exact logical IDs through `bin/wiin-vault with-file` or `run-env`.
- Authenticate with `WIIN_BITWARDEN_TOKEN` or personal native OS credential storage.
  Every developer uses their own account; every CI/runtime account has only its
  provider-enforced, reviewed collection grants.
- Never print/log/commit a secret, token, session, master password or private key,
  or pass it in command arguments. Child consumers receive selected values only.
- `WIIN_BITWARDEN_TOKEN` is the only persisted CI bootstrap input. Application
  values are materialized from Vaultwarden; Kubernetes and job environments are
  scoped runtime copies, not independent secret authorities.
- An API key alone cannot unlock Password Manager. Personal setup uses the
  interactive vault login and native OS credential manager; service credentials
  include only that service account's own login/unlock material.
- Do not consume old age ciphertexts, repository-local keys stores or the retired
  WIIN central-store paths as a fallback. New and rotated values are written and
  verified in Vaultwarden under the owning repository's authorized workflow.
- Backup/result encryption remains mandatory. The independent encrypted emergency
  recovery kit restores Vaultwarden during an outage; it is not a development
  secret provider.
