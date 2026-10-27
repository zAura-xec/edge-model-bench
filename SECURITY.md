# Security

Do not include credentials, private model weights, internal hostnames, internal addresses, sensitive dataset examples, or exploit details in public issues.

Use GitHub's private vulnerability reporting on the repository's Security tab once maintainers have enabled it. If no private reporting channel has been configured, request one without disclosing the sensitive details. No public security email address is currently configured.

Hydra configurations can instantiate arbitrary Python targets. Treat configurations, runtime plugins and model artifacts as trusted code; inspect them before use. Vendor SDKs and model parsers have their own update and security requirements.

The release audit in `tools/audit_release.py` is a heuristic check, not a complete secret detector or a proof that publication is authorized. Maintainers should review the exported source and use an independent secret scanner before the first public release. If a real credential was previously committed, removing its current file does not revoke it or remove it from Git history; rotate it and handle the old history privately.

No stable release or long-term security support policy has been established yet.
