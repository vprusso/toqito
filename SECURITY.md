# Security Policy

## Supported versions

`toqito` is a research library released as a rolling latest version on PyPI.
Security fixes are applied to the current release; there is no long-term support
for older versions. Please upgrade to the latest release before reporting an
issue.

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not through a public issue or
pull request. Use GitHub's private vulnerability reporting for this repository:

- Go to the **Security** tab and choose **Report a vulnerability**
  (https://github.com/vprusso/toqito/security/advisories/new).

Include enough detail to reproduce the problem: affected version, a minimal
example, and the observed versus expected behavior. You can expect an
acknowledgement within a few days. If a fix is warranted, it will be prepared
privately and released before the advisory is published.

## Scope

`toqito` is a numerical library. It performs computations on arrays and numbers
and has no network, authentication, or persistence layer, so the realistic
surface is limited to:

- correctness defects that a caller could be induced to rely on for a security
  decision,
- the supply chain (dependencies and CI), and
- code executed at documentation-build or test time.

Reports about any of these are welcome. General bug reports that are not security
sensitive should go to the public issue tracker instead.
