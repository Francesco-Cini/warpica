# Security policy

## Supported versions

Only the latest release receives security fixes. See
[docs/policy.md](docs/policy.md).

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
reporting instead: on the repository page choose **Security**, then **Report a
vulnerability**.

Include what you found, how to reproduce it, and the affected version. You can
expect an acknowledgement within a week and an update on the fix or decision
within a month. This is a volunteer-run project, so these are goals rather than
guarantees.

## Scope

`warplab` is a numerical library: it reads no network input and runs no servers.
Realistic problems are unsafe handling of crafted input (for example, a metric
dictionary that crashes the interpreter or allocates unbounded memory) and
problems in the build and release process. Wrong physical results are ordinary
bugs; report them as normal issues.
