# security

this is a benchmark. it doesn't accept untrusted input or run user supplied code, so
the attack surface is small. a couple of things are still worth saying:

- api keys (anthropic, openai, brief) are read from the environment and never written
  to disk or committed. keep them in your own `.env`, which is gitignored.
- the brief seeder and the live arm talk to a real workspace over mcp. only point them
  at a workspace you own, and use a scoped token.
- results files can contain task text from the source datasets. treat them like the
  datasets themselves.

if you find something that looks like a genuine vulnerability, please open a private
github security advisory on this repo, or email the maintainers, rather than filing a
public issue. we'll get back to you.
