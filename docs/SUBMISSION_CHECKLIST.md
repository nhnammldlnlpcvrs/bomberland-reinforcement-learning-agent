# Submission Checklist

- [ ] `agent.py` exists at root of zip.
- [ ] `class Agent` exists.
- [ ] `__init__(agent_id)` exists.
- [ ] `act(obs)` returns int 0-5.
- [ ] `py_compile` passes.
- [ ] No banned imports.
- [ ] No network.
- [ ] No file writes in `act`.
- [ ] Startup under 20s.
- [ ] `act` under 100ms.
- [ ] Local match passes.
- [ ] Estimate ranking passes.
- [ ] Logs analyzed.
- [ ] Zip structure verified with `tar -tf submission.zip`.
- [ ] Changelog written.

## PowerShell Package Commands

```powershell
Remove-Item -Recurse -Force submission -ErrorAction SilentlyContinue
Remove-Item -Force submission.zip -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path submission
Copy-Item agent\hybrid_agent\agent.py submission\agent.py
Compress-Archive -Path submission\agent.py -DestinationPath submission.zip -Force
tar -tf submission.zip
```

