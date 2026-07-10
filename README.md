# ModE

Project repository for methods for model-based design of energy systems.

## Python imports

The canonical Python package lives in `Erdem/src`, so the `Erdem` directory must
be on `PYTHONPATH` for imports such as `from src.optimization.core import ...`.

In VS Code this is configured through `.vscode/settings.json`. For a standalone
PowerShell session from the repository root, use:

```powershell
$env:PYTHONPATH="$PWD\Erdem;$env:PYTHONPATH"
python path\to\script.py
```
