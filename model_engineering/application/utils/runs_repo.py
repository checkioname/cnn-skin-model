import os
import subprocess
import shutil
from pathlib import Path


class RunsRepo:
    """Gerencia versionamento de checkpoints/logs em um repositório Git LFS separado.
    
    Uso:
        runs_repo = RunsRepo("/caminho/psoriasis-runs")
        runs_repo.init_if_needed()
        save_path = runs_repo.run_dir("vgg16")
        # ... treino salva em save_path ...
        runs_repo.commit_push("feat: vgg16 50 épocas, acc 94.2%")
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self._git = shutil.which("git")
        self._lfs_installed = False

    @property
    def is_cloned(self) -> bool:
        return (self.repo_path / ".git").exists()

    @property
    def exists(self) -> bool:
        return self.repo_path.exists()

    def init_if_needed(self, remote_url: str | None = None):
        """Clona ou inicializa o repositório de runs."""
        if self.is_cloned:
            return

        if not self.exists and remote_url:
            self._run(["git", "clone", remote_url, str(self.repo_path)])
        elif not self.exists:
            self.repo_path.mkdir(parents=True, exist_ok=True)
            self._run(["git", "init"], cwd=self.repo_path)

        self._setup_lfs()

    def _setup_lfs(self):
        if self._lfs_installed:
            return
        if shutil.which("git-lfs"):
            self._run(["git", "lfs", "track", "*.pt"], cwd=self.repo_path)
            self._run(["git", "lfs", "track", "*.pth"], cwd=self.repo_path)
            self._run(["git", "lfs", "track", "*.pkl"], cwd=self.repo_path)
            self._run(["git", "add", ".gitattributes"], cwd=self.repo_path)
            self._lfs_installed = True

    def run_dir(self, model_name: str, timestamp: int | None = None) -> str:
        """Cria e retorna o caminho do diretório para uma run."""
        import time
        ts = timestamp or int(time.time())
        path = self.repo_path / f"ml-model-{model_name}-{ts}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def commit_push(self, message: str, push: bool = True):
        """Faz commit e push automático dos resultados."""
        self._run(["git", "add", "-A"], cwd=self.repo_path)

        status = self._run(["git", "status", "--porcelain"], cwd=self.repo_path,
                           capture_output=True)
        if not status.strip():
            return

        self._run(["git", "commit", "-m", message], cwd=self.repo_path)
        if push:
            self._run(["git", "push"], cwd=self.repo_path)

    def _run(self, cmd, cwd=None, capture_output=False):
        try:
            result = subprocess.run(
                cmd, cwd=cwd or self.repo_path,
                capture_output=capture_output, text=True, check=True
            )
            return result.stdout if capture_output else ""
        except subprocess.CalledProcessError as e:
            print(f"[RunsRepo] Erro executando {' '.join(cmd)}: {e}")
            if e.stderr:
                print(e.stderr)
            return ""
