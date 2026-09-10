#!/usr/bin/env python3
"""Executor retomavel da escada do segundo alvo (ArithmeticUtils).

Este alvo demora 1h30 a 2h por seed, pelo que as doze seeds sao mais de um dia
de relogio. O custo nao esta nos mutantes que esgotam o timeout, como se supos
a principio: sao 15 a 26 por braco, uns 80 segundos ao todo. Esta em cada braco
ter 2400 testes e cerca de 30 mutantes sobreviventes -- um mutante morto para no
primeiro teste que o mata, um sobrevivente obriga a correr a suite inteira, o que
da umas 70 mil execucoes de teste por braco.

Nao ha atalho seguro. Reduzir os 2400 testes tornaria estas seeds incomparaveis
com as primeiras quatro, e dar mais threads ao PIT aumenta a carga: como a
classificacao TIMED_OUT e sensivel ao tempo e o PIT conta os timeouts como
mortos, enviesaria as seeds novas contra as antigas.

Ao longo desta campanha o computador foi desligado
varias vezes a meio, e cada interrupcao custava a execucao inteira porque o
run_task5_experiment.py reescreve o CSV de saida a cada gravacao: recomecar
apontando ao mesmo ficheiro apagaria o que ja la estava.

A solucao e granularidade. Cada seed corre isolada para o seu proprio ficheiro
em outputs/task5b/seeds/, e o progresso passa a ser o conjunto de ficheiros que
existem. Retomar e simplesmente perguntar quais faltam. Uma seed interrompida a
meio deixa um ficheiro incompleto que e detectado (tem menos de quatro bracos) e
refeito do inicio; perde-se no maximo uma seed, nunca a campanha.

Os resultados ja obtidos por execucoes anteriores, em ficheiros com outro nome,
sao lidos na mesma: o que conta e a seed estar medida, nao onde ficou guardada.

Desenhado para ser invocado repetidamente sem cuidados -- por uma tarefa
agendada no arranque da sessao, por exemplo. Se ja houver uma instancia viva,
sai em silencio; se estiver tudo feito, funde e sai.

Uso:
    python python/scripts/task5b_resume.py
    python python/scripts/task5b_resume.py --status    # so relata, nao corre
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "outputs" / "task5b"
SEED_DIR = OUT_DIR / "seeds"
LOCK = OUT_DIR / "resume.lock"
FINAL = OUT_DIR / "results_arith_final.csv"
LOG = OUT_DIR / "resume.log"
LOG_DIR = OUT_DIR / "logs"

SEEDS = list(range(42, 54))          # 12 seeds, como no primeiro alvo
EPOCHS = 2000
N_ARMS = 4                            # WGAN, RANDOM, CORPUS, KDE


def _log(msg: str) -> None:
    linha = f"[{time.strftime('%d/%m %H:%M')}] {msg}"
    print(linha, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def _seeds_completas() -> dict[int, Path]:
    """Seeds com os quatro bracos medidos, em qualquer CSV de outputs/task5b.

    Varre tudo em vez de so os ficheiros por seed porque as primeiras quatro
    foram obtidas antes desta reestruturacao e nao ha razao para as repetir.
    """
    achadas: dict[int, Path] = {}
    for path in sorted(OUT_DIR.rglob("*.csv")):
        if path.name in {FINAL.name}:
            continue
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f)
                        if (r.get("seed") or "").strip() not in ("", "-")
                        and (r.get("killed") or "").strip()]
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
        for seed, n in Counter(r["seed"] for r in rows).items():
            if n >= N_ARMS:
                try:
                    achadas.setdefault(int(seed), path)
                except ValueError:
                    continue
    return achadas


def _ja_a_correr() -> bool:
    """Ha outra instancia viva? O ficheiro de lock guarda o PID."""
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text(encoding="utf-8").strip().splitlines()[0])
    except (ValueError, OSError, IndexError):
        LOCK.unlink(missing_ok=True)
        return False
    if pid == os.getpid():
        return False
    # Em Windows, tasklist e a forma sem dependencias de confirmar o PID. Nao
    # basta o PID existir: o Windows reutiliza-os, e um desligamento a meio
    # deixa o lock para tras com o PID de um processo ja morto. Se esse numero
    # for entretanto atribuido a outra coisa qualquer, um teste so pelo PID
    # concluiria para sempre que ha uma instancia viva -- e como isto e chamado
    # por uma tarefa periodica, a campanha ficaria bloqueada em silencio.
    # Exigir tambem que seja um python.exe torna a confusao muito improvavel.
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}",
                              "/FI", "IMAGENAME eq python.exe", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        if str(pid) in out and "python" in out.lower():
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    # Lock orfao de um processo que ja morreu.
    _log(f"lock orfao (pid {pid} nao esta vivo); a descartar")
    LOCK.unlink(missing_ok=True)
    return False


def _correr_seed(seed: int) -> bool:
    """Correr uma seed, com a saida do filho a ir para um log proprio.

    A saida vai para ficheiro em vez de ser capturada em memoria porque uma
    seed deste alvo demora perto de uma hora e capturar so mostra o resultado
    no fim: durante a execucao nao havia forma de saber em que braco ia, nem
    sequer se progredia. Com o log, um `tail` responde a pergunta.
    """
    destino = SEED_DIR / f"seed_{seed}.csv"
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_seed = LOG_DIR / f"seed_{seed}.log"
    _log(f"seed {seed}: a comecar (progresso em {log_seed.relative_to(REPO_ROOT)})")
    inicio = time.time()
    with open(log_seed, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(
            [sys.executable, "-u", str(SCRIPT_DIR / "run_task5_experiment.py"),
             "--target", "arith", "--base-seed", str(seed), "--seeds", "1",
             "--epochs", str(EPOCHS), "--out", str(destino)],
            cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
    dur = round(time.time() - inicio)
    if seed in _seeds_completas():
        _log(f"seed {seed}: concluida em {dur}s ({dur // 60} min)")
        return True
    cauda = ""
    try:
        cauda = io.open(log_seed, encoding="utf-8", errors="replace").read()[-300:].strip()
    except OSError:
        pass
    _log(f"seed {seed}: FALHOU em {dur}s (exit={proc.returncode}); {cauda}")
    return False


def _fundir(completas: dict[int, Path]) -> None:
    """Juntar as seeds da escada num ficheiro so, uma linha por braco e seed.

    So entram seeds de SEEDS. O varrimento e deliberadamente permissivo -- le
    qualquer CSV em outputs/task5b -- e apanha por isso medicoes que nao sao da
    escada, como a seed 99 do teste de codificacao. Servem para nao repetir
    trabalho, nunca para entrar na analise emparelhada.
    """
    vistos: set[tuple[str, str]] = set()
    linhas: list[dict] = []
    campos: list[str] = []
    for seed in sorted(s for s in completas if s in set(SEEDS)):
        path = completas[seed]
        with open(path, newline="", encoding="utf-8") as f:
            leitor = csv.DictReader(f)
            campos = campos or list(leitor.fieldnames or [])
            for r in leitor:
                chave = (r.get("arm", ""), (r.get("seed") or "").strip())
                if chave[1] != str(seed) or chave in vistos or not r.get("killed"):
                    continue
                vistos.add(chave)
                linhas.append(r)
    if not linhas:
        return
    with open(FINAL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)
    _log(f"fundido: {len(set(SEEDS) & set(completas))} seeds da escada, "
         f"{len(linhas)} medicoes -> {FINAL.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Executor retomavel do segundo alvo")
    ap.add_argument("--status", action="store_true",
                    help="Relatar o que falta e sair, sem correr nada")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completas = _seeds_completas()
    faltam = [s for s in SEEDS if s not in completas]

    if args.status:
        print(f"completas: {sorted(completas)}")
        print(f"faltam   : {faltam}")
        print(f"a correr : {'sim' if _ja_a_correr() else 'nao'}")
        return

    if _ja_a_correr():
        _log("ja ha uma instancia a correr; nada a fazer")
        return

    if not faltam:
        _fundir(completas)
        _log("todas as seeds completas")
        return

    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        _log(f"retomado: {len(completas)} feitas, {len(faltam)} por fazer {faltam}")
        for seed in faltam:
            # Reavaliar a cada volta: outra instancia pode ter avancado, e uma
            # seed que falhe nao deve impedir as seguintes.
            if seed in _seeds_completas():
                continue
            _correr_seed(seed)
        completas = _seeds_completas()
        _fundir(completas)
        restantes = [s for s in SEEDS if s not in completas]
        _log("campanha completa" if not restantes
             else f"ainda faltam {restantes} (serao retomadas na proxima execucao)")
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
