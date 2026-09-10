@echo off
REM Envolucro para a tarefa agendada. O schtasks nao define directorio de
REM trabalho, e passar caminhos com acentos numa unica linha de comando e
REM fragil; este ficheiro resolve as duas coisas.
REM
REM O task5b_resume.py e idempotente: se ja houver uma instancia viva sai em
REM silencio, e se estiver tudo feito funde os resultados e sai. Pode portanto
REM ser invocado sempre que a sessao arranca, sem verificacoes previas.
cd /d "%~dp0..\.."
python "python\scripts\task5b_resume.py" >> "outputs\task5b\resume_task.log" 2>&1
