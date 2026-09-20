@echo off
REM Regime de poucos dados no segundo alvo: ~18 entradas de treino.
REM
REM O numero e absoluto e nao uma fraccao herdada do primeiro alvo: a hipotese
REM que se testa e sobre contagem de entradas (inversao perto das 30), e o
REM corpus do segundo alvo tem 260 contra 121. As mesmas fraccoes dariam 39
REM entradas, ja acima do cruzamento, e falhariam o fenomeno.
REM
REM Idempotente: sai em silencio se ja houver instancia viva, funde e sai se
REM estiver tudo feito. Pode ser invocado por tarefa periodica.
cd /d "%~dp0..\.."
python "python\scripts\task5b_resume.py" --entradas 18 >> "outputs\task5b\frac18_task.log" 2>&1
