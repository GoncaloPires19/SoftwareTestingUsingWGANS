"""Quick analysis of continuation training results."""
import pandas as pd
import sys

history_path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/continuation_20260222_200918/training_history.csv'

df = pd.read_csv(history_path)

print('=== EVOLUTION OF MAX VALUES ===\n')
milestones = [10, 20, 30, 40, 50, 75, 100]
for e in milestones:
    row = df[df.epoch == e]
    if not row.empty:
        r = row.iloc[0]
        print(f'Epoch {e:3d}: Coverage={r.coverage:5.1f}% | '
              f'Max=[{r.max_methods:3.0f}, {r.max_branches:3.0f}, {r.max_statements:3.0f}]')

print(f'\n--- TREND ANALYSIS ---')
first = df.iloc[0]
last = df.iloc[-1]
print(f'Coverage:   {first.coverage:.1f}% → {last.coverage:.1f}% ({last.coverage-first.coverage:+.1f}%)')
print(f'Branches:   {first.max_branches:.0f} → {last.max_branches:.0f} ({last.max_branches-first.max_branches:+.0f})')
print(f'Statements: {first.max_statements:.0f} → {last.max_statements:.0f} ({last.max_statements-first.max_statements:+.0f})')

print(f'\n--- FINAL vs REAL MAXIMA ---')
real_max = [777, 1000, 1000]
print(f'Methods:    {last.max_methods:.0f} / {real_max[0]} = {last.max_methods/real_max[0]*100:.1f}%')
print(f'Branches:   {last.max_branches:.0f} / {real_max[1]} = {last.max_branches/real_max[1]*100:.1f}%')
print(f'Statements: {last.max_statements:.0f} / {real_max[2]} = {last.max_statements/real_max[2]*100:.1f}%')

# Best epochs for each metric
print(f'\n--- BEST EPOCHS ---')
best_coverage = df.loc[df.coverage.idxmax()]
best_branches = df.loc[df.max_branches.idxmax()]
best_statements = df.loc[df.max_statements.idxmax()]

print(f'Best Coverage:   Epoch {best_coverage.epoch:3.0f} with {best_coverage.coverage:.2f}%')
print(f'Best Branches:   Epoch {best_branches.epoch:3.0f} with {best_branches.max_branches:.0f}')
print(f'Best Statements: Epoch {best_statements.epoch:3.0f} with {best_statements.max_statements:.0f}')
